"""整帧单图引擎：把一帧 decode（16 深度步 + 采样 + frame embedding + trunk 步进）
捕获为一条自包含 CUDA Graph，每帧一次 replay。

自包含设计（无图外算子参与 replay）：
- prefix 每帧在图内清零；codes 写入静态 [Q] 缓冲；
- 位置由静态 pos 标量在图内推进（pos.add_(1)）；mask / position_ids /
  cache_position 全部由 pos 在图内即时计算；
- CFG 合并权重、temperature、q0 的 EOS 屏蔽标志为静态缓冲，可在帧间由
  CPU 改写（replay 读取最新值），min_new_frames 逻辑不在图内分支；
- trunk 用 StaticCache（B=2，cond/uncond 左填充对齐）+ dict 静态加性 mask
  （跳过 HF 动态 mask 构建）。

当前实现 greedy（argmax，无 RNG，可逐位校验）；采样 multinomial 预留接口。
"""

from __future__ import annotations

import torch

from . import tilelang_kernels as TLK


class GraphFrameEngine:
    def __init__(
        self,
        model,
        *,
        device: torch.device,
        max_cache_len: int,
        cfg_scale: float = 1.5,
        batch: int = 2,
    ) -> None:
        from transformers import StaticCache

        self.model = model
        self.config = model.config
        self.device = device
        self.batch = batch
        self.table = model.audio_codebooks.table
        self.scale = float(model.audio_codebooks.scale)
        self.depth = model.depth_head
        self.d_model = int(model.d_model)
        self.Q = int(self.config.num_codebooks)
        self.V = int(self.config.audio_vocab_size)
        self.dtype = next(model.parameters()).dtype

        self.cache = StaticCache(
            config=model.trunk.config,
            max_batch_size=batch,
            max_cache_len=max_cache_len,
            device=device,
            dtype=self.dtype,
        )
        self.max_len = int(max_cache_len)

        # 静态缓冲
        self.hidden = torch.zeros(batch, self.d_model, device=device, dtype=self.dtype)
        self.prefix = torch.zeros(batch, self.d_model, device=device, dtype=self.dtype)
        self.codes = torch.zeros(self.Q, device=device, dtype=torch.long)
        self.pos = torch.zeros(1, device=device, dtype=torch.long)
        self.offsets = torch.zeros(batch, 1, device=device, dtype=torch.long)
        self.pad_len = torch.zeros(1, device=device, dtype=torch.long)
        self.temp_q0 = torch.ones(1, device=device, dtype=torch.float32)
        self.temp = torch.ones(1, device=device, dtype=torch.float32)
        self.mask_eos_q0 = torch.zeros(1, device=device, dtype=torch.int32)
        self.mask_eos_const = torch.ones(1, device=device, dtype=torch.int32)
        self.sample_flag = torch.zeros(1, device=device, dtype=torch.int32)
        self.cfg = torch.tensor([cfg_scale], device=device, dtype=torch.float32)
        self.arange = torch.arange(self.max_len, device=device)
        # 采样参数在捕获时烘焙进图（top_k/top_p 改变会触发重新捕获）
        self.top_k: int | None = None
        self.top_p: float | None = None
        self._captured_sampling: tuple[int | None, float | None] | None = None
        self.rng = torch.Generator(device=device)

        finfo_eps = torch.finfo(torch.float32).eps
        self._eps_in = self.depth.norm_in.eps or finfo_eps
        self._eps_out = self.depth.norm_out.eps or finfo_eps
        self._kernels: dict = {}
        self.graph: torch.cuda.CUDAGraph | None = None
        self.embeds = torch.zeros(
            batch, 1, self.d_model, device=device, dtype=self.dtype
        )

    # ------------------------------------------------------------------ 工具
    def _kernel(self, name, factory, *params):
        key = f"{name}:{'x'.join(map(str, params))}"
        kernel = self._kernels.get(key)
        if kernel is None:
            kernel = factory(*params)
            self._kernels[key] = kernel
        return kernel

    def _build_kernels(self) -> None:
        d, B, V = self.d_model, self.batch, self.V
        inner = int(self.depth.mlp.gate_proj.out_features)
        self.k1 = self._kernel(
            "add_rmsnorm", TLK.add_rmsnorm, B, d, float(self._eps_in)
        )
        self.k2 = self._kernel("swiglu", TLK.swiglu_gemv, B, inner, d)
        self.k3 = self._kernel("down_residual", TLK.down_residual_gemv, B, d, inner)
        self.k4 = self._kernel("rmsnorm", TLK.rmsnorm, B, d, float(self._eps_out))
        self.k5 = self._kernel("classifier", TLK.classifier_gemv, B, V, d)
        self.k6 = self._kernel(
            "frame_embed", TLK.frame_embed, 1, self.Q, V, d, self.scale
        )
        self.k7 = self._kernel("prefix_add", TLK.prefix_add, B, V, d, self.scale)
        self.k8 = self._kernel(
            "sampling_prep",
            TLK.sampling_prep,
            V,
            self.config.audio_bos_id,
            self.config.audio_pad_id,
            self.config.audio_eos_id,
        )

    # ------------------------------------------------------------------ prefill
    @torch.inference_mode()
    def prefill(
        self,
        cond_ids: torch.LongTensor,
        uncond_ids: torch.LongTensor | None,
        prompt_codes: torch.LongTensor | None,
    ) -> torch.Tensor:
        """eager 预填（含 uncond 左填充对齐），返回 [B, d] 末位 hidden。"""
        Lc = cond_ids.shape[1]
        Lu = uncond_ids.shape[1] if uncond_ids is not None else Lc
        if Lu > Lc:
            raise ValueError("uncond 文本长于 cond")
        P = Lc - Lu
        ids = cond_ids.new_zeros(self.batch, Lc)
        ids[0] = cond_ids[0]
        if uncond_ids is not None:
            ids[1, P:] = uncond_ids[0]
        else:
            ids[1] = cond_ids[0]
        text_embeds = self.model.trunk.get_input_embeddings()(ids)

        bos = torch.full(
            (1, 1, self.Q),
            self.config.audio_bos_id,
            dtype=torch.long,
            device=self.device,
        )
        audio_inputs = bos
        if prompt_codes is not None and prompt_codes.numel():
            prompt = prompt_codes.to(self.device).view(1, -1, self.Q)
            audio_inputs = torch.cat((bos, prompt), dim=1)
        La = audio_inputs.shape[1]
        audio_embeds = self.model.audio_codebooks.frame_embedding(
            audio_inputs.repeat(self.batch, 1, 1)
        )
        embeds = torch.cat((text_embeds, audio_embeds.to(text_embeds.dtype)), dim=1)
        L = embeds.shape[1]
        if L > self.max_len:
            raise ValueError(f"prefill {L} 超过 max_cache_len={self.max_len}")

        # 手工 causal+pad 加性 mask：[B,1,L,max_len]（dict 传入，跳过 HF 动态构建；
        # StaticCache 按 max_len 预分配，L 之外的槽位必须 -inf）
        causal = torch.triu(
            torch.full((L, L), float("-inf"), device=self.device, dtype=self.dtype),
            diagonal=1,
        )
        mask = torch.full(
            (self.batch, 1, L, self.max_len),
            float("-inf"),
            device=self.device,
            dtype=self.dtype,
        )
        mask[..., :L] = causal
        if P:
            mask[1, :, :, :P] = float("-inf")
        position_ids = (
            torch.arange(L, device=self.device).unsqueeze(0).repeat(self.batch, 1)
        )
        if P:
            position_ids[1] = torch.cat(
                (
                    torch.zeros(P, dtype=torch.long, device=self.device),
                    torch.arange(Lu, device=self.device),
                    Lu + torch.arange(La, device=self.device),
                )
            )
        cache_position = torch.arange(L, device=self.device)
        out = self.model.trunk(
            inputs_embeds=embeds,
            attention_mask={"full_attention": mask},
            position_ids=position_ids,
            past_key_values=self.cache,
            cache_position=cache_position,
            use_cache=True,
        )
        self.pos.fill_(L)
        self.offsets[0, 0] = 0
        self.offsets[1, 0] = P
        self.pad_len.fill_(P)
        self.hidden.copy_(out.last_hidden_state[:, -1])
        return self.hidden.clone()

    # ------------------------------------------------------------------ 图体
    def _frame_body(self) -> None:
        B, d, Q, V = self.batch, self.d_model, self.Q, self.V
        self.prefix.zero_()
        for q in range(Q):
            a = self.k1(self.hidden, self.prefix, self.depth.norm_in.weight)
            act = self.k2(
                a,
                self.depth.mlp.gate_proj.weight,
                self.depth.mlp.up_proj.weight,
            )
            y = self.k3(act, self.depth.mlp.down_proj.weight, a)
            body = self.k4(y, self.depth.norm_out.weight)
            logits = self.k5(body, self.table[q], self.depth.bias[q])
            merged = (
                logits[0] if B == 1 else torch.lerp(logits[1], logits[0], self.cfg[0])
            )
            temp = self.temp_q0 if q == 0 else self.temp
            mask_eos = self.mask_eos_q0 if q == 0 else self.mask_eos_const
            filtered = self.k8(merged, temp, mask_eos)
            # 采样段与原生 _sample_logits 同序：top-k → top-p → softmax →
            # argmax(greedy) / multinomial(sampled，图内 RNG 逐 replay 推进已实测)
            f = filtered
            if self.top_k is not None and 0 < self.top_k < V:
                cutoff = torch.topk(f, self.top_k).values[-1]
                f = f.masked_fill(f < cutoff, float("-inf"))
            if self.top_p is not None and 0 < self.top_p < 1:
                ordered, order = torch.sort(f, descending=True)
                cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
                # 与原生 remove[1:]=remove[:-1]; remove[0]=False 等价，但全程设备侧
                # （remove[0]=False 的 host→device 赋值在图捕获时不合法）
                remove = torch.cat(
                    (
                        torch.zeros(1, dtype=torch.bool, device=f.device),
                        (cumulative > self.top_p)[:-1],
                    )
                )
                f = f.scatter(0, order, ordered.masked_fill(remove, float("-inf")))
            probs = torch.softmax(f, dim=-1)
            greedy_code = probs.argmax()
            drawn = torch.multinomial(probs, 1, generator=self.rng).squeeze(0)
            code = torch.where(self.sample_flag[0] > 0, drawn, greedy_code)
            self.codes[q] = code
            self.prefix = self.k7(self.prefix, self.table[q], code.view(1))
        embed = self.k6(self.codes.view(1, Q), self.table)  # [1, d]
        self.embeds.copy_(embed.unsqueeze(0).repeat(B, 1, 1))

        # 图内由 pos 计算 mask / position_ids / cache_position
        causal_ok = self.arange <= self.pos[0]  # [max_len] bool
        mask = torch.where(
            causal_ok,
            torch.zeros((), device=self.device, dtype=self.dtype),
            torch.full((), float("-inf"), device=self.device, dtype=self.dtype),
        ).view(1, 1, 1, self.max_len)
        mask = mask.repeat(B, 1, 1, 1)
        pad_ok = self.arange < self.pad_len[0]  # pad_len=0 时全 False，等价无操作
        row1 = torch.where(
            pad_ok,
            torch.full((), float("-inf"), device=self.device, dtype=self.dtype),
            mask[1, 0, 0],
        )
        mask[1, 0, 0] = row1
        position_ids = (self.pos[0] - self.offsets).view(B, 1)
        out = self.model.trunk(
            inputs_embeds=self.embeds,
            attention_mask={"full_attention": mask},
            position_ids=position_ids,
            past_key_values=self.cache,
            cache_position=self.pos,
            use_cache=True,
        )
        self.hidden.copy_(out.last_hidden_state[:, -1])
        self.pos.add_(1)

    @torch.inference_mode()
    def capture(self, warmup: int = 3) -> None:
        if self.graph is not None:
            return
        self._build_kernels()
        self._captured_sampling = (self.top_k, self.top_p)
        saved_hidden = self.hidden.clone()
        saved_pos = self.pos.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                self._frame_body()
        torch.cuda.current_stream().wait_stream(stream)
        # warmup 与捕获会推进 pos 并改写 hidden/cache，必须还原到 prefill 后状态，
        # 否则首批 replay 会读到污染槽位。
        self.hidden.copy_(saved_hidden)
        self.pos.copy_(saved_pos)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._frame_body()
        self.hidden.copy_(saved_hidden)
        self.pos.copy_(saved_pos)

    @torch.inference_mode()
    def step_frame(self, *, mask_eos_q0: bool) -> torch.Tensor:
        """一帧 replay；返回本帧 codes（[Q] 的 CPU 副本）。EOS 判断在图外。"""
        if self.graph is None:
            raise RuntimeError("先调用 capture()")
        self.mask_eos_q0.fill_(1 if mask_eos_q0 else 0)
        self.graph.replay()
        return self.codes.detach().cpu().clone()

    def set_temperature(self, value: float, *, q0: float | None = None) -> None:
        self.temp.fill_(value)
        self.temp_q0.fill_(value if q0 is None else q0)

    def set_sampling(
        self,
        *,
        sampled: bool,
        top_k: int | None,
        top_p: float | None,
    ) -> None:
        """设置采样模式与 top-k/top-p；top_k/top_p 烘焙进图，变化时触发重新捕获。"""
        self.sample_flag.fill_(1 if sampled else 0)
        params = (top_k, top_p)
        if self._captured_sampling is not None and params != self._captured_sampling:
            self.graph = None
        self.top_k = top_k
        self.top_p = top_p

    def set_seed(self, seed: int | None) -> None:
        if seed is not None:
            self.rng.manual_seed(int(seed))
