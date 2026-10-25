"""v2 推理快路径：结构精简 + TileLang 融合 kernel，语义对齐原生 generate()。

与原生实现的差异（均不改变计算语义）：
- CFG 双分支合批：uncond 左填充对齐 cond 长度（pad 恒 mask、真实 token 保持原
  position_ids），trunk 与 depth head 都按 B=2 一次调用；
- depth step 由 5 个融合 kernel 完成（add+rmsnorm / swiglu gemv / down+residual /
  rmsnorm / 分类器 gemv），prefix 更新与 frame embedding 各 1 个 kernel；
- 采样段：融合 kernel 做温度除法与屏蔽（与原生 logits/temperature 逐位一致）；
  默认 native_sampling=True：复刻原生 top-k/top-p/softmax/multinomial 全向量
  序列且共用同一 generator，同 seed 采样输出与原生逐位一致（2026-10-22 实测
  token/frame 一致率 100%）；native_sampling=False 为压缩候选快采（top-50 上
  multinomial），分布等价但 RNG 消耗不同、同 seed 输出不同；
- 每帧仅 q0 的 EOS 判断同步 1 次（原生每个 q 都同步）。

greedy 与 sampled（native_sampling=True）模式均与原生 generate 的输出 codes
完全一致（A/B 校验口径）。
"""

from __future__ import annotations

from typing import Any

import torch

from . import tilelang_kernels as TLK


class FastGenerator:
    def __init__(
        self,
        model,
        *,
        device: torch.device,
        use_tilelang: bool = True,
        compile_trunk: bool = False,
        batch_branches: bool = True,
        native_sampling: bool = True,
        use_graph: bool = False,
    ):
        self.model = model
        self.config = model.config
        if getattr(self.config, "speaker_conditioning", False):
            raise ValueError(
                "speaker-conditioned checkpoints currently require native generate; fast prefill is not validated"
            )
        self.device = device
        self.use_tilelang = use_tilelang
        self.batch_branches = batch_branches
        # True：采样走与原生 _sample_logits 完全相同的序列与 RNG 消耗，
        # 同 seed 下输出与原生一致（推荐，消除"神鬼二相性"）；
        # False：top-k 压缩候选采样（更快但同 seed 输出与原生不同）。
        self.native_sampling = native_sampling
        # True：greedy 走整帧单图引擎（GraphFrameEngine，每帧一次 CUDA Graph
        # replay，实测 ~13ms/帧且与 eager 逐位一致）；sampled 走 eager。
        self.use_graph = use_graph
        self._graph_engine = None
        self._trunk_fn = model.trunk
        if compile_trunk:
            # 编译不改动原模块；DynamicCache 增长导致 dynamic shape，
            # 用默认模式（reduce-overhead/cudagraphs 需要静态形状，不适用）。
            self._trunk_fn = torch.compile(model.trunk, dynamic=False)
        self.table = model.audio_codebooks.table  # [Q, V, d]
        self.scale = float(model.audio_codebooks.scale)
        self.depth = model.depth_head
        self.d_model = int(model.d_model)
        self.Q = int(self.config.num_codebooks)
        self.V = int(self.config.audio_vocab_size)
        self._arange_v = torch.arange(self.V, device=device)
        self._temperature = torch.zeros(1, dtype=torch.float32, device=device)
        self._mask_eos = torch.zeros(1, dtype=torch.int32, device=device)
        # nn.RMSNorm(eps=None) 的缺省是 torch.finfo(dtype).eps
        finfo_eps = torch.finfo(torch.float32).eps
        self._eps_in = self.depth.norm_in.eps
        self._eps_out = self.depth.norm_out.eps
        if self._eps_in is None:
            self._eps_in = finfo_eps
        if self._eps_out is None:
            self._eps_out = finfo_eps
        self._kernels: dict[str, Any] = {}

    # ---------------------------------------------------------- kernel 管理
    def _kernel(self, name: str, factory, *params):
        key = f"{name}:{'x'.join(map(str, params))}"
        kernel = self._kernels.get(key)
        if kernel is None:
            kernel = factory(*params)
            self._kernels[key] = kernel
        return kernel

    def _depth_logits(
        self, hidden: torch.Tensor, prefix: torch.Tensor, q: int
    ) -> torch.Tensor:
        """hidden/prefix [B, d] -> logits [B, V]。"""
        # tilelang kernel 的 packed ABI 校验 strides；HF 的 last_hidden_state
        # 切片是跨步视图，必须落连续内存。
        hidden = hidden.contiguous()
        prefix = prefix.contiguous()
        B = hidden.shape[0]
        if not self.use_tilelang:
            logits = [
                self.depth.step(hidden[b : b + 1], prefix[b : b + 1], q, self.table)[0]
                for b in range(B)
            ]
            return torch.stack(logits, dim=0)

        d = self.d_model
        inner = int(self.depth.mlp.gate_proj.out_features)
        k1 = self._kernel("add_rmsnorm", TLK.add_rmsnorm, B, d, float(self._eps_in))
        k2 = self._kernel("swiglu", TLK.swiglu_gemv, B, inner, d)
        k3 = self._kernel("down_residual", TLK.down_residual_gemv, B, d, inner)
        k4 = self._kernel("rmsnorm", TLK.rmsnorm, B, d, float(self._eps_out))
        k5 = self._kernel("classifier", TLK.classifier_gemv, B, self.V, d)

        a = k1(hidden, prefix, self.depth.norm_in.weight)  # [B, d]，norm_in(x)
        act = k2(a, self.depth.mlp.gate_proj.weight, self.depth.mlp.up_proj.weight)
        # DepthHead.body 的残差接在 norm_in 输出上，不是原始 (hidden+prefix)
        y = k3(act, self.depth.mlp.down_proj.weight, a)  # [B, d]
        body = k4(y, self.depth.norm_out.weight)  # [B, d]
        return k5(body, self.table[q], self.depth.bias[q])  # [B, V]

    # ---------------------------------------------------------- 采样
    def _prep(self, merged: torch.Tensor, *, mask_eos: bool, temperature: float):
        self._temperature.fill_(temperature)
        self._mask_eos.fill_(1 if mask_eos else 0)
        if self.use_tilelang:
            kernel = self._kernel(
                "sampling_prep",
                TLK.sampling_prep,
                self.V,
                self.config.audio_bos_id,
                self.config.audio_pad_id,
                self.config.audio_eos_id,
            )
            return kernel(merged, self._temperature, self._mask_eos)
        out = merged / temperature
        out = out.clone()
        out[self.config.audio_bos_id] = float("-inf")
        out[self.config.audio_pad_id] = float("-inf")
        if mask_eos:
            out[self.config.audio_eos_id] = float("-inf")
        return out

    def _sample_native(
        self,
        filtered: torch.Tensor,
        *,
        top_k: int | None,
        top_p: float | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """与原生 _sample_logits 完全同序、同 RNG 消耗；返回 GPU int64 标量。"""
        if top_k is not None and 0 < top_k < filtered.numel():
            cutoff = torch.topk(filtered, top_k).values[-1]
            filtered = filtered.masked_fill(filtered < cutoff, float("-inf"))
        if top_p is not None and 0 < top_p < 1:
            ordered, order = torch.sort(filtered, descending=True)
            cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
            remove = cumulative > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            filtered = filtered.scatter(
                0, order, ordered.masked_fill(remove, float("-inf"))
            )
        probabilities = torch.softmax(filtered, dim=-1)
        return torch.multinomial(probabilities, 1, generator=generator).squeeze(0)

    def _sample(
        self,
        filtered: torch.Tensor,
        *,
        top_k: int | None,
        top_p: float | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """返回 GPU 上的 int64 标量 code（不掉到 CPU）。"""
        V = filtered.numel()
        if top_k is not None and 0 < top_k < V:
            values, indices = torch.topk(filtered, top_k)
        else:
            values, indices = filtered, self._arange_v
        if top_p is not None and 0 < top_p < 1:
            ordered, order = values.sort(descending=True)
            cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
            remove = cumulative > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            values = ordered.masked_fill(remove, float("-inf"))
            indices = indices[order]
        probs = torch.softmax(values, dim=-1)
        pick = torch.multinomial(probs, 1, generator=generator)
        return indices[pick]

    # ---------------------------------------------------------- trunk
    def _trunk_step(self, state: dict[str, Any], embeds: torch.Tensor) -> torch.Tensor:
        length = embeds.shape[1]
        position_ids = torch.arange(
            state["offset"], state["offset"] + length, device=embeds.device
        ).unsqueeze(0)
        output = self._trunk_fn(
            inputs_embeds=embeds,
            position_ids=position_ids,
            past_key_values=state["cache"],
            use_cache=True,
        )
        state["cache"] = output.past_key_values
        state["offset"] += length
        return output.last_hidden_state[:, -1]

    def _start(
        self, input_ids: torch.LongTensor, prompt_codes: torch.LongTensor | None
    ):
        from transformers import DynamicCache

        text_embeds = self.model.embed_text(input_ids)
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
        audio_embeds = self.model.audio_codebooks.frame_embedding(audio_inputs)
        embeds = torch.cat((text_embeds, audio_embeds.to(text_embeds.dtype)), dim=1)
        state: dict[str, Any] = {"cache": DynamicCache(), "offset": 0}
        state["hidden"] = self._trunk_step(state, embeds)
        return state

    # ---------------------------------------------------------- 双分支合批
    def _start_batched(
        self,
        input_ids: torch.LongTensor,
        uncond_input_ids: torch.LongTensor,
        prompt_codes: torch.LongTensor | None,
        max_new_frames: int,
    ) -> dict[str, Any]:
        """cond/uncond 一次 prefill：uncond 左填充对齐 cond 文本长度。

        pad 位在 attention_mask 中恒为 0（softmax 后权重严格为 0），
        position_ids 对真实 token 保持原逻辑位置，因此与逐分支计算数值等价
        （唯一差异是批量 kernel 内 fp 归约顺序）。
        """
        from transformers import DynamicCache

        cond = input_ids.to(self.device)
        uncond = uncond_input_ids.to(self.device)
        Lc, Lu = cond.shape[1], uncond.shape[1]
        if Lu > Lc:
            raise ValueError("uncond 文本长于 cond，无法左填充对齐")
        pad = Lc - Lu

        ids = cond.new_zeros(2, Lc)
        ids[0] = cond[0]
        ids[1, pad:] = uncond[0]
        text_embeds = self.model.embed_text(ids)  # [2, Lc, d]

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
            audio_inputs.repeat(2, 1, 1)
        )

        embeds = torch.cat(
            (text_embeds, audio_embeds.to(text_embeds.dtype)), dim=1
        )  # [2, L, d]
        L = embeds.shape[1]

        mask = torch.ones(2, L + max_new_frames, dtype=torch.long, device=self.device)
        mask[1, :pad] = 0

        position_ids = torch.arange(L, device=self.device).unsqueeze(0).repeat(2, 1)
        uncond_pos = torch.cat(
            (
                torch.zeros(pad, dtype=torch.long, device=self.device),
                torch.arange(Lu, device=self.device),
                Lu + torch.arange(La, device=self.device),
            )
        )
        position_ids[1] = uncond_pos

        state: dict[str, Any] = {
            "cache": DynamicCache(),
            # 下一个解码位置的逻辑位置：文本长度 + 音频前缀帧数
            "pos": torch.tensor(
                [[Lc + La], [Lu + La]], dtype=torch.long, device=self.device
            ),
            "mask": mask,
            "filled": L,
        }
        output = self._trunk_fn(
            inputs_embeds=embeds,
            attention_mask=mask[:, :L],
            position_ids=position_ids,
            past_key_values=state["cache"],
            use_cache=True,
        )
        state["cache"] = output.past_key_values
        state["hidden"] = output.last_hidden_state[:, -1]  # [2, d]
        return state

    def _trunk_step_batched(
        self, state: dict[str, Any], frame_embeds: torch.Tensor
    ) -> torch.Tensor:
        """frame_embeds [1, 1, d]；两分支写同一帧。返回 [2, d]。"""
        cur = state["filled"]
        embeds = frame_embeds.repeat(2, 1, 1)
        output = self._trunk_fn(
            inputs_embeds=embeds,
            attention_mask=state["mask"][:, : cur + 1],
            position_ids=state["pos"],
            past_key_values=state["cache"],
            use_cache=True,
        )
        state["cache"] = output.past_key_values
        state["pos"] += 1
        state["filled"] = cur + 1
        return output.last_hidden_state[:, -1]

    # ---------------------------------------------------------- 主循环
    @torch.inference_mode()
    def generate(
        self,
        *,
        input_ids: torch.LongTensor,
        prompt_codes: torch.LongTensor | None = None,
        max_new_frames: int = 384,
        min_new_frames: int = 4,
        temperature: float = 0.9,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        temperature_q0: float | None = None,
        top_k_q0: int | None = None,
        top_p_q0: float | None = None,
        cfg_scale: float = 1.0,
        uncond_input_ids: torch.LongTensor | None = None,
        seed: int | None = None,
        on_frame: Any = None,
        use_graph: bool | None = None,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("generate expects input_ids of shape [1,L]")
        if cfg_scale != 1.0 and uncond_input_ids is None:
            raise ValueError("classifier-free guidance requires uncond_input_ids")
        want_graph = self.use_graph if use_graph is None else use_graph
        if want_graph:
            mixed = temperature_q0 is not None and (temperature <= 0) != (
                temperature_q0 <= 0
            )
            if mixed:
                # q0 与其余 q 的 greedy/sampled 混合模式暂走 eager
                pass
            else:
                return self._generate_graph(
                    input_ids=input_ids,
                    prompt_codes=prompt_codes,
                    max_new_frames=max_new_frames,
                    min_new_frames=min_new_frames,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    temperature_q0=temperature_q0,
                    cfg_scale=cfg_scale,
                    uncond_input_ids=uncond_input_ids,
                    seed=seed,
                    on_frame=on_frame,
                )
        config = self.config
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        branches = [input_ids.to(self.device)]
        if cfg_scale != 1.0:
            branches.append(uncond_input_ids.to(self.device))

        if cfg_scale != 1.0 and self.batch_branches:
            states = None
            state_b: dict[str, Any] | None = self._start_batched(
                input_ids, uncond_input_ids, prompt_codes, max_new_frames
            )
            B = 2
        else:
            state_b = None
            states = [self._start(ids, prompt_codes) for ids in branches]
            B = len(states)
        frames: list[torch.Tensor] = []
        stop_reason = "max_frames"

        for index in range(max_new_frames):
            hidden = (
                state_b["hidden"]
                if state_b is not None
                else torch.cat([state["hidden"] for state in states], dim=0)
            )
            prefix = torch.zeros(B, self.d_model, device=self.device)
            row: list[torch.Tensor] = []
            reached_eos = False
            for q in range(self.Q):
                logits = self._depth_logits(hidden, prefix, q)
                merged = (
                    logits[0] if B == 1 else torch.lerp(logits[1], logits[0], cfg_scale)
                )
                temp = (
                    temperature_q0
                    if q == 0 and temperature_q0 is not None
                    else temperature
                )
                mask_eos = q != 0 or index < min_new_frames
                if temp <= 0:
                    filtered = self._prep(merged, mask_eos=mask_eos, temperature=1.0)
                    code = filtered.argmax()
                else:
                    filtered = self._prep(merged, mask_eos=mask_eos, temperature=temp)
                    sampler = (
                        self._sample_native if self.native_sampling else self._sample
                    )
                    code = sampler(
                        filtered,
                        top_k=top_k_q0 if q == 0 and top_k_q0 is not None else top_k,
                        top_p=top_p_q0 if q == 0 and top_p_q0 is not None else top_p,
                        generator=generator,
                    )
                if q == 0 and bool(code.item() == config.audio_eos_id):
                    reached_eos = True
                    break
                row.append(code)
                if self.use_tilelang:
                    k7 = self._kernel(
                        "prefix_add",
                        TLK.prefix_add,
                        B,
                        self.V,
                        self.d_model,
                        self.scale,
                    )
                    prefix = k7(prefix, self.table[q], code.view(1))
                else:
                    token = code.view(1)
                    step_embedding = (
                        torch.nn.functional.embedding(token, self.table[q]) * self.scale
                    )
                    prefix = prefix + step_embedding
            if reached_eos:
                stop_reason = "eos"
                break
            frame = torch.stack(row).view(1, 1, self.Q)
            frames.append(frame)
            if on_frame is not None:
                on_frame(frame)
            if self.use_tilelang:
                k6 = self._kernel(
                    "frame_embed",
                    TLK.frame_embed,
                    1,
                    self.Q,
                    self.V,
                    self.d_model,
                    self.scale,
                )
                embeds = k6(frame[0], self.table).unsqueeze(0)
            else:
                embeds = self.model.audio_codebooks.frame_embedding(frame)
            if state_b is not None:
                state_b["hidden"] = self._trunk_step_batched(state_b, embeds)
            else:
                for state in states:
                    state["hidden"] = self._trunk_step(state, embeds)

        codes_tensor = (
            torch.cat(frames, dim=1).view(-1, self.Q).cpu()
            if frames
            else torch.empty(0, self.Q, dtype=torch.long)
        )
        return codes_tensor, {
            "stop_reason": stop_reason,
            "num_frames": codes_tensor.shape[0],
            "cfg_scale": cfg_scale,
        }

    # ---------------------------------------------------------- 整帧单图 backend
    @torch.inference_mode()
    def _generate_graph(
        self,
        *,
        input_ids: torch.LongTensor,
        prompt_codes: torch.LongTensor | None,
        max_new_frames: int,
        min_new_frames: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        temperature_q0: float | None,
        cfg_scale: float,
        uncond_input_ids: torch.LongTensor | None,
        seed: int | None,
        on_frame: Any = None,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        """整帧单图路径：prefill → capture（首次/参数变化）→ 每帧一次 replay。"""
        from .graph_frame import GraphFrameEngine

        cond = input_ids.to(self.device)
        uncond = (
            uncond_input_ids.to(self.device) if uncond_input_ids is not None else cond
        )
        prompt_len = 1 + (
            0
            if prompt_codes is None or not prompt_codes.numel()
            else prompt_codes.shape[0]
        )
        required = cond.shape[1] + prompt_len + max_new_frames + 8
        if self._graph_engine is None or self._graph_engine.max_len < required:
            self._graph_engine = GraphFrameEngine(
                self.model,
                device=self.device,
                max_cache_len=max(required, 512),
                cfg_scale=cfg_scale,
            )
        engine = self._graph_engine
        engine.cfg.fill_(cfg_scale)
        sampled = temperature > 0
        engine.set_temperature(
            1.0 if not sampled else temperature,
            q0=temperature_q0 if temperature_q0 is not None else None,
        )
        engine.set_sampling(
            sampled=sampled,
            top_k=top_k if sampled else None,
            top_p=top_p if sampled else None,
        )
        engine.set_seed(seed)
        engine.prefill(cond, uncond, prompt_codes)
        if engine.graph is None:
            engine.capture()

        config = self.config
        frames: list[torch.Tensor] = []
        stop_reason = "max_frames"
        for index in range(max_new_frames):
            codes = engine.step_frame(mask_eos_q0=index < min_new_frames)
            if index >= min_new_frames and int(codes[0]) == config.audio_eos_id:
                stop_reason = "eos"
                break
            frames.append(codes.view(1, 1, self.Q))
            if on_frame is not None:
                on_frame(frames[-1])
        codes_tensor = (
            torch.cat(frames, dim=1).view(-1, self.Q)
            if frames
            else torch.empty(0, self.Q, dtype=torch.long)
        )
        return codes_tensor, {
            "stop_reason": stop_reason,
            "num_frames": codes_tensor.shape[0],
            "cfg_scale": cfg_scale,
            "backend": "graph",
            "sampled": sampled,
            "temperature": temperature,
        }
