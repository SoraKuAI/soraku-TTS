"""StaticCache + 手工 CUDA Graph 的 trunk 单 token 步进引擎（原型）。

动机：trunk 单 token 前向在 eager 下 ~20ms（launch-bound），把整步捕获进
CUDA Graph 后 replay 一次完成，目标 ~1-2ms/step，为通话级 RTF（≤0.1）铺路。

关键路径（依据 transformers 4.57 Qwen3 源码）：
- ``attention_mask`` 传 dict（{"full_attention": 静态加性 mask}）可跳过
  ``create_causal_mask`` 的动态构建；层内 ``mask[:, :, :, :key_len]`` 的切片
  在 StaticCache 下为常量（max_len）。
- StaticCache 的 ``index_copy_`` 是算子级写入，可被图捕获；
  position_ids / cache_position / embeds / mask 全部用静态 buffer，步间原位更新。

一致性目标：与 eager DynamicCache 的逐步 hidden state 在 fp32 下逐位一致
（差异只来自 mask 路径的 kernel 归约顺序，greedy 逐位校验为准）。
"""

from __future__ import annotations

import torch


class GraphTrunkStep:
    """把 Qwen3 trunk 的单 token decode 步捕获为 CUDA Graph。

    用法：
        engine = GraphTrunkStep(trunk, max_cache_len=512, device="cuda")
        hidden = engine.prefill(prefill_embeds)          # eager 预填
        engine.capture()                                  # warmup + 捕获
        for each frame:
            hidden = engine.step(frame_embed)             # graph replay
    """

    def __init__(self, trunk, *, max_cache_len: int, device: torch.device) -> None:
        from transformers import StaticCache

        self.trunk = trunk
        self.device = device
        self.max_len = int(max_cache_len)
        self.dtype = next(trunk.parameters()).dtype
        self.cache = StaticCache(
            config=trunk.config,
            max_batch_size=1,
            max_cache_len=self.max_len,
            device=device,
            dtype=self.dtype,
        )
        d = int(trunk.config.hidden_size)
        self.embeds = torch.zeros(1, 1, d, device=device, dtype=self.dtype)
        self.position_ids = torch.zeros(1, 1, device=device, dtype=torch.long)
        self.cache_position = torch.zeros(1, device=device, dtype=torch.long)
        # 加性 mask：可见位置为 0，未来位置为 -inf；步间原位更新
        self.mask = torch.full(
            (1, 1, 1, self.max_len), float("-inf"), device=device, dtype=self.dtype
        )
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_out: torch.Tensor | None = None
        self.t = 0  # cache 中已写入的 token 数

    def prefill(self, embeds: torch.Tensor) -> torch.Tensor:
        """eager 预填：[1, L, d] → 返回最后一个位置的 hidden [1, d]。"""
        if self.graph is not None:
            raise RuntimeError("capture 之后不能再 prefill")
        L = embeds.shape[1]
        if L > self.max_len:
            raise ValueError(f"prefill 长度 {L} 超过 max_cache_len={self.max_len}")
        cache_position = torch.arange(L, device=self.device)
        position_ids = cache_position.unsqueeze(0)
        out = self.trunk(
            inputs_embeds=embeds,
            position_ids=position_ids,
            past_key_values=self.cache,
            cache_position=cache_position,
            use_cache=True,
        )
        self.mask[..., :L] = 0.0
        self.t = L
        return out.last_hidden_state[:, -1]

    def _forward_once(self) -> torch.Tensor:
        out = self.trunk(
            inputs_embeds=self.embeds,
            attention_mask={"full_attention": self.mask},
            position_ids=self.position_ids,
            past_key_values=self.cache,
            cache_position=self.cache_position,
            use_cache=True,
        )
        return out.last_hidden_state[:, -1]

    def _prepare_step_buffers(self, embed: torch.Tensor) -> None:
        self.embeds.view(1, -1).copy_(embed.reshape(-1).to(self.dtype))
        self.cache_position.fill_(self.t)
        self.position_ids.fill_(self.t)
        # 当前位置对注意力可见（此前位置 prefill/历史步已置 0）
        self.mask[..., self.t] = 0.0

    def capture(self, warmup: int = 3) -> None:
        """在 side stream 上 warmup 后捕获整步为 CUDA Graph。"""
        if self.graph is not None:
            return
        dummy = torch.zeros_like(self.embeds)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                self._prepare_step_buffers(dummy)
                self._forward_once()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        self._prepare_step_buffers(dummy)
        with torch.cuda.graph(self.graph):
            self.static_out = self._forward_once()

    def step(self, embed: torch.Tensor) -> torch.Tensor:
        """一帧步进：[1, d] → [1, d]（graph replay）。"""
        if self.graph is None or self.static_out is None:
            raise RuntimeError("先调用 capture()")
        if self.t >= self.max_len:
            raise RuntimeError("KV cache 已满（max_cache_len）")
        self._prepare_step_buffers(embed)
        self.graph.replay()
        self.t += 1
        return self.static_out.clone()
