"""Soraku TTS v2: Qwen3 trunk with one step per acoustic frame and depth MTP.

The layout follows ``arch.txt``: text and audio share a single causal sequence,
the pretrained Qwen3 trunk advances exactly one position per 12 Hz frame, and a
depth head emits ``q0 -> q1 -> ... -> q15`` inside that frame.  The depth head
replaces the original sixteen independent heads because the residual codebooks
are not conditionally independent given ``h_t``.

Differences from v1 that matter:

* the trunk is pretrained and trainable, so text receives gradients;
* there is no diagonal delay, so each position sees one complete frame;
* frame embeddings are scaled by ``1/sqrt(Q)`` rather than ``1/Q``;
* position information comes from the trunk's RoPE, not a learned table;
* only ``q0`` may emit the audio stop token.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100

KIND_PAD = 0
KIND_TEXT = 1
KIND_AUDIO = 2


@dataclass(frozen=True)
class SorakuV2Config:
    text_model_path: str = "pre_train_models/Qwen3-0.6B-Base"
    num_codebooks: int = 16
    codebook_size: int = 2048
    depth_ffn_mult: int = 2
    tie_audio_head: bool = True
    freeze_text_embedding: bool = True
    initializer_range: float = 0.02
    speaker_conditioning: bool = False
    speaker_dim: int = 192
    # bos = 投影叠加到音频 BOS 帧（Arm B/C/D）；slot = 文本段与音频段之间的专用条件位（Arm E）
    speaker_condition_mode: str = "bos"
    language_token_ids: tuple[int, ...] = ()

    @property
    def audio_bos_id(self) -> int:
        return self.codebook_size

    @property
    def audio_eos_id(self) -> int:
        return self.codebook_size + 1

    @property
    def audio_pad_id(self) -> int:
        return self.codebook_size + 2

    @property
    def audio_vocab_size(self) -> int:
        return self.codebook_size + 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SorakuV2Output:
    loss: torch.Tensor | None = None
    code_loss: torch.Tensor | None = None
    stop_loss: torch.Tensor | None = None
    code_accuracy: torch.Tensor | None = None
    q0_accuracy: torch.Tensor | None = None
    stop_accuracy: torch.Tensor | None = None
    per_codebook_loss: torch.Tensor | None = None
    num_code_targets: torch.Tensor | None = None


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: int) -> None:
        super().__init__()
        inner = d_model * mult
        self.gate_proj = nn.Linear(d_model, inner, bias=False)
        self.up_proj = nn.Linear(d_model, inner, bias=False)
        self.down_proj = nn.Linear(inner, d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class AudioCodebooks(nn.Module):
    """One table per codebook, shared by the frame input and the depth head."""

    def __init__(self, config: SorakuV2Config, d_model: int) -> None:
        super().__init__()
        self.num_codebooks = config.num_codebooks
        self.scale = config.num_codebooks**-0.5
        self.table = nn.Parameter(
            torch.empty(config.num_codebooks, config.audio_vocab_size, d_model)
        )
        nn.init.normal_(self.table, mean=0.0, std=config.initializer_range)

    def frame_embedding(self, codes: torch.LongTensor) -> torch.Tensor:
        """``[..., Q] -> [..., d]``, the sum of the frame's codebooks."""

        if codes.shape[-1] != self.num_codebooks:
            raise ValueError("codes must end with the codebook dimension")
        total = None
        for q in range(self.num_codebooks):
            piece = F.embedding(codes[..., q], self.table[q])
            total = piece if total is None else total + piece
        return total * self.scale

    def exclusive_prefix(self, codes: torch.LongTensor) -> torch.Tensor:
        """``[M, Q] -> [M, Q, d]``, the sum of every codebook below ``q``."""

        stacked = torch.stack(
            [
                F.embedding(codes[..., q], self.table[q])
                for q in range(self.num_codebooks)
            ],
            dim=-2,
        )
        return (stacked.cumsum(dim=-2) - stacked) * self.scale


class DepthHead(nn.Module):
    """Shared body predicting codebook ``q`` from ``h_t`` and the lower codes."""

    def __init__(self, config: SorakuV2Config, d_model: int) -> None:
        super().__init__()
        self.tie_audio_head = config.tie_audio_head
        self.norm_in = nn.RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, config.depth_ffn_mult)
        self.norm_out = nn.RMSNorm(d_model)
        self.bias = nn.Parameter(
            torch.zeros(config.num_codebooks, config.audio_vocab_size)
        )
        if not config.tie_audio_head:
            self.out_proj = nn.Parameter(
                torch.empty(config.num_codebooks, config.audio_vocab_size, d_model)
            )
            nn.init.normal_(self.out_proj, mean=0.0, std=config.initializer_range)

    def body(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.norm_in(hidden)
        return self.norm_out(hidden + self.mlp(hidden))

    def forward(
        self,
        hidden: torch.Tensor,
        prefix: torch.Tensor,
        table: torch.Tensor,
    ) -> torch.Tensor:
        """``hidden`` is ``[M, d]`` and ``prefix`` is ``[M, Q, d]``."""

        body = self.body(hidden.unsqueeze(-2) + prefix)
        weight = table if self.tie_audio_head else self.out_proj
        return torch.einsum("mqd,qvd->mqv", body, weight) + self.bias

    def step(
        self,
        hidden: torch.Tensor,
        prefix: torch.Tensor,
        q: int,
        table: torch.Tensor,
    ) -> torch.Tensor:
        """Single codebook during generation.  ``hidden``/``prefix`` are ``[B, d]``."""

        body = self.body(hidden + prefix)
        weight = table[q] if self.tie_audio_head else self.out_proj[q]
        return body @ weight.transpose(0, 1) + self.bias[q]


class SorakuTTSv2(nn.Module):
    def __init__(self, config: SorakuV2Config, *, trunk: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.trunk = trunk
        d_model = int(trunk.config.hidden_size)
        self.d_model = d_model
        self.audio_codebooks = AudioCodebooks(config, d_model)
        self.depth_head = DepthHead(config, d_model)
        if config.language_token_ids:
            if (
                len(config.language_token_ids) != 4
                or len(set(config.language_token_ids)) != 4
            ):
                raise ValueError("exactly four unique language token IDs are required")
            # Only new rows train; the pretrained text table stays frozen.
            # Construct before optional speaker modules so A/D initialise identically.
            self.language_embeddings = nn.Embedding(4, d_model)
            nn.init.normal_(
                self.language_embeddings.weight, std=config.initializer_range
            )
        if config.speaker_conditioning:
            if config.speaker_condition_mode not in ("bos", "slot"):
                raise ValueError(
                    f"unknown speaker_condition_mode: {config.speaker_condition_mode}"
                )
            self.speaker_projection = nn.Linear(config.speaker_dim, d_model, bias=False)
            # Start from exactly the baseline function, with a live gradient.
            nn.init.zeros_(self.speaker_projection.weight)
            if config.speaker_condition_mode == "slot":
                # 专用条件位的可学习位置标记（零初始化；即使向量尚未学到，
                # slot 位置本身的存在也是可学习的信号）。
                self.speaker_slot_embedding = nn.Parameter(torch.zeros(d_model))
        if config.freeze_text_embedding:
            self.trunk.get_input_embeddings().requires_grad_(False)

    @classmethod
    def from_pretrained_trunk(
        cls,
        config: SorakuV2Config,
        *,
        dtype: torch.dtype = torch.float32,
        attn_implementation: str = "sdpa",
    ) -> SorakuTTSv2:
        from transformers import AutoModel

        trunk = AutoModel.from_pretrained(
            config.text_model_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
        return cls(config, trunk=trunk)

    def parameter_report(self) -> dict[str, int]:
        groups = {
            "trunk": sum(p.numel() for p in self.trunk.parameters()),
            "audio_codebooks": sum(
                p.numel() for p in self.audio_codebooks.parameters()
            ),
            "depth_head": sum(p.numel() for p in self.depth_head.parameters()),
        }
        groups["total"] = sum(groups.values())
        if self.config.language_token_ids:
            groups["language_embeddings"] = self.language_embeddings.weight.numel()
            groups["total"] += groups["language_embeddings"]
        if self.config.speaker_conditioning:
            groups["speaker_projection"] = sum(
                p.numel() for p in self.speaker_projection.parameters()
            )
            groups["total"] += groups["speaker_projection"]
        groups["trainable"] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return groups

    def optimizer_parameter_groups(
        self, *, audio_lr_multiplier: float = 5.0, weight_decay: float = 0.01
    ) -> list[dict[str, Any]]:
        """Split by branch and by whether weight decay applies.

        Randomly initialised audio parameters need a larger learning rate than
        the pretrained trunk.  Norms and biases are one-dimensional and are
        excluded from decay, which is standard and matters here because the
        trunk carries large RMSNorm gains (``max|w|`` of about 15 in the base
        checkpoint) that decay would erode.
        """

        audio_modules = (self.audio_codebooks, self.depth_head)
        if self.config.language_token_ids:
            audio_modules += (self.language_embeddings,)
        if self.config.speaker_conditioning:
            audio_modules += (self.speaker_projection,)
        audio_ids = {
            id(parameter)
            for module in audio_modules
            for parameter in module.parameters()
        }
        buckets: dict[tuple[str, bool], list[nn.Parameter]] = {
            ("trunk", True): [],
            ("trunk", False): [],
            ("audio", True): [],
            ("audio", False): [],
        }
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            branch = "audio" if id(parameter) in audio_ids else "trunk"
            buckets[(branch, parameter.ndim >= 2)].append(parameter)

        groups: list[dict[str, Any]] = []
        for (branch, decays), parameters in buckets.items():
            if not parameters:
                continue
            groups.append(
                {
                    "params": parameters,
                    "lr_scale": (
                        float(audio_lr_multiplier) if branch == "audio" else 1.0
                    ),
                    "weight_decay": float(weight_decay) if decays else 0.0,
                    "name": f"{branch}_{'decay' if decays else 'nodecay'}",
                }
            )
        return groups

    def _build_inputs_embeds(
        self,
        *,
        input_ids: torch.LongTensor,
        input_codes: torch.LongTensor,
        kind: torch.Tensor,
        speaker_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_embedding = self.trunk.get_input_embeddings()
        dtype = text_embedding.weight.dtype
        is_text = (kind == KIND_TEXT).unsqueeze(-1)
        is_audio = (kind == KIND_AUDIO).unsqueeze(-1)
        text_part = self.embed_text(input_ids) * is_text
        audio_part = self.audio_codebooks.frame_embedding(input_codes) * is_audio
        embeds = (text_part + audio_part).to(dtype)
        if self.config.speaker_condition_mode == "bos":
            speaker = self.project_speaker(speaker_embedding, input_ids.shape[0])
            if speaker is not None:
                bos = (kind == KIND_AUDIO) & (
                    input_codes == self.config.audio_bos_id
                ).all(-1)
                embeds = embeds + bos.unsqueeze(-1) * speaker.unsqueeze(1).to(dtype)
        return embeds

    def embed_text(self, input_ids: torch.LongTensor) -> torch.Tensor:
        table = self.trunk.get_input_embeddings()
        if not self.config.language_token_ids:
            return table(input_ids)
        special = torch.zeros_like(input_ids, dtype=torch.bool)
        indices = torch.zeros_like(input_ids)
        for index, identifier in enumerate(self.config.language_token_ids):
            match = input_ids == identifier
            special |= match
            indices = torch.where(match, index, indices)
        ordinary = table(input_ids.masked_fill(special, 0))
        language = self.language_embeddings(indices).to(ordinary.dtype)
        return torch.where(special.unsqueeze(-1), language, ordinary)

    def _insert_condition_slot(
        self,
        embeds: torch.Tensor,
        kind: torch.Tensor,
        attention_mask: torch.Tensor,
        target_codes: torch.LongTensor | None,
        speaker_embedding: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor | None]:
        """slot 模式：在每行首个音频位置之前插入专用条件位（Arm E）。

        因果注意力下，该位对所有音频位置可见；无 prompt 时 data.py 提供零向量，
        该位仅剩可学习位置标记。条件位的插入使音频段整体后移一位，
        target_codes 同步插入 IGNORE 行，loss 位置因此保持不变义。
        """
        batch, length, _ = embeds.shape
        speaker = self.project_speaker(speaker_embedding, batch)
        assert speaker is not None  # 条件模型在 slot 模式下必须提供向量
        slot = (speaker + self.speaker_slot_embedding).to(embeds.dtype)
        has_audio = (kind == KIND_AUDIO).any(dim=1)
        if not bool(has_audio.all()):
            raise ValueError(
                "slot conditioning requires at least one audio position per row"
            )
        audio_start = (kind == KIND_AUDIO).int().argmax(dim=1)

        out_embeds = embeds.new_zeros(batch, length + 1, embeds.shape[-1])
        out_mask = attention_mask.new_zeros(batch, length + 1)
        out_targets = None
        if target_codes is not None:
            out_targets = torch.full(
                (batch, length + 1, target_codes.shape[-1]),
                IGNORE_INDEX,
                dtype=target_codes.dtype,
                device=target_codes.device,
            )
        for row in range(batch):
            boundary = int(audio_start[row])
            out_embeds[row] = torch.cat(
                (
                    embeds[row, :boundary],
                    slot[row].unsqueeze(0),
                    embeds[row, boundary:],
                ),
                dim=0,
            )
            out_mask[row] = torch.cat(
                (
                    attention_mask[row, :boundary],
                    attention_mask.new_ones(1),
                    attention_mask[row, boundary:],
                ),
                dim=0,
            )
            if out_targets is not None:
                out_targets[row] = torch.cat(
                    (
                        target_codes[row, :boundary],
                        out_targets[row, boundary : boundary + 1].clone(),
                        target_codes[row, boundary:],
                    ),
                    dim=0,
                )
        return out_embeds, out_mask, out_targets

    def project_speaker(
        self, embedding: torch.Tensor | None, batch_size: int
    ) -> torch.Tensor | None:
        if not self.config.speaker_conditioning:
            if embedding is not None:
                raise ValueError("speaker embedding supplied to an unconditioned model")
            return None
        if embedding is None or embedding.shape != (
            batch_size,
            self.config.speaker_dim,
        ):
            raise ValueError(
                f"speaker_embedding must be [{batch_size},{self.config.speaker_dim}]"
            )
        embedding = embedding.to(self.speaker_projection.weight)
        if not torch.isfinite(embedding).all():
            raise ValueError("non-finite speaker embedding")
        return self.speaker_projection(F.normalize(embedding, dim=-1))

    def forward(
        self,
        *,
        input_ids: torch.LongTensor,
        input_codes: torch.LongTensor,
        kind: torch.Tensor,
        attention_mask: torch.Tensor,
        target_codes: torch.LongTensor | None = None,
        speaker_embedding: torch.Tensor | None = None,
    ) -> SorakuV2Output:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,L]")
        if input_codes.shape[:2] != input_ids.shape:
            raise ValueError("input_codes must be [B,L,Q] matching input_ids")
        if input_codes.shape[-1] != self.config.num_codebooks:
            raise ValueError("input_codes must end with the codebook dimension")

        inputs_embeds = self._build_inputs_embeds(
            input_ids=input_ids,
            input_codes=input_codes,
            kind=kind,
            speaker_embedding=speaker_embedding,
        )
        if (
            self.config.speaker_conditioning
            and self.config.speaker_condition_mode == "slot"
        ):
            inputs_embeds, attention_mask, target_codes = self._insert_condition_slot(
                inputs_embeds, kind, attention_mask, target_codes, speaker_embedding
            )
        hidden = self.trunk(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state

        if target_codes is None:
            return SorakuV2Output()
        if target_codes.shape[-1] != input_codes.shape[-1]:
            raise ValueError(
                "target_codes must match input_codes in the codebook dimension"
            )

        loss_mask = (target_codes != IGNORE_INDEX).any(dim=-1)
        selected_hidden = hidden[loss_mask]
        selected_targets = target_codes[loss_mask]
        return self._depth_loss(selected_hidden, selected_targets)

    def _depth_loss(
        self, hidden: torch.Tensor, targets: torch.LongTensor
    ) -> SorakuV2Output:
        config = self.config
        prefix_codes = torch.where(
            targets == IGNORE_INDEX,
            torch.full_like(targets, config.audio_pad_id),
            targets,
        )
        prefix = self.audio_codebooks.exclusive_prefix(prefix_codes)
        logits = self.depth_head(hidden, prefix, self.audio_codebooks.table)

        flat_logits = logits.reshape(-1, config.audio_vocab_size).float()
        flat_targets = targets.reshape(-1)
        per_token = F.cross_entropy(
            flat_logits, flat_targets, ignore_index=IGNORE_INDEX, reduction="none"
        )
        valid = flat_targets != IGNORE_INDEX
        is_stop = valid & (flat_targets == config.audio_eos_id)
        is_code = valid & ~is_stop

        loss = (per_token * valid).sum() / valid.sum().clamp_min(1)
        predictions = flat_logits.argmax(dim=-1)
        matches = predictions == flat_targets

        per_codebook = per_token.view(-1, config.num_codebooks)
        per_codebook_valid = valid.view(-1, config.num_codebooks)
        q0_valid = per_codebook_valid[:, 0]
        q0_matches = matches.view(-1, config.num_codebooks)[:, 0]

        return SorakuV2Output(
            loss=loss,
            code_loss=(per_token * is_code).sum() / is_code.sum().clamp_min(1),
            stop_loss=(per_token * is_stop).sum() / is_stop.sum().clamp_min(1),
            code_accuracy=(matches & is_code).sum() / is_code.sum().clamp_min(1),
            q0_accuracy=(q0_matches & q0_valid).sum() / q0_valid.sum().clamp_min(1),
            stop_accuracy=(matches & is_stop).sum() / is_stop.sum().clamp_min(1),
            per_codebook_loss=(
                (per_codebook * per_codebook_valid).sum(dim=0)
                / per_codebook_valid.sum(dim=0).clamp_min(1)
            ).detach(),
            num_code_targets=is_code.sum().detach(),
        )

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
        speaker_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        """Greedy/sampled rollout for a single utterance.

        ``prompt_codes`` are reference frames that seed the voice; they are fed
        with teacher forcing and excluded from the returned codes.  ``cfg_scale``
        above one applies classifier-free guidance against ``uncond_input_ids``.
        """

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("generate expects input_ids of shape [1,L]")
        if cfg_scale != 1.0 and uncond_input_ids is None:
            raise ValueError("classifier-free guidance requires uncond_input_ids")
        config = self.config
        device = next(self.parameters()).device
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(seed))

        branches = [input_ids.to(device)]
        if cfg_scale != 1.0:
            branches.append(uncond_input_ids.to(device))

        table = self.audio_codebooks.table
        # Structural tokens are never sampled; only q0 may end the utterance.
        blocked = torch.zeros(config.audio_vocab_size, dtype=torch.bool, device=device)
        blocked[config.audio_bos_id] = True
        blocked[config.audio_pad_id] = True

        states = [
            self._start_state(ids, prompt_codes, speaker_embedding) for ids in branches
        ]
        frames: list[list[int]] = []
        stop_reason = "max_frames"

        for index in range(max_new_frames):
            hiddens = [state["hidden"] for state in states]
            prefix = [torch.zeros_like(hiddens[0]) for _ in states]
            codes: list[int] = []
            reached_eos = False
            for q in range(config.num_codebooks):
                logits = [
                    self.depth_head.step(hiddens[branch], prefix[branch], q, table)[
                        0
                    ].float()
                    for branch in range(len(states))
                ]
                merged = logits[0]
                if len(logits) > 1:
                    merged = logits[1] + cfg_scale * (logits[0] - logits[1])
                merged = merged.masked_fill(blocked, -torch.inf)
                if q != 0 or index < min_new_frames:
                    merged[config.audio_eos_id] = -torch.inf
                code = _sample_logits(
                    merged,
                    temperature=(
                        temperature_q0
                        if q == 0 and temperature_q0 is not None
                        else temperature
                    ),
                    top_k=top_k_q0 if q == 0 and top_k_q0 is not None else top_k,
                    top_p=top_p_q0 if q == 0 and top_p_q0 is not None else top_p,
                    generator=generator,
                )
                if q == 0 and code == config.audio_eos_id:
                    reached_eos = True
                    break
                codes.append(code)
                token = torch.tensor([code], device=device)
                step_embedding = (
                    F.embedding(token, table[q]) * self.audio_codebooks.scale
                )
                for branch in range(len(states)):
                    prefix[branch] = prefix[branch] + step_embedding
            if reached_eos:
                stop_reason = "eos"
                break
            frames.append(codes)
            frame = torch.tensor(codes, device=device).view(1, 1, -1)
            for state in states:
                state["hidden"] = self._advance(state, frame)

        codes_tensor = torch.tensor(frames, dtype=torch.long).view(
            -1, config.num_codebooks
        )
        return codes_tensor, {
            "stop_reason": stop_reason,
            "num_frames": codes_tensor.shape[0],
            "cfg_scale": cfg_scale,
        }

    def _start_state(
        self,
        input_ids: torch.LongTensor,
        prompt_codes: torch.LongTensor | None,
        speaker_embedding: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        from transformers import DynamicCache

        device = next(self.parameters()).device
        text_embeds = self.embed_text(input_ids)
        bos = torch.full(
            (1, 1, self.config.num_codebooks),
            self.config.audio_bos_id,
            dtype=torch.long,
            device=device,
        )
        audio_inputs = bos
        if prompt_codes is not None and prompt_codes.numel():
            prompt = prompt_codes.to(device).view(1, -1, self.config.num_codebooks)
            audio_inputs = torch.cat((bos, prompt), dim=1)
        audio_embeds = self.audio_codebooks.frame_embedding(audio_inputs)
        speaker = self.project_speaker(speaker_embedding, 1)
        parts = [text_embeds]
        if speaker is not None and self.config.speaker_condition_mode == "slot":
            slot = (speaker + self.speaker_slot_embedding).to(text_embeds.dtype)
            parts.append(slot.unsqueeze(1))
        elif speaker is not None:
            audio_embeds = audio_embeds.clone()
            audio_embeds[:, 0] += speaker.to(audio_embeds.dtype)
        parts.append(audio_embeds.to(text_embeds.dtype))
        embeds = torch.cat(parts, dim=1)
        state: dict[str, Any] = {"cache": DynamicCache(), "offset": 0}
        state["hidden"] = self._run(state, embeds)
        return state

    def _run(self, state: dict[str, Any], embeds: torch.Tensor) -> torch.Tensor:
        length = embeds.shape[1]
        position_ids = torch.arange(
            state["offset"], state["offset"] + length, device=embeds.device
        ).unsqueeze(0)
        output = self.trunk(
            inputs_embeds=embeds,
            position_ids=position_ids,
            past_key_values=state["cache"],
            use_cache=True,
        )
        state["cache"] = output.past_key_values
        state["offset"] += length
        return output.last_hidden_state[:, -1]

    def _advance(self, state: dict[str, Any], frame: torch.LongTensor) -> torch.Tensor:
        embeds = self.audio_codebooks.frame_embedding(frame)
        text_dtype = self.trunk.get_input_embeddings().weight.dtype
        return self._run(state, embeds.to(text_dtype))

    def audio_stop_frame(self) -> list[int]:
        """The frame layout the collator must use for the stop position."""

        return [self.config.audio_eos_id] + [IGNORE_INDEX] * (
            self.config.num_codebooks - 1
        )


def _sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    generator: torch.Generator | None,
) -> int:
    if temperature <= 0:
        return int(logits.argmax().item())
    filtered = logits / temperature
    if top_k is not None and 0 < top_k < filtered.numel():
        cutoff = torch.topk(filtered, top_k).values[-1]
        filtered = filtered.masked_fill(filtered < cutoff, -torch.inf)
    if top_p is not None and 0 < top_p < 1:
        ordered, order = torch.sort(filtered, descending=True)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered = filtered.scatter(0, order, ordered.masked_fill(remove, -torch.inf))
    probabilities = torch.softmax(filtered, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())
