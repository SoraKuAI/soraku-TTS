# pyright: reportInvalidTypeForm=false
# TileLang evaluates T.Tensor(...) as DSL metadata; it is not a PEP 484 type form.
"""v2 推理快路径用的 TileLang 融合 kernel（fp32，与原生 generate 数值口径一致）。

注意：
- 本模块不能使用 ``from __future__ import annotations``——该模式把 kernel
  函数注解变成字符串，tilelang 的 eager builder 延迟求值时无法解析工厂
  闭包中的形状参数（NameError）。
- 不用 ``T.gemm``：fp32 输入会被降到 TF32 张量核，破坏与原生 fp32 的
  greedy 一致性。分类器/FFN 一律手写串行 K 循环的 GEMV（FFMA，fp32 累加）。

融合 kernel 清单：

- add_rmsnorm：out = rmsnorm(hidden + prefix) * w
- swiglu_gemv：act = silu(x @ Wg^T) * (x @ Wu^T)
- down_residual_gemv：y = act @ Wd^T + hidden + prefix
- rmsnorm：out = rmsnorm(y) * w
- classifier_gemv：logits = body @ Wq^T + bias
- frame_embed：out = scale * Σ_q table[q, codes[q], :]
- prefix_add：out = prefix + table_q[code] * scale
- sampling_prep：logits * inv_temp + 结构 token/EOS 屏蔽

所有 kernel 以工厂函数 + @tilelang.jit 提供，首次调用编译并落磁盘缓存。
"""

import torch

import tilelang
import tilelang.language as T

NEG_INF = -1e30


@tilelang.jit(out_idx=[-1])
def add_rmsnorm(B: int, D: int, eps: float):
    @T.prim_func
    def kernel(
        hidden: T.Tensor((B, D), "float32"),
        prefix: T.Tensor((B, D), "float32"),
        weight: T.Tensor((D,), "float32"),
        out: T.Tensor((B, D), "float32"),
    ):
        with T.Kernel(B, threads=256) as b:
            shared = T.alloc_shared((D,), "float32")
            for j in T.Parallel(D):
                shared[j] = hidden[b, j] + prefix[b, j]
            acc = T.alloc_fragment((1,), "float32")
            T.clear(acc)
            for j in T.serial(D):
                acc[0] += shared[j] * shared[j]
            inv = 1.0 / T.sqrt(acc[0] / D + eps)
            for j in T.Parallel(D):
                out[b, j] = shared[j] * inv * weight[j]

    return kernel


@tilelang.jit(out_idx=[-1])
def rmsnorm(B: int, D: int, eps: float):
    @T.prim_func
    def kernel(
        x: T.Tensor((B, D), "float32"),
        weight: T.Tensor((D,), "float32"),
        out: T.Tensor((B, D), "float32"),
    ):
        with T.Kernel(B, threads=256) as b:
            shared = T.alloc_shared((D,), "float32")
            for j in T.Parallel(D):
                shared[j] = x[b, j]
            acc = T.alloc_fragment((1,), "float32")
            T.clear(acc)
            for j in T.serial(D):
                acc[0] += shared[j] * shared[j]
            inv = 1.0 / T.sqrt(acc[0] / D + eps)
            for j in T.Parallel(D):
                out[b, j] = shared[j] * inv * weight[j]

    return kernel


@tilelang.jit(out_idx=[-1])
def swiglu_gemv(B: int, I: int, D: int, block_I: int = 128):
    """act[b] = silu(x[b] @ Wg^T) * (x[b] @ Wu^T)；fp32 串行 K 累加。"""

    @T.prim_func
    def kernel(
        x: T.Tensor((B, D), "float32"),
        wg: T.Tensor((I, D), "float32"),
        wu: T.Tensor((I, D), "float32"),
        out: T.Tensor((B, I), "float32"),
    ):
        with T.Kernel(T.ceildiv(I, block_I), B, threads=128) as (bi, b):
            x_shared = T.alloc_shared((D,), "float32")
            for j in T.Parallel(D):
                x_shared[j] = x[b, j]
            gacc = T.alloc_fragment((block_I,), "float32")
            uacc = T.alloc_fragment((block_I,), "float32")
            T.clear(gacc)
            T.clear(uacc)
            for k in T.serial(D):
                for j in T.Parallel(block_I):
                    gi = bi * block_I + j
                    if gi < I:
                        gacc[j] += wg[gi, k] * x_shared[k]
                        uacc[j] += wu[gi, k] * x_shared[k]
            for j in T.Parallel(block_I):
                gi = bi * block_I + j
                if gi < I:
                    g = gacc[j]
                    out[b, gi] = g / (1.0 + T.exp(-g)) * uacc[j]

    return kernel


@tilelang.jit(out_idx=[-1])
def down_residual_gemv(B: int, D: int, I: int, block_D: int = 128):
    """y[b] = act[b] @ Wd^T + residual[b]（residual 为 norm_in 输出，见
    DepthHead.body：残差接在归一化后的 hidden 上，不是原始输入）。"""

    @T.prim_func
    def kernel(
        act: T.Tensor((B, I), "float32"),
        wd: T.Tensor((D, I), "float32"),
        residual: T.Tensor((B, D), "float32"),
        out: T.Tensor((B, D), "float32"),
    ):
        with T.Kernel(T.ceildiv(D, block_D), B, threads=128) as (bd, b):
            act_shared = T.alloc_shared((I,), "float32")
            for j in T.Parallel(I):
                act_shared[j] = act[b, j]
            acc = T.alloc_fragment((block_D,), "float32")
            T.clear(acc)
            for k in T.serial(I):
                for j in T.Parallel(block_D):
                    col = bd * block_D + j
                    if col < D:
                        acc[j] += wd[col, k] * act_shared[k]
            for j in T.Parallel(block_D):
                col = bd * block_D + j
                if col < D:
                    out[b, col] = acc[j] + residual[b, col]

    return kernel


@tilelang.jit(out_idx=[-1])
def classifier_gemv(B: int, V: int, D: int, block_V: int = 128):
    """logits[b] = body[b] @ Wq^T + bias。"""

    @T.prim_func
    def kernel(
        body: T.Tensor((B, D), "float32"),
        wq: T.Tensor((V, D), "float32"),
        bias: T.Tensor((V,), "float32"),
        out: T.Tensor((B, V), "float32"),
    ):
        with T.Kernel(T.ceildiv(V, block_V), B, threads=128) as (bv, b):
            body_shared = T.alloc_shared((D,), "float32")
            for j in T.Parallel(D):
                body_shared[j] = body[b, j]
            acc = T.alloc_fragment((block_V,), "float32")
            T.clear(acc)
            for k in T.serial(D):
                for j in T.Parallel(block_V):
                    col = bv * block_V + j
                    if col < V:
                        acc[j] += wq[col, k] * body_shared[k]
            for j in T.Parallel(block_V):
                col = bv * block_V + j
                if col < V:
                    out[b, col] = acc[j] + bias[col]

    return kernel


@tilelang.jit(out_idx=[-1])
def frame_embed(R: int, Q: int, V: int, D: int, scale: float, block_D: int = 256):
    """out[r] = scale * Σ_q table[q, codes[r, q], :]；R 为帧/批行数。"""

    @T.prim_func
    def kernel(
        codes: T.Tensor((R, Q), "int64"),
        table: T.Tensor((Q, V, D), "float32"),
        out: T.Tensor((R, D), "float32"),
    ):
        with T.Kernel(T.ceildiv(D, block_D), R, threads=128) as (bd, r):
            acc = T.alloc_fragment((block_D,), "float32")
            T.clear(acc)
            for q in T.serial(Q):
                code_q = T.cast(codes[r, q], "int32")
                for j in T.Parallel(block_D):
                    acc[j] += table[q, code_q, bd * block_D + j]
            for j in T.Parallel(block_D):
                out[r, bd * block_D + j] = acc[j] * scale

    return kernel


@tilelang.jit(out_idx=[-1])
def prefix_add(B: int, V: int, D: int, scale: float, block_D: int = 256):
    """out = prefix + table_q[code] * scale；code 为分支间共享的标量。"""

    @T.prim_func
    def kernel(
        prefix: T.Tensor((B, D), "float32"),
        table_q: T.Tensor((V, D), "float32"),
        codes: T.Tensor((1,), "int64"),
        out: T.Tensor((B, D), "float32"),
    ):
        with T.Kernel(T.ceildiv(D, block_D), B, threads=128) as (bd, b):
            code_b = T.cast(codes[0], "int32")
            for j in T.Parallel(block_D):
                col = bd * block_D + j
                out[b, col] = prefix[b, col] + table_q[code_b, col] * scale

    return kernel


@tilelang.jit(out_idx=[-1])
def sampling_prep(
    V: int,
    bos_id: int,
    pad_id: int,
    eos_id: int,
    block_V: int = 256,
):
    """out = logits / temperature，屏蔽 bos/pad（以及可选 EOS）。

    用除法（而非乘倒数）与原生 _sample_logits 的 logits/temperature 逐位一致。
    temperature 与 mask_eos 以运行时张量传入，避免重复编译。
    """

    @T.prim_func
    def kernel(
        logits: T.Tensor((V,), "float32"),
        temperature: T.Tensor((1,), "float32"),
        mask_eos: T.Tensor((1,), "int32"),
        out: T.Tensor((V,), "float32"),
    ):
        with T.Kernel(T.ceildiv(V, block_V), threads=128) as bv:
            for j in T.Parallel(block_V):
                v = bv * block_V + j
                if v < V:
                    # 全程用 T.if_then_else 表达式，不用 Python if 改写变量
                    # （eager builder 对符号条件下的变量重赋值会错误地恒执行）。
                    value = logits[v] / temperature[0]
                    value = T.if_then_else(v == bos_id, NEG_INF, value)
                    value = T.if_then_else(v == pad_id, NEG_INF, value)
                    value = T.if_then_else(
                        (mask_eos[0] > 0) & (v == eos_id), NEG_INF, value
                    )
                    out[v] = value

    return kernel
