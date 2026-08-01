# burn-asr-spike

Qwen3-ASR-0.6B 移植到 **burn 0.22.0-pre.1**（本地 checkout，`/Volumes/Workspace/GitHub/burn`，
`metal` feature = burn-wgpu → cubecl wgpu runtime + MSL 编译器），用于评估 tiny-cpm
从 candle 迁移到 burn 的可行性。

## 用法

```bash
cargo run --release -- <model-dir> <audio-file> [max_tokens]
# 例：
./target/release/burn-asr-spike /Volumes/Workspace/GitHub/tiny-cpm/models/Qwen3-ASR-0.6B \
    /Volumes/Workspace/GitHub/tiny-cpm/models/Fun-ASR-Nano-2512/example/en.mp3
```

- 转录 → stdout；加载/解码耗时、tok/s、RTF → stderr。
- 先完整跑 1 轮 warmup（MSL 着色器编译 + matmul autotune），再计正式轮。
- `BURN_ASR_F32=1`：诊断开关，权重与计算全部 f32（对照 f16 数值问题用）。
- 权重 bf16 safetensors → 加载时统一转 **f16**（cubecl-wgpu Metal 无 BF16）。

## 基准结果（同机同负载，负载 ~12–14）

**en.mp3（7.17s 英语，48kHz mp3→16k 重采样，109 prompt tokens）**

| | candle 0.11 (bf16) | burn 0.22-pre (f16) |
|---|---|---|
| load | 0.26–1.04 s (mmap) | 0.93–1.02 s (读文件+转换+上传) |
| decode | 865–902 ms | 840–872 ms |
| tok/s | 23.3–24.3 | 25.2–26.2 |
| RTF | 0.121–0.126 | **0.117–0.122** |

转录逐字一致：
`The tribal chief then called for the boy, and presented him with fifty pieces of gold.`

**input_question.wav（4.4s 中文，2ch/48kHz/Int16→16k 重采样，72 prompt tokens）**

| | candle | burn |
|---|---|---|
| decode | 627 ms (22.3 tok/s) | 661 ms (21.2 tok/s) |
| RTF | 0.142 | 0.150 |

转录逐字一致：
`我能用一句话解释一下什么是边缘计算吗？`

**结论：burn 在 Metal 上与 candle 解码性能持平（±5% 内），正确性在 2 个音频（mp3 单声道 F32 / wav 立体声 Int16，英语 / 中文）上逐字一致。**

## 移植要点与踩坑记录（burn 0.22.0-pre.1）

这些坑会以相同形式出现在每个模型的移植里：

1. **Metal 无 BF16**：cubecl-wgpu 能力表没有 BF16，`supports_dtype` 返回 false。
   权重 bf16→f16 转换（`TensorData::convert_dtype`，字节数不变）。
2. **f16 指数位不够（本项目最重要的一条）**：Qwen3-ASR 的 bf16 激活值合法地达到 ~5800，
   rms_norm 的 x²≈3.4e7 **超过 f16 上限 65504 → inf → rsqrt(inf)=0 → 整个 norm 清零 →
   后续所有层退化为恒等直通**，模型输出全乱。修复：**rms_norm 的方差计算上提 f32**
   （输入在 f16 可表示，只有平方溢出）。`rms_norm`/`rms_norm4`/`layer_norm` 都改了。
   bf16 的 8 位指数天然免疫此问题（candle 无此坑）。
3. **`select_assign` / `scatter` 的 `IndexingUpdateOp::Assign` 未实现**（bridge/ops/float.rs
   只支持 Add）——masked_scatter 改回 candle 原版的连续区间 + `slice_assign`（已实现）。
4. **`triu_mask`/`tril_mask` 命名与 torch 相反**：`triu_mask(offset=0)` 标的是**下三角**
   为 true（doc 例子为证）。causal mask 要用 `tril_mask([s,s], 0)`（true 在 j>i）。
   用反了会导致第 0 行全 -inf → softmax 0/0 → NaN → logits 全 NaN → 输出恒为 token 0。
5. **`unsqueeze` 插在 rank 最前**（`unsqueeze_dim(dim)` 才指定位置）；**`squeeze` 删所有
   size-1 维**（用 `squeeze_dim`）；`Tensor::cat` 收 `Vec`（不是数组）；`matmul` 要求
   同 rank（权重转置要 `.unsqueeze::<3>()`）；`SliceArg` 缺失维自动补**全区间**（slice_assign
   必须写全所有维）；`from_floats`/`from_ints` 从 slice 推 rank 常需 `Tensor::<1>::` 注解；
   Int→Float 用 `.float()` 而非 `cast`；`cast` 三 impl 歧义需要显式 kind 注解。
6. **`gen` 是 Rust 2024 保留字**（同 AGENTS.md 提醒）。
7. **立体声解交织**：交错 buffer（LRLR...）按连续块切分会把 L/R 混在一起
   （首版 bug，wav 转录直接错乱；mp3 单声道无此问题所以首测没暴露）。
   必须像 candle 一样按 `chan(ch)` 语义逐声道 stride 采样。
   教训：**单声道测试通过 ≠ 立体声正确**。
8. 每步 decode 与 candle 相同的同步点（logits `to_data` 拉回 CPU argmax）直接可用；
   `Device::sync()` 用于计时收尾。

## 文件结构（镜像 candle 侧）

- `src/model.rs` — 音频塔（3×Conv2d k3s2p1 + 正弦 PE + 18 层）+ Qwen3 文本解码器
  （28 层 QK-norm attention + GatedMLP + mrope + KV cache + tied lm_head）+ masked_scatter。
- `src/audio.rs` — CPU 音频管线：symphonia 解码 → sinc 重采样 → realfft STFT → slaney mel。
- `src/config.rs` — serde config（只留用到的字段）。
- `src/main.rs` — 权重加载（bf16→f16）、greedy 解码、warmup + 计时。
