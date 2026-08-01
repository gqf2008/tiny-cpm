# burn-tts-spike

Qwen3-TTS-12Hz-1.7B-Base 核心路径移植到 **burn 0.22.0-pre.1**（本地 checkout，
`/Volumes/Workspace/GitHub/burn`，`metal` feature = burn-wgpu → cubecl wgpu runtime +
MSL 编译器），评估 tiny-cpm 从 candle 迁移到 burn 的可行性（第二阶段：TTS）。

范围（用户确认）：**talker（28 层 Qwen3）+ code predictor（5 层 × 15 AR 步）+ Mimi
codec decoder + GPU 采样**。`--ref` 声音克隆 / Q4_K 量化 / fusion 是 Phase 4-6 追加
（初始范围外）。

## 用法

```bash
cargo run --release -- <model-dir> "<text>" <out.wav> [--language zh|en] [--max-frames N] [--codes-file <file>] [--encode-wav <file>] [--spk-embed <file>] [--ref <wav> --ref-text "<text>"]
# 例：
../target-shared/release/burn-tts-spike /Volumes/Workspace/GitHub/tiny-cpm/models/Qwen3-TTS-12Hz-1.7B-Base \
    "The quick brown fox jumps over the lazy dog." /tmp/out.wav \
    --ref ref.wav --ref-text "full ref transcript"
```

`--talker-quant`：Q4_K 量化 talker + predictor backbone（自定义 cubecl GEMV kernel，
省内存、速度无收益 —— 见 Phase 5）。`--qmat-test`：Q4K GEMV vs f32 linear 的
数值 + 延迟对照（验证工具）。`--bench-gen <N>`：无 readback 生成基准（融合加速的
可复现 harness —— 见 Phase 6）。

- 音频时长 / gen、codec 分段耗时、RTF → stderr。
- 先完整跑 1 轮 warmup（MSL 着色器编译 + matmul autotune），再计正式轮。
- 加载/构建/释放各阶段的进程 RSS（MiB）随模型加载打印到 stderr（burn 0.22-pre 无
  `Backend::memory()` API，用 `ps` 取 RSS）。
- `--codes-file <file>`：跳过 talker，把外部 (n,16) codes 直接喂给 codec
  （codec 单独验证用）。**注意**：文件是 frame-major 的（每帧 16 个 code 连续），
  见踩坑 #2。
- 权重 bf16 safetensors → 加载时统一转 **f16**（cubecl-wgpu Metal 无 BF16）。
  talker 用 f16，codec 保持 **f32**（与 candle 一致，也便于逐位对照）。

### 诊断环境变量（全部可选）

| env | 作用 |
|---|---|
| `BURN_TTS_GREEDY=1` | 强制 argmax（对照 candle 的 `QWEN3_TTS_GREEDY=1` 验证用） |
| `BURN_TTS_DUMP_CODES=1` | stderr 打印 (n,16) codes（对照 candle 的 `CANDLE_CODES` dump） |
| `BURN_TTS_DUMP_PCM=1` | stderr 打印 F32 PCM（codec 数值对照用） |
| `BURN_TTS_DEBUG=1` | predictor 每步中间值 dump（g=2 hidden / g=3 logits top5，对照 `CANDLE_DUMP`） |
| `BURN_TTS_F32=1` | talker 改用 f32（诊断用，**慢 8 倍**，见基准表） |

## 验证结果

### Phase 1 — talker + predictor codes 逐位对比（candle `QWEN3_TTS_GREEDY=1` + codes dump）

同文本 greedy 对比：**帧 0–1 的 32 个 codes 逐位一致**；帧 2 起分歧，归因为
**并行 logits 平票的精度翻转**而非移植 bug：

- candle（bf16）：`logits[1174] == logits[1531] == 32.75`（完全平票，argmax 取先者）
- burn（f32 对照）：`logits[1531]` 高 0.04 → 选 1531
- f16 与 f32 两个 burn 版本在同一位置分歧方向一致，排除 dtype 转换 bug；
  gumbel 采样模式下平票被噪声打破，实际输出不受影响。

（candle 侧靠临时 env-gated 补丁 dump codes，验证完已还原。）

### Phase 2 — 同 codes → PCM 数值对比（candle 41 帧 greedy codes 喂两边 codec）

| 指标 | 值 |
|---|---|
| max\|Δ\| | **10 LSB**（满量程 32767） |
| 逐位一致 | 71.7%（56434/78720 samples） |
| mean\|Δ\| | 0.37 LSB |
| RMS | 1438.2 vs 1438.0（candle / burn） |

结论：**codec 移植数值等价**（余量是 fp32 算子级舍入，-70 dB 以下听不到）。
顺带验证了 `chunked_decode(300,25)` 语义：41 帧 < 300 → 单 chunk、context 0 → 等价于
全量 decode。

### Phase 4 — --ref 声音克隆（ICL）验证

参考音频 `Fun-ASR-Nano-2512/example/en.mp3`（7.17s，172224 samples @ 24k → 90 codec 帧）：

| 验证项 | 结果 |
|---|---|
| V-1 codec encoder codes | **99.44% 逐位一致**（1432/1440），8 帧各 1 个 code 分歧，归因为 RVQ 最近邻**近平票**（top-2 距离差 1.6e-3 ~ 1e-1，F32 舍入翻转 argmin） |
| V-2 speaker embedding | **max\|Δ\|=0.008**（信号 RMS 0.42，rel 1.9%，F32 舍入水平）。期间抓到并修复一个真 bug：ASP `weighted_stats` 把 softmax 权重先乘到残差上再平方（`((x-mean)·m)²`），应为 `(x-mean)²·m` —— 症状是 embedding 整体 ~0.65× 缩放（BLK0-4 统计量一致、BLK5 分歧，二分定位） |
| V-3 端到端 ICL greedy codes | **前 6 帧（96 codes）逐位一致**；帧 0 cb7 分歧为**精确平票**：candle `logits[1202]==logits[1434]==26.625` 取先者，burn 差 0.06 取 1434 —— 近平票 tie-break，非移植 bug |
| 端到端 wav | 能量正常（RMS ~5000，与 candle 同级），无 babble |

ICL 模式基准（采样）：candle RTF ~1.9 vs burn ~6.3（gen ~3×，codec ~3× 已知）。gen 的额外惩罚来自**长 prompt decode**：ref 90 帧 → prompt ~100 token，burn 的 eager attention 每帧重建 O(seq²) mask 且无因果优化，长序列下每帧成本放大（无 ref 时 113ms/帧 vs 有 ref 350ms/帧；candle 同条件只慢 1.8×）。记录为后续优化点。

### Phase 3 — 端到端基准（同机同负载，M4，`vm.loadavg` 1min ~2.4，采样模式默认配置）

文本：`The quick brown fox jumps over the lazy dog.`

| | candle 0.11 (bf16) | burn 0.22-pre (f16 talker + f32 codec) |
|---|---|---|
| load | 9.0 s（冷，首次 mmap）→ 0.6 s（暖） | 3.7–6.9 s |
| RTF | 1.03 / 1.01 / 0.99 | **2.49 / 2.38 / 2.51** |
| 分段 | — | gen ~4.5–5.0 s（40–46 帧），codec ~3.4–3.8 s |

结论：**总 RTF ~2.5，比 candle BF16（~1.0）慢 ~2.4 倍**。分段看：talker 生成
~1.8×，codec ~3×。对照 burn-asr-spike 的持平结论（ASR 是 prefill 主导的大 matmul、
占用率高；TTS 是每帧 16 步串行 m=1 decode + 大量小 conv，**launch-bound**）——
burn 的算子融合没有自动补偿这部分开销，且 `BURN_TTS_F32=1` 时 gen 慢 8 倍
（37.5 s vs 4.5 s），说明 f16 是硬需求，剩余差距在 kernel 调度而非 dtype。

后续优化点（记录，不在本次范围）：causal conv 的左 pad+cat 合并进单个 kernel；
codec 小 conv 的 fused kernel；预测器 15 步的并行/投机解码提高 m=1 占用率。

## 移植要点与踩坑记录（burn 0.22.0-pre.1，TTS 特有）

1. **f16 方差溢出**（与 ASR spike 相同）：所有 rms_norm/layer_norm 的方差计算上提
   f32，否则 talker 激活 ~5800 的平方溢出 65504 → norm 退化为恒等 → 整网乱码。
2. **codes 的 frame-major → (1,16,n) 布局必须转置，不能 reshape**（本 spike 最大的坑）：
   dump/生成是 frame-major（`[f0c0..f0c15, f1c0..]`），codec 要 codebook-major
   `(1,16,n)`。`reshape([1,16,n])` 会把 codebook 和帧交织（元素 `[cb][f] = flat[cb*n+f]`，
   而正确值在 `flat[f*16+cb]`；n=41 时全乱）。正确做法：
   `from_ints(..).reshape([n,16]).swap_dims(0,1).unsqueeze()`（与 candle 的
   `codes.t().unsqueeze(0)` 等价）。症状：**codes 逐位一致但 PCM 完全不对**
   （波形相关度 ~0、音量差 3.5×），非常隐蔽——Phase 1 只验 codes 抓不到。
3. **`conv_transpose1d` 在 burn 0.22 可用**：权重布局 `(in, out, k)`（与 candle
   一致），`ConvTransposeOptions`；EnCodec 语义 = 裸 conv + 右侧裁 `k-stride` 样本。
4. **EnCodec causal conv 精确 padding**：左侧 pad `kernel_eff - stride` + 右侧
   `extra padding`（整除补长）——burn 的 `PaddedConvOptions` 不支持这种不对称左右
   补法，手动 `Tensor::cat` 拼零再裸 `ConvOptions` 卷积（多花 2-3 次 kernel launch，
   正是 codec 慢的来源之一）。
5. **`scatter_nd` 的 `IndexingUpdateOp::Assign` 可用**（rep-penalty 乘子逐帧打散用），
   而 `select_assign` 只有 Add（ASR spike 已记录）——同一更新语义两个 API 支持度不同。
6. **`squeeze_dim<D2>` 的 D2 是输出 rank**：`wav` (1,1,T) 要
   `squeeze_dim::<2>(0).squeeze_dim::<1>(0)` 连剥两维，不是 `squeeze()`（后者删所有
   size-1 维）。
7. **权重名嵌套**：codec 的 RVQ 前缀必须以 `decoder.` 开头（`decoder.quantizer.*`），
   不是 `quantizer.*`；conv 权重在 `.conv.weight/.conv.bias` 下（与 candle 同）；
   codebook 用 `embedding_sum`（encoder 侧才叫 `embed_sum`）；
   `cluster_usage` 归一要在 f32 上 clamp 后 `unsqueeze_dim::<2>(1)`（unsqueeze 插最前，
   位置 1 要显式）。
8. **sliding-window causal mask / suppress bias**：`from_floats` + `reshape([1,1,t,t])`
   需 `Tensor::<2,Float>` 注解推 rank；suppress bias 用 `mask_fill(-inf)` 盖
   `[2048,3072)` 除 `codec_eos(2150)`（与 candle 相同）。
9. **GPU gumbel-max 采样**：`Tensor::random_like` Uniform(1e-7,1) →
   `-log(-log(u))` → `argmax(logits + gumbel)`，f32 上算；每帧仅 1 次标量读回
   （1 code0 + 15 predictor u32），与 candle 的同步点一致。
10. **tokenizer**：Qwen2 BPE 走 vocab+merges 文件路径（sentencepiece 风格的特殊
    token 由模型 config 提供，无需额外加载）。

11. **codec encoder 权重是双层 `encoder.` 前缀**：`encoder.encoder.layers.*`（SEANet）、
    `encoder.encoder_transformer.layers.*`、`encoder.downsample.conv.weight`、
    `encoder.quantizer.{semantic,acoustic}_residual_vector_quantizer.*` —— candle 的
    `vb.pp("encoder")` 与外层前缀叠加；encoder 侧 codebook 用 `embed_sum`（decoder 侧才叫
    `embedding_sum`）。
12. **ASP 的加权 std 必须是 `(x-mean)²·m`**（softmax 权重乘在平方**后**）—— 先乘再平方
    会导致 embedding 整体 ~0.65× 缩放（见 Phase 4）。
13. **speaker encoder 的 Hann 窗是 torch `hann_window(periodic=True)`**
    （`0.5·(1-cos(2πn/N))`），不是 candle `create_hann_window` 的公式（≈periodic=False）——
    两者差 ~0.002，虽然后来证明不是主因。
14. **RVQ 最近邻 argmin 有近平票**：`dists = a²+b²-2ab` 在候选接近时 catastrophic
    cancellation，top-2 距离差可到 1e-3（绝对距离 ~O(100)），F32 舍入即可翻转 —— 与
    talker 的 logits 平票同属"非移植 bug"类差异，采样式（gumbel）下无影响。
15. **ReflectConv 的 `.conv` 前缀由调用方决定**：`TimeDelayNetBlock` 加 `.conv`，
    `SqueezeExcitationBlock` 的 conv1/conv2 和 `fc` 不加（镜像 candle 的 `vb.pp` 结构）。
16. **长 prompt 下 decode 慢**：eager attention 每帧重建 (seq,seq) mask，无因果优化；
    90 帧 ref → ~100 token prompt → 每帧 ~3× 慢（candle 同条件 1.8×）。优化方向：
    mask 复用 / 滑动 attention。

### Phase 5 — Q4_K 量化（自定义 cubecl GEMV kernel）

**实现**：`src/qmat.rs`（量化 + kernel）+ `src/quant_talker.rs`（QuantDecoderLayer）。
权重按 32 元素块量化（f32 min + f32 scale=(max-min)/15 + 4 u32 打包 32 个 4-bit 值，
0.75 B/elem vs f16 2 B），每层 7 个 GEMV（q/k/v/o/gate/up/down）替换 f16 matmul，
norms/attention 保持原 dtype。kernel 是 `#[cube(launch_unchecked)]` 自定义 GEMV：
每输出行一个线程，块内解包累加，激活 F32（与 candle QMatMul 相同约束）。
通过 `Tensor::try_into_primitive::<burn_wgpu::Metal>()` 与 burn 的 GPU 上下文共享 buffer。

**验证**：
- kernel 数值：随机权重 GEMV vs CPU rel err **3.3e-3**（Q4 精度内）；延迟 **8µs/launch**
  （独立 bin；集成路径含 burn 包装 ~200µs）
- 真实权重：单块 dequant 误差可达 0.015 —— 值域大的块（±0.22）scale 大，是 **Q4 的固有
  精度**，非实现 bug
- greedy codes：**burn Q4K vs f16 2.08% 一致；candle Q4K vs BF16 1.83% 一致** —— 行为
  一致（连第一个分歧点都相同：frame0 cb2 933→412）。Q4 权重噪声使 greedy argmax 大面积
  翻转，是量化固有现象（candle 同）；**采样式**（实际使用）音频正常（RMS 4033，无 babble）
- 端到端采样 RTF ~2.8（与 f16 同级）

**性能结论（重要）**：量化**没有速度收益**（gen 128ms/帧 vs f16 113ms）。原因：burn 的
m=1 每步成本是 **op/launch 数瓶颈**（每层 ~20 个 op × 5 层 × 16 步/帧 × ~40µs），
带宽不是瓶颈 —— Q4K 只省 matmul 的带宽，动不了 op 数。candle 的 Q4_K 收益（0.53）来自
其 kernel 效率（带宽是真实瓶颈，且 predictor 量化也只省 4% —— candle 自己的注释）。
**burn Q4K 的价值 = 内存**，实测（`--bench-gen` 的 rss 打印，drop CPU 权重表后）：
f16 路径 **1213 MiB** vs Q4K **408 MiB**。`new_quant` 只持有 Q4K 块 + 共享小张量
（embeddings/norms/heads）；早期版本先 `Self::new` 全量加载 f16 28+5 层再叠加 Q4K
（驻留 ≈ f16 + Q4K 两份，README 的 2.7× 不成立）—— 已修：`new_inner(build_layers=false)`
使 quant 路径根本不加载 f16 层权重（GPU 侧 Metal 统一内存不完全计入 RSS，权重实际差
~2.6×：f16 3.4 GB → Q4K 1.3 GB，RSS 只反映其中一部分）。CPU 权重表（f16 3.4 GB）在
模型构建完成后 `drop`（仅构建期需要），各阶段 RSS 随加载打印到 stderr。

**新坑**：
17. **CubeDim 不能超过 wgpu 的 max units/cube（1024）**：2560 线程的 `CubeDim::new_1d(2560)`
    静默失败（kernel 完全不执行，输出全 0，无报错）。必须 `CubeCount × CubeDim(≤1024)`。
18. **burn metal feature 的 backend 类型是 `Metal`（MslCompiler），不是 `Wgpu`（AutoCompiler）**：
    `try_into_primitive::<Wgpu>()` 报 BackendMismatch("Expected Wgpu tensor, got variant: Metal")。
19. `try_into_primitive`/`from_primitive` 需要 burn-tensor 的 **`extension` feature**
    （`BackendPrimitive` trait，路径 `burn_tensor::BackendPrimitive`）。
20. **Q4KMatmul 只支持 s=1**（kernel 是单 token GEMV）；prefill（s>1）按 token 循环
    （一次性成本，可接受；batched kernel 是后续优化）。

### Phase 6 — burn fusion（mlx-style graph fusion，`--features fusion`）

参考 mlx-audio 的核心结论：**MLX 的快来自 lazy graph + 自动融合**（每帧 op 链融合成
少数 kernel），不是量化。burn 的等价物是 **burn-cubecl-fusion 引擎**（opt-in feature，
我们之前一直没开 —— 这是 m=1 每步慢的真正原因）。

**复现方法（评审要求，harness 已落进仓库）**：`--bench-gen <N>` 跑 N 帧完整 AR 循环
（talker + code0 采样 + 15 步 predictor + KV cache 更新），所有采样读回推迟到循环
结束的**一次**同步（跳过 EOS，强制 greedy）。这是"真实生成减去每帧 readback"的测量：

```bash
CARGO_TARGET_DIR=../target-shared cargo build --release               # 非融合
CARGO_TARGET_DIR=../target-shared cargo build --release --features fusion
../target-shared/release/burn-tts-spike <model-dir> "text" /tmp/x.wav --bench-gen 60
```

**实测（f16，greedy 60 帧，同机同构建）**：

| 构建 | ms/帧 | 相对 |
|---|---|---|
| 非融合 | 113 | 1× |
| `--features fusion` | **当前不可复现**（见下） | — |

非融合 bench 的 code0 头部与 `BURN_TTS_GREEDY=1` 生成逐位一致（确定性 ✓）。

**状态（截至评审后复测，2026-08-01）**：
- `[features] fusion = ["burn/fusion", "burn-wgpu/fusion"]`；`--features fusion` 构建。
  融合下 Q4_K 自定义 kernel 不可用（FusionTensor 无原始 buffer handle），
  `--talker-quant`/`--qmat-test` 已 cfg 门控禁用。
- **引擎 bug（burn 0.22-pre 本地 fork）比此前记录的更宽**：最初以为只有"每帧 readback
  打断优化图"触发，实测（`--bench-gen`，单次最终 readback）**同样崩溃**。带埋点复测
  确认完整根因链：
  1. **第一 panic（真根因）**：cached plan 复用到新 queue 时 fused kernel 启动失败
     —— `burn-cubecl-fusion/src/engine/launch/output.rs:207` 输出句柄
     `unwrap(None)`（10-op composed plan 在 lockstep 队列上首次执行即崩，三表
     `global/relative/operations` 长度一致）。**fused kernel 的跨帧重放句柄解析在
     这个 AR streaming 场景坏了** —— 这是"plan 缓存重复执行"的实质。
  2. **panic 损坏队列**：panic 在 `execute_block_optimization` 的
     `core::mem::swap` 之后、`self.operations = operations` 之前展开 —— UnfusedOp
     列表被清空而 `global`/`relative` 保留（实测：下一次派发 `global 12,
     operations 1`）→ 此后所有派发看到损坏队列 → `ordering.rs:49` 连锁 panic
     （`Ordering is bigger than operations`，循环到 OOM 被 SIGKILL）。
  3. `Policy::action` 对 `found` plan 无长度校验是**独立的理论风险**（`action_sync`
     有 `len == len` 校验、lazy 路径没有），但本次崩溃中不是触发点。
  `device.sync()` 前置无效。**结论：12.6ms/帧（8.7×）目前无法通过仓库内任何路径
  复现** —— 该数字来自当时未提交的测量（可能依赖当时本地 patch 过的 burn）。
- 修复方向（已确认，上游级）：(a) fused kernel 重放的句柄/view 解析
  （output.rs 的 `handle.unwrap()` 需对跨队列重放安全）；(b)
  `execute_block_optimization` 的 panic 安全（swap 后必须恢复 `self.operations`，
  或用 RAII guard 保证 unwind 不破坏队列）；(c) `Policy::action` 对 `found` 加
  `length <= operations.len()` 校验。修复后需用 `--bench-gen` 复测 fusion 端到端。

**结论（校准后）**：burn 的 m=1 性能正解 = **fusion**（方向正确，机制上成立 ——
融合消掉 op 数瓶颈）；Q4K 量化在 burn 上无速度收益（op 数瓶颈，fusion 才是解药）；
但 **burn 0.22-pre 的 fusion 引擎在 AR streaming 场景有缺陷，当前 fusion 构建连
无 readback 的 bench 都无法跑通，完整生成更不可用**。引擎 bug 修复后，gen 端到端
可望从 2.5 → ~0.3 —— **RTF 0.32 是外推，不是实测**（假设 bench 的 ms/帧直接迁移到
真实生成）；"性能逆天"的对外说法应撤回，改为"方向已验证 + 潜力可复现测量
（`--bench-gen`）+ 引擎 bug 阻塞交付"。

### Phase 7 — fusion 缓存重放崩溃的修复（2026-08-01,评审期实测）

**症状回顾**（Phase 6）：`--features fusion` 下真实生成必崩
（`output.rs:207` 相对 shape id 缺失 → panic → `mem::swap` 队列损坏 →
`ordering.rs:49` 连锁 → OOM SIGKILL）。根因：plan store 全局共享，但每个 plan
内嵌的 relative id 是构建时 converter 的编号；每次块执行后队列
`reset_relative()` 重编号，旧 plan 重放时编号失效。

**修复（burn fork 本地 patch，5 文件）**：
- **核心**：`ElemwiseOptimization::run` 在 fused launch 前校验所有 block 的
  `shape_ref` dims 能否在当前 context 的 `shapes_relative2global` 解析；
  **不可解析 → 走引擎预留的 unfused fallback 逐个执行，而不是 panic**。
- 配套：`Policy::action` found 路径精确跨度校验（plan ops + trigger ops == 当前
  segment）、`action_sync` 全 op 校验（原来只查长度）、探索复用时用新 optimization
  覆盖旧 plan、`ExecutionPlanStore::clear()`（供 `BURN_FUSION_NO_CACHE` 门控）。

**实测（同机夹逼，M4，`--max-frames 20` greedy，真实生成含每帧 readback）**：

| 构建 | gen ms/帧 | RTF |
|---|---|---|
| 非融合 f16（负载 6.2 / 3.6 两轮） | 100.5 / 97.5 | 2.27 / 2.23 |
| **融合 + 修复（负载 4.4）** | **94.0** | **2.15** |

- 零崩溃、零 shader 失败；codes 首 8 帧与非融合 greedy 逐位一致；跨运行确定。
- fallback 频率 ~1.4 次/帧，**始终是同一个过期 plan**（apply_rope，shape_ref
  `[0,1,6,2]` 缺 dim 6），其余块全部正常 fused。
- 6 帧内 **49562 次 plan 缓存命中 / 249 次探索** —— 融合覆盖巨大，净收益 ~5%
  （省掉的 launch 开销被融合自身的 per-op 簿记部分对冲）。
- **校准：8.7×（12.6ms/帧）依然不可复现**。融合收益真实但有限（~5%），远达不到
  作者原声称的量级；该数字依赖未保存的本地 burn patch。

**新坑（续 Phase 6）**：
21. **融合 kernel 的 shape 引用必须做执行期校验**：plan 的 relative shape id 在
    重放时可能失效（队列重编号）。校验失败要**回退 unfused 执行**（用引擎的
    `fallback` 钩子，`fallback(i)` 逐个执行第 i 个 op），而不是 panic——fallback
    钩子就是为这个设计的，elemwise 原实现没用它。
22. `fallback(i)` 的 `i` 是 plan 内 op 序号（0..len），不是 segment 位置；
    传 `self.len` 当单个调用会越界（ordering `[0,1,2]` 下标 3 → OOB）。

**补丁归档**：`burn-fusion-fix.patch`（本目录）是 burn fork 上 5 文件的完整 diff
（`ElemwiseOptimization` stale-shape fallback + stale-plan 作废 + policy 校验 +
`BURN_FUSION_NO_CACHE` 门控 + `ExecutionPlanStore::clear`），可直接
`git apply` 到 tracel-ai/burn 的 `610991889` 附近版本。优化后 fallback 频率
从 ~1.4 次/帧降到 **~0.1 次/帧**（10+10 帧仅 1 次：首次 stale 触发后缓存作废，
下一个同样式按当前编号重新探索并正常 fused）。

### Phase 8 — 最终裁定：burn 迁移结论（2026-08-01）

**三实现 RTF 对比（同模型 Qwen3-TTS-12Hz-1.7B，M4，含每帧 readback + codec）**：

| 实现 | RTF（量化） | RTF（bf16/f16） |
|---|---|---|
| mlx-audio（参考） | **0.47** | 0.86 |
| candle（tiny-cpm 现路径） | **0.56** | 0.98 |
| burn + fusion 修复（本 spike） | 无量化路径 | **~2.1–2.5** |

**结论**：

1. **burn 移植在正确性上成立**：核心路径（talker/predictor/codec/speaker-encoder/GPU
   采样）全部跑通，codes 与非融合逐位一致；Q4_K 自定义 GEMV、--ref 克隆、融合缓存
   重放崩溃均已修复（`burn-fusion-fix.patch` 已归档）。
2. **burn 在性能上不达标**：融合修复后 ~2.1–2.5，仍为 candle bf16 的 ~2.2×、
   mlx-audio 的 ~4.5×。差距来自基线（burn 非融合 2.5 vs candle 1.0）+ 融合 kernel
   通用代码生成 vs 手写调优 + codec 未融合 + 每帧 ~4100 次 plan 命中的 CPU 簿记。
3. **8.7× / RTF 0.3 的原始宣称不成立**：即使融合完美工作，bench（无 readback）数字
   也低估真实生成（每帧 readback + codec）的成本。
4. **建议**：tiny-cpm 的 TTS 性能路径保持 candle（bf16 0.98 / Q4_K 0.56）；本 spike
   的价值 = 移植验证 + fusion 修复 patch（`burn-fusion-fix.patch`）+ 可复现基准
   （`--bench-gen`）。若要在 burn 上追平 candle，需另开分支做引擎级优化（cubecl
   fused kernel 调优、codec 融合、降低 plan 簿记、等待 burn 0.22 stable），
   预计投入以周计且结果不确定——不在本 spike 范围。

**交付物清单**：
- `burn-fusion-fix.patch` — burn fork 5 文件修复 diff（`git apply` 可复用）
- `--bench-gen <N>` — 无 readback 生成基准（复现/验证用）
- `--qmat-test` — Q4K GEMV 数值/延迟对照
- Phase 1–7 完整验证记录（逐位对比、基准表、踩坑）

### Phase 9 — codec 卷积优化（burn-perf 分支，2026-08-01）

**症状**：codec（chunked_decode）占全流程 RTF 的近一半（~1.0–1.1），其中
`blocks` 循环（4 个 DecoderBlock）独占 ~1.6s（20 帧）：`conv_transpose1d`
（stride 8/5/4/3）每个 130–190ms，residual 的 k7 dilated causal conv 在
T=12800/38400 时每个 77–97ms。

**根因**：burn 的 conv 走 im2col 物化（大 T 时 ~百 MB 中间矩阵，带宽爆炸）；
`conv_transpose1d` 内部同样物化零插值输入。且 burn matmul 在 m<1000 时
occupancy 极低（m=128: ~2 GFLOP/s，m=12800: ~56 GFLOP/s）。

**修复**（`src/codec.rs`，`BURN_TTS_OLD_CONV=1` 可回退旧路径）：
- `causal_conv_shifted`：stride-1 causal conv = k 次"移位 + 通道 matmul"
  （dense,groups=1）或逐元素乘（depthwise）——**无 im2col 物化**。
  普通 causal conv kernel 需**翻转**（cross-correlation 语义）；
  转置卷积等价（零插值 + conv）kernel **不翻转**（convolution 语义）。
- `CausalTransConv`：stride S 转置卷积 = 零插值（(T-1)*S+1）+ 右补 k-1 +
  shifted causal conv，仅当插值后长度 ≥2048（matmul occupancy 足够）；
  小长度保留 `conv_transpose1d`（m-bound matmul 反而更慢）。
- `CausalConv`：仅当 T ≥2048 用 shifted（小 T 的 tap matmul 是 m-bound 病态）。

**验证**：新旧路径 PCM 逐点 max|d| = **1.3e-6**（fp32 舍入级，长度逐位相等）；
`src/bin/tconv_bench2.rs` / `conv_shifted_bench.rs` 为等价性回归基准
（max|d| ~1e-6，转置卷积 4–5×、大 T 残差 conv 2–4×）。

**实测（块级内部 prof，20 帧，同次运行内对比，避开负载噪声）**：

| 块 | 旧路径 | 新路径 |
|---|---|---|
| trans T=640（s8→3200） | 184ms | **39ms** |
| trans T=3200（s4→12800） | ~180ms | 73ms |
| trans T=12800（s3→38400） | ~180ms | 66ms |
| residual T=3200（k7 dil） | ~95ms | **17ms** |
| residual T=12800/38400 | 77–97ms | 62–68ms |
| blocks 合计 | ~1600ms | **~1070ms**（-33%） |

codec-only RTF 约 2.3 → 2.0；全流程 codec 部分 RTF ~1.07 → ~0.85（负载
波动大，内部块级对比为准）。**残余**：T=80 trans（192ms，m-bound 无法移位）、
T=640 residual（92ms，im2col 但 T 太小）、shifted matmul 在 codec 内比隔离
bench 慢 ~4–5×（非连续输入 + fusion 组合），是 burn kernel 效率上限。

### Phase 10 — talker/predictor CPU-bound 定位（burn-perf 分支，阶段 2 结论）

**测量**（`BURN_TTS_GENPROF=1`，gen 循环内拆分 wall / GPU / CPU + fwd28 /
predictor / other）：

| 项 | 值（负载 ~7–13） |
|---|---|
| gen wall | 90–94 ms/帧 |
| GPU（每帧 readback 等待） | **1.1–3.6 ms/帧（~3%）** |
| CPU | **87–91 ms/帧（~97%）** |
| └ fwd28（28 层 talker） | 30 ms/帧 |
| └ predictor（15 步 × 5 层） | **59 ms/帧（65%）** |
| └ 其他（采样/embed/rep） | 4 ms/帧 |

predictor 每步 ~3.4ms CPU：lm_head linear 0.01ms + sample 0.02ms + **5 层
forward 3.4ms** —— 纯 op 分发成本（~100 op × ~35µs）。

**CPU 采样**（macOS `sample`，60 帧）：热点栈 =
`cubecl ComputeClient::launch_inner` → `wgpu create_bind_group` /
`create_buffer_binding` → Metal `MTLRangeAllocator` → cubecl `memory_pool`
（coalesce/try_reserve）+ `semaphore_wait`/`mach_msg`（同步）。即**每次 kernel
launch 的绑定组创建 + 缓冲分配 + 内存池管理**，~35–50µs/op。

**结论**：
1. **talker+predictor 是 CPU-bound**（GPU 闲置 ~3%），瓶颈在 cubecl/wgpu/Metal
   的逐 kernel 启动路径（bind group + 分配），不是 matmul 本身。
2. **fusion 对 talker 无净收益**（同负载 A/B：融合 CPU 281ms/帧 vs 非融合
   231–236ms/帧——plan 簿记 ≥ launch 节省）。
3. **spike 层可动的只有 op 数**：`apply_rope` 的 `narrow` 在 burn 是 slice 拷贝
   kernel（~412 次/帧，占 CPU 采样 ~7%），消除可省 ~15–20ms/帧；predictor 的
   75 层/帧 forward 无法减 op（顺序依赖）。
4. **要根治需引擎级**：cubecl 缓存 bind group / 复用 buffer binding、减少内存池
   争用、或换后端策略——超出本分支范围。

## 文件结构（镜像 candle 侧）

- `src/config.rs` — serde config（只留用到的字段）+ `Qwen3TTSGenerationConfig` Default。
- `src/model.rs` — 共享算子：`rms_norm`/`layer_norm`（f32 方差）、`linear`、`repeat_kv`
  （沿 seq 维 cat）、`eager_attention`、`apply_rope`（rank-2 cos/sin）、`DecoderLayer`
  （Qwen3：QK-norm attention + KV cache + GatedMLP silu）、RoPE 参数。
- `src/talker.rs` — `Talker`（双 embedding + text_projection + 28 层 + codec_head +
  prompt 构造）、`CodePredictor`（5 层 × 15 路 embedding/lm_head）、`gpu_sample_token`
  （gumbel-max + suppression bias + rep-penalty scatter）、`build_prompt`（generate 与
  `bench_generate` 共用）、`bench_generate`（无 readback 生成基准，Phase 6）。
- `src/codec.rs` — Mimi decoder：CausalConv（手动 pad）、CausalTransConv（右裁）、
  SnakeBeta、EuclideanCodebookDecode、SplitRvqDecode、DecoderPreTransformer（8 层、
  sliding-window 72、LayerScale、RoPE θ1e4）、ConvNeXtBlock、DecoderResidualUnit、
  `CodecDecoder::decode` + `chunked_decode(300,25)`。
- `src/qmat.rs` — Q4_K 量化（CPU）+ 自定义 cubecl GEMV kernel（`#[cube(launch_unchecked)]`，
  Metal backend 集成 via `try_into_primitive`）。
- `src/quant_talker.rs` — `QuantDecoderLayer`（7 Q4K GEMV + norms + KV cache）+
  QuantTalkerBackbone（28 层）+ QuantPredictorBackbone（5 层）。
- `src/speaker_encoder.rs` — ECAPA-TDNN（ReflectConv、Res2Net scale8、SE、ASP、fc）+ F32。
- `src/audio.rs` — 解码（symphonia）、sinc 重采样、speaker mel 前端（reflect pad +
  periodic Hann + realfft STFT + slaney mel）—— CPU 侧 f32，与 candle 参考一致。
- `src/main.rs` — 权重加载（bf16→f16）、BPE tokenizer、warmup + 计时、`--codes-file`
  codec-only 路径、`--encode-wav`/`--spk-embed` 单模块验证入口、`--ref`/`--ref-text`
  ICL 声音克隆、`save_wav_mono`（i16，ratio 32767，与 candle 相同）。
