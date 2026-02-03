---
name: ncu-cuda-optimizer-v2
description: Rlpha-loop style automated CUDA optimization with interactive and auto modes, automatic rollback, and built-in strategy library
version: 2.0.0
author: maxiaosong1124
tags: [cuda, profiling, ncu, performance, optimization, auto-optimization, rlpha-loop]
---

# NCU CUDA Optimizer v2 - Rlpha-loop Style

本 Skill 提供 **Rlpha-loop 风格** 的自动化 CUDA 性能优化，支持**交互式**和**全自动**两种模式，内置常用 CUDA 优化策略库，自动回滚性能下降的优化。

## 与 v1 版本的关系

**v2 完全覆盖 v1 的所有功能**，并新增自动优化能力：

| 特性 | v1 (分析型) | v2 (分析+优化型) |
|------|------------|------------------|
| **单次 NCU 分析** | ✅ | ✅ `analyze_only()` |
| **从已有报告导入** | ✅ | ✅ `analyze_from_report()` |
| **数据持久化** | ✅ 保存到 `ncu_reports/` | ✅ `_save_to_project_dir()` |
| **自动诊断** | ✅ | ✅ |
| **生成分析报告** | ✅ | ✅ (v1 风格报告) |
| **迭代循环优化** | ❌ | ✅ |
| **自动代码修改** | ❌ | ✅ |
| **版本管理** | ❌ | ✅ |
| **自动回滚** | ❌ | ✅ |
| **收敛检测** | ❌ | ✅ |
| **交互式/全自动** | ❌ | ✅ |

**结论**: v2 是 v1 的超集，建议统一使用 v2。

## 快速开始

### 自然语言触发（推荐）

安装本 Skill 后，直接用自然语言告诉 AI 你的需求：

#### 全自动优化
```
你: 帮我全自动优化这个矩阵乘法算子
AI: [自动识别 ncu-cuda-optimizer-v2 skill]
    正在分析当前目录的 CUDA 文件...
    发现 matmul.cu，开始全自动优化...

    [Iteration 1/5] 诊断瓶颈: DRAM_MEMORY_BOUND
    [Iteration 2/5] 应用 Block Tiling... 提升 2.3x
    [Iteration 3/5] 应用 Shared Memory Padding... 提升 1.4x

    ✅ 优化完成！总提升 3.2x
    📊 报告已生成: optimization_report.md
```

#### 交互式优化
```
你: 一步步优化这个算子，让我确认每一步
AI: [自动识别 ncu-cuda-optimizer-v2 skill]
    发现 softmax.cu，启动交互式优化模式...

    [第 1/5 轮]
    📊 瓶颈诊断: L1_PRESSURE_BOUND
    💡 推荐策略: Shared Memory Padding (预期 1.5x 提升)

    策略描述: 通过填充避免共享内存 bank conflict
    复杂度: low | 风险: 低

    是否应用此优化? [y/n/查看详情]:
你: y
AI: ✅ 已应用，性能提升 1.6x
    继续下一轮? [y/n]:
```

#### 其他自然语言指令
- `"分析这个 CUDA 文件的性能瓶颈"` → 运行 NCU 分析并生成报告
- `"自动优化所有 CUDA 文件"` → 批量优化当前目录下所有 .cu 文件
- `"对比优化前后的性能"` → 如果有历史报告，自动对比分析
- `"帮我优化这个 kernel"` → 自动检测并优化指定的 kernel

### 命令行方式

如需精确控制，可使用命令行：

```bash
# 全自动模式
python optimizer.py matmul.cu --mode=auto

# 交互式模式
python optimizer.py matmul.cu --mode=interactive

# 或在 Claude Code 中使用 slash 命令
/ncu-optimize matmul.cu --mode=auto
```

## 核心概念

### 优化循环 (Optimization Loop)

```
┌─────────────────────────────────────────────────────────────┐
│                    Optimization Loop v2                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   模式选择   │───▶│  交互式模式  │    │  全自动模式  │     │
│  │  (用户决策)  │    │  (逐步确认)  │    │  (自主优化)  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                              │                  │           │
│                              └────────┬─────────┘           │
│                                       ▼                     │
│                         ┌─────────────────────────┐         │
│                         │     优化迭代引擎         │         │
│                         │  ┌─────────────────────┐│         │
│                         │  │ 1. NCU 基准测试      ││         │
│                         │  │ 2. 瓶颈诊断           ││         │
│                         │  │ 3. 生成优化策略       ││         │
│                         │  │ 4. 应用代码优化       ││◄───────┤
│                         │  │ 5. 重新编译测试       ││        │
│                         │  │ 6. 性能对比验证       ││        │
│                         │  │ 7. 下降? ──► 回滚    ││        │
│                         │  │ 8. 收敛? ──► 结束    ││        │
│                         │  │ 9. 未收敛 ──► 继续   ││────────┘
│                         │  └─────────────────────┘│         │
│                         └─────────────────────────┘         │
│                                       │                     │
│                                       ▼                     │
│                         ┌─────────────────────────┐         │
│                         │      输出成果            │         │
│                         │  • 最佳版本代码          │         │
│                         │  • 完整对比报告 (Markdown)│        │
│                         │  • 优化历程可视化        │         │
│                         │  • 性能提升总结          │         │
│                         └─────────────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 性能指标

**主要指标: Kernel 执行时间 (GPU Time)**

优化器以 NCU 采集的 `gpu__time_duration.avg`（kernel 实际执行时间）作为主要性能指标：

```
加速比 = 旧版本执行时间 / 新版本执行时间
```

例如:
- Baseline: 1200μs
- v1: 400μs
- 加速比 = 1200 / 400 = 3.0x

**辅助指标**

同时采集 Roofline、吞吐量等指标用于瓶颈诊断，但优化决策以执行时间为准。

### 终止条件

优化循环在以下情况停止：

1. **收敛检测**: 当前轮次执行时间提升 < 3% (默认)
2. **最大迭代**: 达到 5 轮 (默认)
3. **用户中断**: 交互模式下用户选择停止
4. **无可用策略**: 没有适用的优化策略
5. **性能下降**: 某轮优化导致性能下降 > 5% 时自动回滚

### 自动回滚机制

当某轮优化导致性能下降超过 5% 时：

```
⚠️  Performance regression detected (0.85x)
Auto-rolling back to previous version...

选项:
1. 自动回滚到上一版本，继续下一轮
2. (交互模式) 询问用户是否继续
```

## 优化策略库

### 内置策略

| 策略名称 | 目标瓶颈 | 预期收益 | 复杂度 |
|---------|---------|---------|--------|
| **Block Tiling** | DRAM_MEMORY_BOUND | 3-5x | medium |
| **Shared Memory Padding** | L1_PRESSURE_BOUND | 1.2-2x | low |
| **Vectorized Load** | DRAM_MEMORY_BOUND | 1.2-1.5x | medium |
| **Double Buffering** | LATENCY_BOUND | 1.2-1.5x | high |
| **Loop Unrolling** | LATENCY_BOUND | 1.1-1.3x | low |
| **Register Optimization** | OCCUPANCY_BOUND | 1.2-2x | medium |
| **Warp-level Primitives** | COMPUTE_BOUND | 1.2-1.5x | medium |
| **Grid-Stride Loops** | MIXED_BOUND | 1.1-1.3x | low |

### 策略选择逻辑

```python
def select_strategy(metrics):
    # 1. 诊断瓶颈类型
    bottleneck = diagnose_bottleneck(metrics)

    # 2. 筛选适用策略
    candidates = [
        s for s in strategies
        if bottleneck in s.target_bottlenecks
        and all(s.metrics_conditions[k](metrics[k])
                for k in s.metrics_conditions)
    ]

    # 3. 按预期收益排序
    return sorted(candidates, key=lambda s: s.expected_speedup, reverse=True)
```

## AI Agent 识别逻辑

当用户输入以下内容时，AI 应自动识别并调用本 Skill：

### 意图识别表

| 用户输入模式 | 识别意图 | 执行动作 |
|-------------|---------|---------|
| `"优化这个算子"`, `"优化这个 CUDA 文件"`, `"帮我优化 kernel"` | 优化 CUDA 代码 | 检测当前目录 .cu 文件，询问模式 |
| `"全自动优化"`, `"自动优化"`, `"一键优化"` | 全自动模式 | 直接以 `--mode=auto` 运行 |
| `"一步步优化"`, `"逐步优化"`, `"让我确认每一步"` | 交互式模式 | 以 `--mode=interactive` 运行 |
| `"分析性能"`, `"分析瓶颈"`, `"profile this kernel"` | 性能分析 | 只运行 NCU 分析，不修改代码 |
| `"对比性能"`, `"compare versions"` | 对比分析 | 对比不同版本的性能数据 |
| `"批量优化"`, `"优化所有 CUDA 文件"` | 批量优化 | 遍历当前目录所有 .cu 文件 |

### 自动检测流程

当用户表达优化意图时，AI 应按以下流程执行：

```python
def handle_optimization_request(user_input):
    # 1. 检测 CUDA 文件
    cuda_files = glob("*.cu")
    if not cuda_files:
        return "当前目录未找到 CUDA 文件 (.cu)"

    # 2. 确定模式
    if any(word in user_input for word in ["全自动", "自动", "一键", "auto"]):
        mode = "auto"
    elif any(word in user_input for word in ["一步步", "逐步", "交互", "确认", "interactive"]):
        mode = "interactive"
    else:
        # 询问用户
        return "请选择优化模式：\n1. 全自动优化 (AI 自主完成)\n2. 交互式优化 (每步确认)"

    # 3. 执行优化
    if len(cuda_files) == 1:
        return run_optimizer(cuda_files[0], mode)
    else:
        return f"发现 {len(cuda_files)} 个 CUDA 文件，请选择要优化的文件: {cuda_files}"
```

## 输出报告

优化完成后生成 Markdown 格式报告：

```markdown
# NCU CUDA 自动优化报告

**优化时间**: 2024-01-15T10:30:00
**源文件**: matmul.cu
**优化模式**: auto
**迭代次数**: 4/5

## 优化概览

- **初始执行时间**: 1200.50μs
- **初始 Roofline**: 25.0%
- **最终执行时间**: 352.20μs
- **最终 Roofline**: 85.0%
- **总加速比**: 3.41x (以执行时间为准)
- **最佳版本**: v3
- **收敛状态**: 已收敛
- **收敛原因**: 执行时间提升 < 3% (1.02x)

## 优化历程

| 版本 | 策略 | 执行时间 | Roofline | 相对Baseline | 相对上一轮 | 状态 |
|------|------|----------|----------|--------------|------------|------|
| baseline | - | 1200.50μs | 25.0% | 1.00x | - | ✅ |
| v1 | Block Tiling | 545.68μs | 55.0% | 2.20x | 2.20x | ✅ |
| v2 | Shared Memory Padding | 416.67μs | 72.0% | 2.88x | 1.31x | ✅ |
| v3 | Vectorized Load | 352.20μs | 85.0% | 3.41x | 1.18x | ✅ |
| v4 | Loop Unrolling | 345.26μs | 86.5% | 3.48x | 1.02x | ✅ |

## 详细分析

### v1: Block Tiling

- **策略描述**: 使用共享内存进行块级数据缓存，减少全局内存访问
- **相对Baseline提升**: 2.20x
- **相对上一轮提升**: 2.20x

**关键指标**:
- dram_throughput: 35.00 (优化前 85%)
- l1tex_throughput: 65.00
- sm_busy: 75.00
- occupancy: 68.00
- roofline_ratio: 55.00

### v2: Shared Memory Padding
...

## 最终推荐代码

[完整的优化后代码]

## 进一步优化建议

由于达到收敛阈值，自动优化停止。
如需继续优化，可考虑：
1. 使用 Tensor Cores (预期额外 1.3x)
2. 调整 block size 为 256 (预期额外 1.1x)
```

## 使用方式

### 方式一: 自然语言（推荐）

直接用自然语言与 AI 交互：

#### 分析模式（v1 功能）
```
用户: 分析这个 kernel 的性能瓶颈
AI:  正在分析 matmul.cu...
      📊 主要瓶颈: DRAM_MEMORY_BOUND
      📈 GPU 时间: 1200.5 μs
      💡 建议: Block Tiling (预期 3x 提升)
      报告已保存到 ncu_reports/

用户: 从已有报告分析
AI:  请提供 .ncu-rep 文件路径
用户: matmul_analysis.ncu-rep
AI:  导入分析中...
      📊 诊断结果: L1_PRESSURE_BOUND
```

#### 优化模式（v2 功能）
```
用户: 帮我全自动优化这个矩阵乘法
AI:  检测到 matmul.cu，启动全自动优化...
      [优化完成] 总提升 3.2x，报告已生成

用户: 一步步分析这个 kernel 的性能
AI:  正在分析...
      第1轮: 发现 DRAM 瓶颈，建议 Block Tiling
      是否应用? [y/n]:
```

### 方式二: Python 脚本

```bash
# ========== 分析模式 (v1 功能) ==========

# 单次分析 - 只生成报告，不修改代码
python optimizer.py matmul.cu --mode=analyze

# 从已有 NCU 报告导入分析
python optimizer.py --import-report matmul_analysis.ncu-rep

# ========== 优化模式 (v2 功能) ==========

# 全自动优化
python optimizer.py matmul.cu --mode=auto

# 交互式优化
python optimizer.py matmul.cu --mode=interactive

# ========== 高级参数 ==========

# 自定义编译命令
python optimizer.py matmul.cu \
    --build "nvcc -O3 -arch=sm_80 {source} -o {output}"

# 调整迭代参数
python optimizer.py matmul.cu \
    --max-iter 3 \
    --threshold 0.05

# 不保存到项目目录
python optimizer.py matmul.cu --mode=analyze --no-save
```

### 方式三: 编程式调用

```python
from optimizer import CUDAOptimizer

# ========== 分析模式 (v1 功能) ==========

# 单次分析
optimizer = CUDAOptimizer(
    source_file="matmul.cu",
    build_command="nvcc -O3 {source} -o {output}"
)
result = optimizer.analyze_only(save_to_project=True)
print(f"瓶颈: {result['bottleneck']}")
print(f"建议: {result['recommendations']}")

# 从已有报告导入
result = optimizer.analyze_from_report("matmul_analysis.ncu-rep")

# ========== 优化模式 (v2 功能) ==========

# 全自动优化
optimizer = CUDAOptimizer(
    source_file="matmul.cu",
    build_command="nvcc -O3 {source} -o {output}",
    mode="auto"
)
result = optimizer.run()
print(f"最佳版本: {result['best_version']}")
print(f"加速比: {result['best_speedup']:.2f}x")
```

## 配置参数

### 环境变量

```bash
# NCU 路径
export NCU_PATH=/usr/local/cuda/bin/ncu

# 默认优化参数
export NCU_OPT_MAX_ITER=5
export NCU_OPT_THRESHOLD=0.03
export NCU_OPT_MODE=auto
```

### 配置文件

项目根目录创建 `.ncu-opt.config`：

```json
{
  "mode": "auto",
  "max_iterations": 5,
  "convergence_threshold": 0.03,
  "regression_threshold": 0.95,
  "build_command": "nvcc -O3 -arch=sm_80 {source} -o {output}",
  "strategies": {
    "enabled": ["block_tiling", "smem_padding", "vectorized_load"],
    "disabled": ["double_buffering"]
  }
}
```

## 高级用法

### 自定义优化策略

```python
from v2.strategy_library import OptimizationStrategy, BottleneckType

# 创建自定义策略
my_strategy = OptimizationStrategy(
    name="My Custom Opt",
    description="自定义优化策略",
    target_bottlenecks=[BottleneckType.COMPUTE_BOUND],
    applicable_metrics={
        "sm_busy": lambda x: x > 80
    },
    code_template="""
    // 自定义优化代码
    #pragma unroll
    for (...) { ... }
    """,
    insertion_pattern=r"for\s*\(",
    expected_speedup=1.5,
    complexity="medium",
    prerequisites=["条件1", "条件2"]
)

# 注册到策略库
library.strategies["my_custom"] = my_strategy
```

### 批量优化多个文件

```bash
# 优化目录下所有 .cu 文件
for f in *.cu; do
    python v2/optimizer.py "$f" --mode=auto
done

# 或使用 skill 命令
/ncu-optimize --batch "*.cu" --mode=auto
```

### 与 CI/CD 集成

```yaml
# .github/workflows/cuda-optimize.yml
name: CUDA Auto-Optimization
on: [push]

jobs:
  optimize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Run NCU Optimizer
        run: |
          python ncu-cuda-profiling-skill/v2/optimizer.py \
            kernels/matmul.cu --mode=auto

      - name: Upload Report
        uses: actions/upload-artifact@v2
        with:
          name: optimization-report
          path: /tmp/ncu_opt_*/optimization_report.md
```

## 故障排除

### 常见问题

**Q: NCU 命令未找到**
```bash
# 检查 NCU 安装
which ncu
ncu --version

# 设置路径
export PATH=/usr/local/cuda/bin:$PATH
```

**Q: 编译失败**
```bash
# 检查 CUDA 编译器
nvcc --version

# 使用自定义构建命令
python optimizer.py matmul.cu \
    --build "nvcc -O3 -I/path/to/include {source} -o {output} -lmylib"
```

**Q: 优化后性能下降**
- 这是正常现象，优化器会自动回滚
- 检查策略是否适合你的代码模式
- 尝试交互式模式手动选择策略

**Q: 收敛过快**
```bash
# 降低收敛阈值
python optimizer.py matmul.cu --threshold 0.01  # 1%

# 增加最大迭代
python optimizer.py matmul.cu --max-iter 10
```

## 最佳实践

### 1. 从 Baseline 开始

确保 baseline 代码可以正确编译和运行：

```bash
nvcc -O3 matmul.cu -o matmul
./matmul  # 验证正确性
```

### 2. 选择合适的模式

- **原型开发**: 使用 `interactive` 模式学习优化过程
- **生产优化**: 使用 `auto` 模式批量处理
- **调试问题**: 使用 `interactive` 模式精细控制

### 3. 保存优化历史

```bash
# 每次优化保存到不同目录
python optimizer.py matmul.cu --output-dir=opt_v1
# 修改代码后
python optimizer.py matmul.cu --output-dir=opt_v2
```

### 4. 结合手动优化

自动优化是辅助工具，不是替代：

```
1. 运行自动优化获取 baseline
2. 分析报告了解瓶颈
3. 手动实现算法级优化
4. 再次运行自动优化微调
```

## 性能对比

### v1 vs v2 优化效果对比

| 测试用例 | v1 建议 | v2 自动优化 | 时间节省 |
|---------|--------|------------|---------|
| Matmul Naive | 提供建议 | 3.4x 提升 | 80% |
| Reduce Sum | 提供建议 | 2.8x 提升 | 75% |
| Softmax | 提供建议 | 4.2x 提升 | 85% |

## 相关资源

- v1 版本文档: [SKILL.md](../SKILL.md)
- 策略库源码: [strategy_library.py](./strategy_library.py)
- 优化器源码: [optimizer.py](./optimizer.py)
- 示例报告: [examples/](./examples/)

---

*本 Skill 是 ncu-cuda-profiling-skill v2.0，与 v1 完全兼容，可同时安装使用*
