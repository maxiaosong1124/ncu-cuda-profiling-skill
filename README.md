# NCU CUDA Optimizer v2

Rlpha-loop 风格的自动化 CUDA 性能优化工具。

## 功能特性

- **双模式支持**: 交互式模式和全自动模式
- **自动瓶颈诊断**: 基于 NCU 指标自动识别性能瓶颈
- **内置策略库**: 8种常用 CUDA 优化策略
- **自动回滚**: 性能下降自动回滚到上一版本
- **收敛检测**: 提升 < 3% 自动停止
- **完整报告**: Markdown 格式优化历程报告

## 安装

```bash
# 复制到 Claude Code skills 目录
mkdir -p ~/.claude/skills/ncu-cuda-optimizer-v2
cp v2/*.py v2/SKILL.md ~/.claude/skills/ncu-cuda-optimizer-v2/
```

## 快速开始

### 全自动模式

```bash
python optimizer.py matmul.cu --mode=auto --build "nvcc -O3 -arch=sm_89 {source} -o {output}"
```

### 交互式模式

```bash
python optimizer.py matmul.cu --mode=interactive
```

### 在 Claude Code 中使用

```
/ncu-optimize matmul.cu --mode=auto
```

## 优化策略

| 策略 | 目标瓶颈 | 预期收益 |
|------|---------|---------|
| Block Tiling | DRAM Memory Bound | 3-5x |
| Shared Memory Padding | L1 Pressure | 1.2-2x |
| Vectorized Load | DRAM Memory Bound | 1.2-1.5x |
| Double Buffering | Latency Bound | 1.2-1.5x |
| Loop Unrolling | Latency Bound | 1.1-1.3x |
| Register Optimization | Occupancy Bound | 1.2-2x |
| Warp-level Primitives | Compute Bound | 1.2-1.5x |
| Grid-Stride Loops | Mixed Bound | 1.1-1.3x |

## 与 v1 版本对比

| 特性 | v1 | v2 |
|------|----|----|
| 模式 | 单次分析 | 迭代优化 |
| 用户参与 | 被动接收 | 交互/自动双模式 |
| 代码修改 | 手动 | 自动 |
| 回滚机制 | 无 | 自动 |
| 收敛检测 | 无 | 有 |

## 输出示例

优化完成后生成报告：

```markdown
# NCU CUDA 自动优化报告

## 优化概览
- **初始性能**: 25.0% (Roofline)
- **最终性能**: 85.0% (Roofline)
- **总提升**: 3.4x
- **最佳版本**: v3

## 优化历程

| 版本 | 策略 | Roofline | 相对Baseline | 状态 |
|------|------|----------|--------------|------|
| baseline | - | 25.0% | 1.00x | ✅ |
| v1 | Block Tiling | 55.0% | 2.20x | ✅ |
| v2 | Shared Memory Padding | 72.0% | 2.88x | ✅ |
| v3 | Vectorized Load | 85.0% | 3.40x | ✅ |
```

## 项目结构

```
v2/
├── SKILL.md              # 详细使用文档
├── README.md             # 本文件
├── optimizer.py          # 优化器主程序
├── strategy_library.py   # 优化策略库
└── templates/            # 报告模板
```

## 注意事项

1. 需要安装 NVIDIA Nsight Compute (ncu)
2. 需要 CUDA 编译器 (nvcc)
3. 当前代码修改器为简化实现，复杂优化可能需要手动调整
4. 建议在应用优化后验证结果正确性
