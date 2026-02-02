---
name: ncu-cuda-profiling
description: Automated NCU (Nsight Compute) profiling workflow with full metrics collection and persistent storage
version: 1.0.0
author: maxiaosong1124
tags: [cuda, profiling, ncu, performance, optimization]
---

# NCU CUDA 自动化性能分析

本 Skill 提供完整的自动化 NCU 性能分析流程，支持**全量指标采集**和**持久化存储**。

---

## 🚀 快速开始

### 推荐: 一键完整采集

```bash
# 使用 --set full 采集所有指标，并持久化保存
ncu --set full \
    -o <report_name> \
    --target-processes all \
    ./your_kernel

# 示例
ncu --set full -o matmul_analysis --target-processes all ./matmul0_perf

# 自动生成:
# - matmul_analysis.ncu-rep    (NCU 报告文件)
# - matmul_analysis.csv        (CSV 格式指标)
```

### 指标提取 (采集后)

```bash
# 从已保存的报告提取关键指标 (无需重新运行 kernel)
ncu --import matmul_analysis.ncu-rep --print-summary per-kernel

# 导出为 CSV
ncu --import matmul_analysis.ncu-rep --page raw --csv > metrics.csv
```

---

## 📋 AI 分析流程

当用户提供 NCU 数据时，AI 按以下流程处理：

### Phase 1: 数据获取 (优先顺序)

**情况 A: 用户提供了 .ncu-rep 文件**

```bash
# 直接导入已有报告
ncu --import <file.ncu-rep> --print-summary per-kernel
```

**情况 B: 用户需要新分析**

```bash
# 完整采集并持久化
ncu --set full -o <report_name> --target-processes all ./kernel
```

**情况 C: 用户提供了截图/文本**

- 直接提取其中的数值进行分析

### Phase 2: 数据持久化

AI 会自动保存分析数据到项目目录：

```
project_root/
├── ncu_reports/                    # NCU 报告目录
│   ├── matmul_analysis.ncu-rep    # 完整报告
│   ├── matmul_analysis.csv        # CSV 指标
│   └── matmul_analysis.md         # AI 分析报告
└── ...
```

### Phase 3: 自动诊断

使用决策引擎自动分析：

```python
def auto_diagnose(metrics):
    roofline = metrics.get('roofline_ratio', 0)
    dram = metrics.get('dram_throughput', 0)
    l1tex = metrics.get('l1tex_throughput', 0)
    sm_busy = metrics.get('sm_busy', 0)
    occupancy = metrics.get('occupancy', 0)
    
    if roofline < 30:
        if dram > 70:
            return "DRAM_MEMORY_BOUND"
        elif l1tex > 80 and dram < 30:
            return "L1_PRESSURE_BOUND"
        else:
            return "LATENCY_BOUND"
    elif roofline > 60:
        if sm_busy > 80:
            return "COMPUTE_BOUND"
        else:
            return "OCCUPANCY_BOUND"
    else:
        return "MIXED_BOUND"
```

---

## 📊 输出模板

```markdown
# NCU 性能分析报告

## 📁 报告信息
- **Kernel**: {kernel_name}
- **采集时间**: {timestamp}
- **报告文件**: {report_file}
- **原始数据**: {csv_file}

## 📈 执行摘要

| 项目 | 数值 |
|------|------|
| **主要瓶颈** | {bottleneck_type} |
| **置信度** | {confidence} |
| **性能** | {performance} GFLOPS |
| **优化潜力** | {potential}x |

## 📊 关键指标

### 性能指标
| 指标 | 数值 | 健康阈值 | 状态 |
|------|------|----------|------|
| Roofline 性能比 | {roofline}% | > 60% | {status} |
| SM Busy | {sm_busy}% | > 70% | {status} |
| Occupancy | {occupancy}% | > 50% | {status} |

### 内存指标
| 指标 | 数值 | 健康阈值 | 状态 |
|------|------|----------|------|
| DRAM Throughput | {dram}% | < 50% | {status} |
| L1/TEX Throughput | {l1tex}% | < 80% | {status} |
| L2 Throughput | {l2}% | < 80% | {status} |

## 🔍 诊断详情

**瓶颈类型**: {bottleneck_type}

**判断依据**:
- {reason_1}
- {reason_2}

## 💡 优化建议

### 高优先级
{high_priority_suggestions}

## 🛠️ 下一步操作

### 建议的 NCU 命令
```bash
# 优化后重新采集
ncu --set full -o {report_name}_optimized --target-processes all ./kernel_optimized
```

### 验证清单

- [ ] 实施建议的优化
- [ ] 重新运行 NCU 采集
- [ ] 对比优化前后数据

```

---

## 🔧 工具使用说明

### 完整采集 (推荐)

```bash
# 采集所有指标并保存
ncu --set full -o my_analysis --target-processes all ./kernel

# 参数说明:
# --set full          # 采集完整指标集
# -o my_analysis      # 输出文件名 (生成 my_analysis.ncu-rep)
# --target-processes all  # 监控所有进程
```

### 增量分析 (已有报告)

```bash
# 从已有报告提取特定指标
ncu --import my_analysis.ncu-rep --print-summary per-kernel

# 导出为 CSV 便于分析
ncu --import my_analysis.ncu-rep --page raw --csv > metrics.csv
```

### 自动化脚本

使用提供的自动化脚本：

```bash
cd examples/

# 全自动分析
./auto_profile.sh ./kernel report_name

# Python 分析器
python ncu_analyzer.py --import report_name.ncu-rep
```

---

## 📖 诊断规则详解

### DRAM_MEMORY_BOUND

```
IF dram_throughput > 70% AND roofline < 30%:
    诊断: DRAM_MEMORY_BOUND (置信度: HIGH)
    
    优化策略:
    1. Block Tiling (共享内存缓存)
    2. Vectorized Load (float4)
    3. Prefetching (数据预取)
```

### L1_PRESSURE_BOUND

```
IF l1tex_throughput > 80% AND dram_throughput < 30%:
    诊断: L1_PRESSURE_BOUND (置信度: HIGH)
    
    优化策略:
    1. Shared Memory Padding
    2. Data Transpose
    3. Fragment Caching
```

### LATENCY_BOUND

```
IF sm_busy < 50% AND occupancy > 60%:
    诊断: LATENCY_BOUND (置信度: HIGH)
    
    优化策略:
    1. Double Buffering
    2. Instruction-level Parallelism
    3. Loop Unrolling
```

### COMPUTE_BOUND

```
IF roofline > 60% AND sm_busy > 80%:
    诊断: COMPUTE_BOUND (置信度: HIGH)
    
    优化策略:
    1. Use FMA instructions
    2. Reduce precision (FP32 -> FP16/TF32)
    3. Tensor Cores
```

### OCCUPANCY_BOUND

```
IF occupancy < 30% AND sm_busy > 70%:
    诊断: OCCUPANCY_BOUND (置信度: HIGH)
    
    优化策略:
    1. Reduce register usage
    2. Adjust block size
    3. Use __launch_bounds__
```

---

## 🎯 优化策略速查

| 瓶颈类型 | 立即行动 | 代码示例 | 预期收益 |
|---------|---------|---------|---------|
| **DRAM_MEMORY_BOUND** | Block Tiling | `__shared__ float As[BM][BK];` | 3-5x |
| **L1_PRESSURE_BOUND** | Padding | `As[BM][BK+1]` | 1.2-2x |
| **LATENCY_BOUND** | Double Buffer | `As[2][BM*BK]` | 1.2-1.5x |
| **COMPUTE_BOUND** | FMA | `fmaf(a, b, c)` | 1.1-1.3x |
| **OCCUPANCY_BOUND** | 调整 block size | `__launch_bounds__(256, 2)` | 1.2-2x |

---

## 📚 完整 NCU 命令参考

### 推荐采集命令

```bash
# 完整采集 (推荐)
ncu --set full -o report_name --target-processes all ./kernel

# 指定 sections
ncu --section SpeedOfLight,Occupancy,LaunchStats -o report_name ./kernel

# 特定指标
ncu --metrics sm__throughput.avg.pct,dram__throughput.avg.pct -o report_name ./kernel
```

### 报告操作

```bash
# 查看摘要
ncu --import report.ncu-rep --print-summary per-kernel

# 查看详情
ncu --import report.ncu-rep --page details

# 导出 CSV
ncu --import report.ncu-rep --page raw --csv > metrics.csv

# 对比两个报告
ncu --diff report1.ncu-rep report2.ncu-rep
```

---

## ⚠️ 常见误区

1. **高 Throughput ≠ 高效率**
   - Compute + Memory Throughput 都很高但 Roofline 很低 = GPU 在"忙碌地等待"

2. **DRAM Throughput 低可能是好事**
   - 优化后 DRAM 降低说明数据在缓存中复用

3. **Occupancy 不是越高越好**
   - 目标是最小足够 occupancy 隐藏延迟

---

## 🔗 相关资源

- 自动化脚本: `examples/`
- GitHub: <https://github.com/maxiaosong1124/ncu-cuda-profiling-skill>

---

*本 Skill 支持完整的自动化 NCU 性能分析工作流，包含全量采集和持久化存储*
