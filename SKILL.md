---
name: ncu-cuda-profiling
description: Automated NCU (Nsight Compute) profiling workflow with comprehensive metrics collection, bottleneck analysis, and optimization guidance
---

# NCU CUDA 自动化性能分析 (v2 Enhanced)

本 Skill 提供完整的自动化 NCU 性能分析流程，支持**全量指标采集**、**智能瓶颈诊断**和**针对性优化建议**。

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

## 📋 标准分析流程 (改进版)

### Phase 1: 数据获取 (优先顺序)

**情况 A: 用户提供了 .ncu-rep 文件**
```bash
# 直接导入已有报告
ncu --import <file.ncu-rep> --page raw --csv > metrics.csv
```

**情况 B: 用户需要新分析**
```bash
# 完整采集并持久化
ncu --set full -o <report_name> --target-processes all ./kernel
```

**情况 C: 用户提供了截图/文本**
- 直接提取其中的数值进行分析

### Phase 2: 核心指标解析 (按优先级)

#### Step 1: GPU Speed Of Light Throughput (首要)
**判断瓶颈类型：Memory Bound vs Compute Bound**

| 指标 | 阈值 | 说明 |
|------|------|------|
| **Memory Throughput** | >80% | Memory Bound |
| **DRAM Throughput** | >80% | 显存瓶颈 |
| **Compute (SM) Throughput** | >80% | Compute Bound |
| **L1/TEX Cache Throughput** | >80% | L1 压力大 |
| **L2 Cache Throughput** | >80% | L2 压力大 |

**判断逻辑**：
```
Memory Throughput > 80% 且 Compute Throughput < 50%  →  Memory Bound（内存瓶颈）
Compute Throughput > 80% 且 Memory Throughput < 50%  →  Compute Bound（计算瓶颈）
两者都高 → 需要进一步分析 Memory Workload 和 Compute Workload
```

#### Step 2: Compute Workload Analysis
**分析 SM 计算资源利用情况**

| 指标 | 健康范围 | 说明 |
|------|----------|------|
| **Executed Ipc Active** | >0.5 | 每周期执行指令数 |
| **Issue Slots Busy** | >50% | 发射槽忙碌率 |
| **SM Busy** | >70% | SM 忙碌程度 |

**解读**：
- **SM Busy 很低**（<20%）→ 算力没被充分利用，可能原因：
  - 内存等待导致算力闲着 (Memory Dependency stall)
  - warp 数量不足 (Occupancy 低)
  - 指令依赖链过长 (Execution Dependency stall)

#### Step 3: Memory Workload Analysis
**分析 GPU 内存子系统性能**

| 指标 | 健康范围 | 说明 |
|------|----------|------|
| **Mem Busy** | <80% | 内存单元忙碌程度 |
| **L1/TEX Hit Rate** | >50% | L1/TEX 缓存命中率 |
| **L2 Hit Rate** | >70% | L2 缓存命中率 |

#### Step 4: Occupancy (占用率分析)
**分析 SM 占用情况**

| 指标 | 健康范围 | 说明 |
|------|----------|------|
| **Theoretical Occupancy** | >50% | 理论占用率 |
| **Achieved Occupancy** | >40% | 实际占用率 |

**注意**：理论 vs 实际差距大 → 工作负载不均衡或分支发散

#### Step 5: Scheduler Statistics (调度器统计)
**分析 warp 调度效率**

| 指标 | 说明 |
|------|------|
| **Active Warps** | 活跃 warp 数量 |
| **Eligible Warps** | 准备好发射的 warp |
| **No Eligible** | 每周期没有 warp 准备好 |

**解读**：No Eligible 比例高 → warp 停滞严重

#### Step 6: Warp State Statistics (Warp状态分析)
**分析 warp 停滞原因**

| Stall Reason | 说明 | 优化方向 |
|--------------|------|----------|
| **Wait** | 等待指令获取 | 检查指令缓存 |
| **Barrier** | 等待 `__syncthreads` | 减少同步点 |
| **Memory Dependency** | 等待内存操作 | 增加独立计算指令 |
| **Execution Dependency** | 等待前一指令结果 | 增加 ILP |
| **Memory Throttle** | 内存压力过大 | 优化内存访问模式 |
| **Instruction Fetch** | 指令获取延迟 | 减少代码体积 |

### Phase 3: 智能诊断 (自动决策树)

```python
def auto_diagnose(metrics):
    """
    自动诊断瓶颈类型
    
    决策树：
    1. 首先看 Speed Of Light Throughput
    2. 然后看 Occupancy 和 Scheduler Stats
    3. 最后看 Warp State Stall Reasons
    """
    memory_throughput = metrics.get('memory_throughput', 0)
    dram_throughput = metrics.get('dram_throughput', 0)
    sm_throughput = metrics.get('sm_throughput', 0)
    sm_busy = metrics.get('sm_busy', 0)
    occupancy = metrics.get('occupancy', 0)
    issue_slots_busy = metrics.get('issue_slots_busy', 0)
    
    # Level 1: 判断 Memory vs Compute
    if dram_throughput > 80 and sm_throughput < 50:
        # Memory Bound - 进一步细分
        l1_hit_rate = metrics.get('l1_hit_rate', 100)
        if l1_hit_rate < 30:
            return BottleneckType.L1_PRESSURE_BOUND
        else:
            return BottleneckType.DRAM_MEMORY_BOUND
    
    elif sm_throughput > 80 and dram_throughput < 50:
        # Compute Bound
        return BottleneckType.COMPUTE_BOUND
    
    elif sm_busy < 30 and occupancy > 50:
        # SM 空闲但 Occupancy 高 → 可能是 warp 停滞
        return BottleneckType.LATENCY_BOUND
    
    elif occupancy < 30:
        # Occupancy 低
        return BottleneckType.OCCUPANCY_BOUND
    
    else:
        return BottleneckType.MIXED_BOUND
```

---

## 📊 详细指标说明

### 1. GPU Speed Of Light Throughput

**指标含义**：GPU 极限吞吐量分析，判断是**算力瓶颈**还是**带宽瓶颈**

| 指标名 | 单位 | 说明 | 分析要点 |
|--------|------|------|----------|
| **DRAM Frequency** | Ghz | 显存频率 | 硬件固有频率 |
| **SM Frequency** | Ghz | SM 运行频率 | 硬件固有频率 |
| **Elapsed Cycles** | cycle | 经过的时钟周期数 | 总执行周期 |
| **Memory Throughput** | % | 内存吞吐量 | **>80% 表示 memory bound** |
| **DRAM Throughput** | % | 显存吞吐量 | **>80% 表示显存瓶颈** |
| **Duration** | us/ms | 执行时间 | ncu采集时间（非真实时间） |
| **L1/TEX Cache Throughput** | % | L1/Tex缓存吞吐量 | 缓存利用情况 |
| **L2 Cache Throughput** | % | L2 缓存吞吐量 | 二级缓存利用情况 |
| **SM Active Cycles** | cycle | SM活跃周期 | SM实际工作时间 |
| **Compute (SM) Throughput** | % | SM计算吞吐量 | **>80% 表示 compute bound** |

### 2. Compute Workload Analysis

| 指标名 | 单位 | 说明 |
|--------|------|------|
| **Executed Ipc Active** | inst/cycle | 每周期执行指令数 |
| **Issue Slots Busy** | % | 发射槽忙碌率 |
| **SM Busy** | % | SM 忙碌程度 |

### 3. Memory Workload Analysis

| 指标名 | 说明 |
|--------|------|
| **Mem Busy** | 内存单元忙碌程度 |
| **Max Bandwidth** | 内存带宽利用率峰值 |
| **Mem Pipes Busy** | 内存管道忙碌程度 |
| **L1/TEX Hit Rate** | L1/TEX 缓存命中率 |
| **L2 Hit Rate** | L2 缓存命中率 |

### 4. Warp State Statistics (Stall Reasons)

| Stall Reason | 说明 | 优化方向 |
|--------------|------|----------|
| **Wait** | 等待指令获取 | 检查指令缓存压力 |
| **Barrier** | 等待同步屏障 (`__syncthreads`) | 减少同步点，或使用 warp-level 原语 |
| **Memory Dependency** | 等待内存操作完成 | 增加独立计算指令，使用 __launch_bounds__ |
| **Execution Dependency** | 等待前一指令结果 | 增加指令级并行 (ILP) |
| **Memory Throttle** | 内存子系统压力过大 | 优化内存访问模式，使用共享内存 |
| **Texture** | 等待纹理操作 | 优化纹理访问模式 |
| **Constant** | 等待常量缓存 | 检查常量内存使用 |
| **Instruction Fetch** | 指令获取延迟 | 减少代码体积 |
| **Not Selected** | 有 eligible warp 但未被选中 | 正常调度行为 |
| **Sleep** | warp 处于睡眠状态 | 检查 `nanosleep` 使用 |
| **Ipc** | 每周期指令数限制 | 正常情况 |

---

## 🎯 优化策略库

### DRAM_MEMORY_BOUND (显存瓶颈)

**判断依据**：
- DRAM Throughput > 80%
- Memory Throughput > 80%
- SM Busy < 50%

**优化策略**：

| 策略 | 代码示例 | 预期收益 |
|------|----------|----------|
| **Block Tiling** | `__shared__ float As[BM][BK];` | 3-5x |
| **Vectorized Load** | `float4 vec = *(float4*)&A[i];` | 1.3-1.5x |
| **Prefetching** | `prefetch_l1(&A[next]);` | 1.1-1.3x |

### L1_PRESSURE_BOUND (L1 压力)

**判断依据**：
- L1/TEX Throughput > 80%
- DRAM Throughput < 30%
- L1 Hit Rate < 30%

**优化策略**：

| 策略 | 代码示例 | 预期收益 |
|------|----------|----------|
| **Shared Memory Padding** | `As[BM][BK+1]` | 1.2-2x |
| **Data Transpose** | 调整访问模式 | 1.1-1.5x |
| **Fragment Caching** | 寄存器缓存 | 1.1-1.3x |

### LATENCY_BOUND (延迟瓶颈)

**判断依据**：
- SM Busy < 30%
- Occupancy > 50%
- Memory Dependency stall 高

**优化策略**：

| 策略 | 代码示例 | 预期收益 |
|------|----------|----------|
| **Double Buffering** | `As[2][BM][BK]` | 1.2-1.5x |
| **Loop Unrolling** | `#pragma unroll 4` | 1.1-1.3x |
| **ILP Increase** | 独立计算指令交错 | 1.1-1.2x |

### COMPUTE_BOUND (计算瓶颈)

**判断依据**：
- SM Throughput > 80%
- SM Busy > 80%
- DRAM Throughput < 50%

**优化策略**：

| 策略 | 代码示例 | 预期收益 |
|------|----------|----------|
| **FMA Usage** | `fmaf(a, b, c)` | 1.1-1.3x |
| **Tensor Core** | `mma_sync` | 2-8x |
| **Warp Primitives** | `__shfl_down_sync` | 1.2-1.5x |

### OCCUPANCY_BOUND (占用率瓶颈)

**判断依据**：
- Occupancy < 30%
- Registers Per Thread > 64

**优化策略**：

| 策略 | 代码示例 | 预期收益 |
|------|----------|----------|
| **Launch Bounds** | `__launch_bounds__(256, 2)` | 1.2-2x |
| **Register Reduce** | 复用变量 | 1.1-1.3x |
| **Block Size Tuning** | 调整 threads per block | 1.1-1.5x |

---

## 📊 输出模板

```markdown
# NCU 性能分析报告 (v2)

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

### Speed Of Light Throughput
| 指标 | 数值 | 健康阈值 | 状态 |
|------|------|----------|------|
| Memory Throughput | {memory_throughput}% | < 80% | {status} |
| DRAM Throughput | {dram_throughput}% | < 80% | {status} |
| Compute (SM) Throughput | {sm_throughput}% | < 80% | {status} |
| L1/TEX Throughput | {l1tex_throughput}% | < 80% | {status} |

### Compute Workload
| 指标 | 数值 | 健康阈值 | 状态 |
|------|------|----------|------|
| SM Busy | {sm_busy}% | > 70% | {status} |
| Issue Slots Busy | {issue_slots_busy}% | > 50% | {status} |
| Executed Ipc Active | {ipc_active} | > 0.5 | {status} |

### Memory Workload
| 指标 | 数值 | 健康阈值 | 状态 |
|------|------|----------|------|
| L1/TEX Hit Rate | {l1_hit_rate}% | > 50% | {status} |
| L2 Hit Rate | {l2_hit_rate}% | > 70% | {status} |

### Occupancy
| 指标 | 数值 | 健康阈值 | 状态 |
|------|------|----------|------|
| Theoretical Occupancy | {theoretical_occupancy}% | > 50% | {status} |
| Achieved Occupancy | {achieved_occupancy}% | > 40% | {status} |

## 🔍 诊断详情

**瓶颈类型**: {bottleneck_type}

**判断依据**:
- {reason_1}
- {reason_2}
- {reason_3}

## 💡 优化建议

### 高优先级
{high_priority_suggestions}

### 中优先级
{medium_priority_suggestions}

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
- [ ] 验证结果正确性
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

```bash
# Python 分析器
python optimizer.py --import report_name.ncu-rep

# 分析模式 (仅分析不优化)
python optimizer.py matmul.cu --mode=analyze

# 全自动优化
python optimizer.py matmul.cu --mode=auto --build "nvcc -O3 {source} -o {output}"
```

---

## 📖 诊断规则详解

### DRAM_MEMORY_BOUND

```
IF dram_throughput > 80% AND sm_throughput < 50%:
    诊断: DRAM_MEMORY_BOUND (置信度: HIGH)
    
    优化策略:
    1. Block Tiling (共享内存缓存)
    2. Vectorized Load (float4)
    3. Prefetching (数据预取)
```

### L1_PRESSURE_BOUND

```
IF l1tex_throughput > 80% AND dram_throughput < 30% AND l1_hit_rate < 30%:
    诊断: L1_PRESSURE_BOUND (置信度: HIGH)
    
    优化策略:
    1. Shared Memory Padding
    2. Data Transpose
    3. Fragment Caching
```

### LATENCY_BOUND

```
IF sm_busy < 30% AND occupancy > 50% AND memory_dependency_stall > 30%:
    诊断: LATENCY_BOUND (置信度: MEDIUM)
    
    优化策略:
    1. Double Buffering
    2. Loop Unrolling
    3. ILP Increase
```

---

## 🎯 优化策略速查

| 瓶颈类型 | 立即行动 | 代码示例 | 预期收益 |
|---------|---------|---------|---------|
| **DRAM_MEMORY_BOUND** | Block Tiling | `__shared__ float As[BM][BK];` | 3-5x |
| **L1_PRESSURE_BOUND** | Padding | `As[BM][BK+1]` | 1.2-2x |
| **LATENCY_BOUND** | Double Buffer | `As[2][BM*BK]` | 1.2-1.5x |
| **COMPUTE_BOUND** | FMA | `fmaf(a, b, c)` | 1.1-1.3x |
| **OCCUPANCY_BOUND** | Launch Bounds | `__launch_bounds__(256, 2)` | 1.2-2x |

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

4. **NCU 采集时间 ≠ 真实时间**
   - ncu 会多次重放 kernel，采集时间会大幅膨胀
   - 测真实性能用 cudaEvent / nsys

5. **不要过度优化 Stall Reasons**
   - 只有当调度器无法每周期发射时才关注 stall
   - Issue Slot 利用率已高时，stall 可能是正常调度行为

---

## 🔗 相关资源

- 自动化脚本: `optimizer.py`, `strategy_library.py`
- 示例报告: 见项目 `examples/` 目录
- NVIDIA 官方文档: https://docs.nvidia.com/nsight-compute/

---

*本 Skill 支持完整的自动化 NCU 性能分析工作流，包含全量采集、智能诊断和优化建议*
