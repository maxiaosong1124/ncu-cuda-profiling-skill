# 🚀 NCU CUDA Profiling Skill

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-Profiling-green?style=flat-square&logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/NCU-Nsight%20Compute-blue?style=flat-square&logo=nvidia" alt="NCU">
  <img src="https://img.shields.io/badge/AI-Agent-ff69b4?style=flat-square" alt="AI Agent">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License">
</p>

专为 **AI Agent** 设计的 NCU (Nsight Compute) 自动化性能分析 Skill。

安装后，你的 AI 助手（Kimi、Claude、Cursor 等）将自动掌握 CUDA 性能分析能力，一键诊断瓶颈并给出优化建议。

---

## ✨ 核心能力

安装本 Skill 后，你的 AI 助手将能够：

```
你: 帮我分析这个 matmul kernel 的性能

AI: [自动识别 ncu-cuda-profiling skill]
    
    我来帮你分析这个矩阵乘法 kernel：
    
    1️⃣ 首先运行 NCU 采集性能数据...
    $ ncu --set full -o report ./matmul
    
    2️⃣ 诊断结果：
    ┌─────────────────────────────────────┐
    │ 主要瓶颈: DRAM_MEMORY_BOUND         │
    │ L1 Hit Rate: 1.52% 🔴 (极低)        │
    │ 优化潜力: 4.2x                      │
    └─────────────────────────────────────┘
    
    3️⃣ 优化建议：
    • 添加 Shared Memory Padding 解决 bank conflict
    • 实现双缓冲隐藏延迟
    
    预期性能提升: 3-5x 🚀
```

---

## 🤖 支持的 AI 工具

| AI 工具 | 安装命令 | 自动识别 | 状态 |
|---------|----------|----------|------|
| **Kimi Code CLI** | `./install.sh --kimi` | ✅ 启动即加载 | ⭐ 推荐 |
| **Claude Code** | `./install.sh --claude` | ✅ 自动识别 | ✅ 支持 |
| **Cursor** | `./install.sh --cursor` | ✅ Rules 自动应用 | ✅ 支持 |
| **Codex** | `./install.sh --codex` | ✅ 自动识别 | ✅ 支持 |

---

## 🚀 安装（30 秒）

### 方式一：安装到 Kimi Code CLI（推荐）

```bash
git clone https://github.com/maxiaosong1124/ncu-cuda-profiling-skill.git
cd ncu-cuda-profiling-skill
./install.sh

# 完成！启动 Kimi Code CLI 即可使用
kimi
```

### 方式二：安装到所有 AI 工具

```bash
./install.sh --all-agents
```

### 方式三：安装到特定工具

```bash
./install.sh --kimi      # Kimi Code CLI
./install.sh --claude    # Claude Code
./install.sh --cursor    # Cursor
./install.sh --codex     # Codex
```

---

## 💡 使用示例

安装完成后，直接在你的 AI 工具中询问：

### 示例 1：性能分析

```
你: 分析这个 CUDA kernel 为什么慢

AI: 正在使用 NCU 分析...
    
    📊 诊断报告：
    • 瓶颈类型: L1_PRESSURE_BOUND
    • SM 利用率: 30% (偏低)
    • L1 命中率: 2% 🔴 (bank conflict 严重)
    
    🔧 建议修改:
    ```cpp
    // 修改前
    __shared__ float Bs[BK * BN];
    
    // 修改后 (+1 padding)
    __shared__ float Bs[BK * (BN + 1)];
    ```
```

### 示例 2：优化建议

```
你: 怎么优化这个矩阵乘法？

AI: 基于 NCU 分析，我建议：
    
    1. Block Tiling (预期 3x 提升)
    2. Shared Memory Padding (预期 1.5x 提升)
    3. 向量化加载 float4 (预期 1.3x 提升)
    
    总预期提升: 5-6x 🚀
```

### 示例 3：对比分析

```
你: 对比优化前后的性能

AI: 正在分析两个版本...
    
    📈 性能对比:
    ┌─────────────┬──────────┬─────────┬────────┐
    │ 指标        │ 优化前   │ 优化后  │ 提升   │
    ├─────────────┼──────────┼─────────┼────────┤
    │ 执行时间    │ 1200μs   │ 340μs   │ 3.5x   │
    │ L1 命中率   │ 2%       │ 78%     │ 39x    │
    │ DRAM 带宽   │ 85%      │ 25%     │ 🟢     │
    └─────────────┴──────────┴─────────┴────────┘
```

---

## 📖 系统要求

| 项目 | 要求 |
|------|------|
| **操作系统** | Linux (Ubuntu 18.04+) |
| **CUDA** | 11.0+ |
| **NCU** | Nsight Compute 2022.1+ |
| **GPU** | NVIDIA Volta 或更新架构 |

### 检查环境

```bash
./check_env.sh
```

---

## 🔧 手动使用（备选方案）

如果 AI Agent 集成遇到问题，可以手动使用：

### 基础用法

```bash
# 1. 运行 NCU 采集
ncu --set full -o my_report --target-processes all ./your_kernel

# 2. 查看结果
ncu --import my_report.ncu-rep --print-summary per-kernel
```

### 使用提供的脚本

```bash
# 自动化分析
cd examples
./auto_profile.sh ../your_kernel my_analysis

# Python 深度分析
python ncu_analyzer.py --import my_report.ncu-rep
```

### CLI 工具（如果添加到 PATH）

```bash
ncu-profile ./your_kernel      # 一键分析
ncu-analyze my_report.ncu-rep  # 分析已有报告
```

---

## 📊 诊断能力

本 Skill 支持自动识别 5 种瓶颈类型：

| 瓶颈类型 | 识别条件 | 优化策略 | 预期收益 |
|---------|---------|---------|---------|
| **DRAM_MEMORY_BOUND** | DRAM > 70% | Block Tiling, Vectorized Load | 3-5x |
| **L1_PRESSURE_BOUND** | L1/TEX > 80% | Padding, Transpose | 1.2-2x |
| **LATENCY_BOUND** | SM < 50%, Occupancy > 60% | Double Buffering | 1.2-1.5x |
| **COMPUTE_BOUND** | Roofline > 60%, SM > 80% | FMA, Tensor Cores | 1.1-1.3x |
| **OCCUPANCY_BOUND** | Occupancy < 30% | 调整 block size | 1.2-2x |

---

## 📁 项目结构

```
ncu-cuda-profiling-skill/
├── README.md                    # 本文件
├── SKILL.md                     # AI Agent 核心知识库
├── AGENTS_COMPATIBILITY.md      # 多 Agent 兼容性文档
├── LICENSE                      # MIT 许可证
├── install.sh                   # ⭐ 一键安装脚本
├── check_env.sh                 # 环境检查
├── examples/                    # 示例和工具
│   ├── auto_profile.sh          # 自动化脚本
│   └── ncu_analyzer.py          # Python 分析器
└── .github/workflows/           # CI 配置
```

---

## 🤝 贡献

欢迎提交 Issue 和 PR！

```bash
git clone https://github.com/maxiaosong1124/ncu-cuda-profiling-skill.git
cd ncu-cuda-profiling-skill
# 修改后提交 PR
```

---

## 📄 许可证

[MIT License](LICENSE) - 自由使用和修改

---

<p align="center">
  如果这个项目对你有帮助，请 ⭐ Star 支持！
  <br>
  <a href="https://github.com/maxiaosong1124/ncu-cuda-profiling-skill">GitHub</a>
</p>
