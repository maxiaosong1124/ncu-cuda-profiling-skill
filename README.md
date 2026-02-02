# 🚀 NCU CUDA Profiling Skill

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-Profiling-green?style=flat-square&logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/NCU-Nsight%20Compute-blue?style=flat-square&logo=nvidia" alt="NCU">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Platform-Linux-orange?style=flat-square" alt="Platform">
</p>

自动化 NCU (Nsight Compute) 性能分析工作流，支持**全量指标采集**、**智能诊断**和**一键生成分析报告**。

## ✨ 特性

- 🔥 **一键完整采集** - `ncu --set full` 自动化运行
- 🧠 **智能瓶颈诊断** - 自动识别 Memory/Compute/Occupancy 瓶颈
- 📊 **自动生成报告** - Markdown + CSV 双格式输出
- 🎯 **优化建议** - 针对具体问题提供优化策略
- 📈 **性能对比** - 支持多版本 kernel 对比分析

## 🚀 快速开始

### 方式一：一键安装（推荐）

```bash
# 克隆仓库
git clone https://github.com/maxiaosong1124/ncu-cuda-profiling-skill.git
cd ncu-cuda-profiling-skill

# 一键安装到系统
./install.sh

# 或者安装到指定目录（适用于 Kimi Code CLI 等 Agent 环境）
./install.sh --target ~/.config/agents/skills/
```

### 方式二：手动安装

```bash
# 1. 克隆仓库
git clone https://github.com/maxiaosong1124/ncu-cuda-profiling-skill.git
cd ncu-cuda-profiling-skill

# 2. 复制到 skill 目录
cp -r ncu-cuda-profiling ~/.config/agents/skills/
# 或
cp -r ncu-cuda-profiling /path/to/your/skills/
```

### 方式三：Docker 使用

```bash
# 构建镜像
docker build -t ncu-skill .

# 运行分析
docker run --gpus all -v $(pwd):/workspace ncu-skill ./your_kernel
```

## 📖 使用方法

### 基础用法

```bash
# 进入你的 CUDA 项目目录
cd your_cuda_project

# 运行完整分析
ncu-profile ./matmul

# 或直接使用完整命令
ncu --set full -o report --target-processes all ./matmul
```

### 自动化脚本

```bash
# 使用提供的自动化脚本
cd examples

# 基础分析
./auto_profile.sh ../your_cuda_project/matmul my_report

# Python 深度分析
python ncu_analyzer.py --import my_report.ncu-rep
```

### 分析已有报告

```bash
# 从已有 .ncu-rep 生成分析报告
ncu-analyze my_report.ncu-rep

# 导出为 CSV
ncu --import my_report.ncu-rep --page raw --csv > metrics.csv
```

## 📊 输出示例

```markdown
# NCU 性能分析报告

## 📈 执行摘要
| 项目 | 数值 |
|------|------|
| **主要瓶颈** | DRAM_MEMORY_BOUND |
| **性能** | 156.7 GFLOPS |
| **优化潜力** | 4.2x |

## 📊 关键指标
| 指标 | 数值 | 状态 |
|------|------|------|
| SM Busy | 71.05% | 🟢 正常 |
| DRAM Throughput | 55.35% | 🟡 偏高 |
| L1 Hit Rate | 3.08% | 🔴 差 |

## 💡 优化建议
1. **Block Tiling** - 使用共享内存缓存数据
2. **Vectorized Load** - 使用 float4 加载
3. **Shared Memory Padding** - 避免 bank conflict
```

## 🔧 系统要求

| 项目 | 要求 |
|------|------|
| **操作系统** | Linux (Ubuntu 18.04+) |
| **CUDA** | 11.0+ |
| **NCU** | Nsight Compute 2022.1+ |
| **GPU** | NVIDIA Volta 或更新架构 |
| **Python** | 3.7+ (用于高级分析) |

### 检查环境

```bash
# 检查 NCU 是否安装
ncu --version

# 检查 GPU
nvidia-smi

# 运行环境检查脚本
./check_env.sh
```

## 📁 项目结构

```
ncu-cuda-profiling-skill/
├── README.md                 # 本文件
├── install.sh                # 一键安装脚本
├── check_env.sh              # 环境检查脚本
├── SKILL.md                  # Skill 核心文档
├── LICENSE                   # MIT 许可证
├── examples/                 # 示例和工具
│   ├── README.md            # 示例说明
│   ├── auto_profile.sh      # 自动化分析脚本
│   └── ncu_analyzer.py      # Python 分析器
└── .github/                 # GitHub 配置
    └── workflows/           # CI/CD 工作流
```

## 🎯 使用场景

### 场景一：优化 CUDA Kernel

```bash
# 1. 采集性能数据
ncu --set full -o before ./matmul_before

# 2. 实施优化（如添加 shared memory tiling）
# ... 修改代码 ...

# 3. 重新采集
ncu --set full -o after ./matmul_after

# 4. 对比分析
ncu --diff before.ncu-rep after.ncu-rep
```

### 场景二：集成到 CI/CD

```yaml
# .github/workflows/ncu.yml
name: Performance Check
on: [push]
jobs:
  ncu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup NCU Skill
        run: |
          git clone https://github.com/maxiaosong1124/ncu-cuda-profiling-skill.git
          ./ncu-cuda-profiling-skill/install.sh
      - name: Run Profiling
        run: ncu-profile ./my_kernel
```

### 场景三：AI Agent 集成

对于 Kimi Code CLI 等 AI Agent，安装后可直接使用：

```bash
# Agent 会自动识别 skill
@ncu-profile ./matmul

# Agent 会返回结构化分析结果
"""
主要瓶颈: DRAM_MEMORY_BOUND
优化建议:
1. 使用 Block Tiling 减少全局内存访问
2. 添加 Shared Memory Padding 避免 bank conflict
预期收益: 3-5x 性能提升
"""
```

## 📚 文档

- [详细使用指南](SKILL.md) - 完整的诊断规则和优化策略
- [示例教程](examples/README.md) - 实际案例分析
- [FAQ](docs/FAQ.md) - 常见问题解答

## 🤝 贡献

欢迎提交 Issue 和 PR！

```bash
# 开发流程
git clone https://github.com/maxiaosong1124/ncu-cuda-profiling-skill.git
cd ncu-cuda-profiling-skill

# 创建分支
git checkout -b feature/your-feature

# 提交更改
git commit -am "Add your feature"
git push origin feature/your-feature
```

## 📄 许可证

[MIT License](LICENSE) - 自由使用和修改

## 🙏 致谢

- NVIDIA Nsight Compute 团队
- CUDA 社区

---

<p align="center">
  如果这个项目对你有帮助，请 ⭐ Star 支持！
</p>
