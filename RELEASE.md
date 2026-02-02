# 🚀 发布指南

本文档指导如何将 NCU CUDA Profiling Skill 发布到 GitHub。

## 📋 发布前检查清单

### 1. 文件完整性检查

```bash
# 确保所有文件已创建
ls -la ncu-cuda-profiling-skill/
# 应有:
# - README.md
# - SKILL.md
# - LICENSE
# - install.sh
# - check_env.sh
# - examples/
# - .github/
# - .gitignore
```

### 2. 内容审核

- [ ] README.md 中的 `maxiaosong1124` 替换为实际 GitHub 用户名
- [ ] LICENSE 中的 `[Your Name]` 替换为实际姓名
- [ ] SKILL.md 中的 GitHub 链接更新
- [ ] 版本号确认 (SKILL.md 头部)

### 3. 脚本权限

```bash
chmod +x ncu-cuda-profiling-skill/install.sh
chmod +x ncu-cuda-profiling-skill/check_env.sh
chmod +x ncu-cuda-profiling-skill/examples/auto_profile.sh
```

---

## 🚀 GitHub 发布步骤

### 方法一：命令行发布（推荐）

```bash
# 1. 进入项目目录
cd ncu-cuda-profiling-skill

# 2. 初始化 git 仓库
git init

# 3. 添加所有文件
git add .

# 4. 提交
git commit -m "Initial commit: NCU CUDA Profiling Skill v1.0.0"

# 5. 添加远程仓库（替换 maxiaosong1124）
git remote add origin https://github.com/maxiaosong1124/ncu-cuda-profiling-skill.git

# 6. 推送到 GitHub
git push -u origin main
# 或如果默认分支是 master:
# git push -u origin master
```

### 方法二：GitHub Web 界面

1. 登录 GitHub
2. 点击右上角 `+` → `New repository`
3. 填写信息:
   - **Repository name**: `ncu-cuda-profiling-skill`
   - **Description**: `Automated NCU (Nsight Compute) profiling workflow for CUDA optimization`
   - **Visibility**: Public (或 Private)
   - **Initialize**: 不勾选（因为我们已有文件）
4. 创建后按页面提示推送现有仓库

---

## 🏷️ 创建 Release

### 1. 标签版本

```bash
# 创建标签
git tag -a v1.0.0 -m "Release v1.0.0: Initial release with full profiling workflow"

# 推送标签
git push origin v1.0.0
```

### 2. GitHub Release

1. 进入仓库 → Releases → `Create a new release`
2. 选择标签 `v1.0.0`
3. 填写发布信息:

```markdown
## NCU CUDA Profiling Skill v1.0.0

🚀 首个正式版本发布！

### 特性
- ✅ 一键完整 NCU 采集
- ✅ 智能瓶颈诊断 (DRAM/LATENCY/COMPUTE/OCCUPANCY)
- ✅ 自动生成 Markdown + CSV 报告
- ✅ 详细的优化建议
- ✅ 支持 AI Agent 集成

### 安装
```bash
git clone https://github.com/maxiaosong1124/ncu-cuda-profiling-skill.git
cd ncu-cuda-profiling-skill
./install.sh
```

### 快速开始
```bash
ncu-profile ./your_cuda_kernel
```

### 文档
- [详细文档](SKILL.md)
- [示例教程](examples/README.md)
```

4. 发布！

---

## 🌍 推广分享

发布后可以在以下平台分享：

### 中文社区
- [V2EX](https://www.v2ex.com/) - CUDA/GPU 节点
- [知乎](https://www.zhihu.com/)
- [稀土掘金](https://juejin.cn/)
- [CSDN](https://www.csdn.net/)

### 国际社区
- [Reddit r/CUDA](https://www.reddit.com/r/CUDA/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
- [Hacker News](https://news.ycombinator.com/)
- Twitter / X

### 分享模板

```markdown
🚀 开源发布: NCU CUDA Profiling Skill

一个自动化 NCU (Nsight Compute) 性能分析工具，帮助开发者快速定位和优化 CUDA Kernel 性能瓶颈。

✨ 特性:
• 一键采集全量指标
• 智能诊断瓶颈类型 (Memory/Latency/Compute)
• 自动生成分析报告
• 详细的优化建议

📦 GitHub: https://github.com/maxiaosong1124/ncu-cuda-profiling-skill

#CUDA #GPU #Profiling #HPC
```

---

## 🔧 持续维护

### 版本规划

| 版本 | 计划 | 时间 |
|------|------|------|
| v1.1.0 | 添加更多诊断规则 | TBD |
| v1.2.0 | Web UI 可视化 | TBD |
| v2.0.0 | 支持多 GPU 分析 | TBD |

### Issue 模板

创建 `.github/ISSUE_TEMPLATE/`:

```bash
mkdir -p .github/ISSUE_TEMPLATE
```

**bug_report.md**:
```markdown
---
name: Bug report
about: 报告问题
title: '[BUG] '
labels: bug
---

**描述问题**

**复现步骤**
1. 
2. 
3. 

**期望行为**

**环境信息**
- OS: 
- CUDA: 
- NCU: 
- GPU: 
```

**feature_request.md**:
```markdown
---
name: Feature request
about: 功能建议
title: '[FEATURE] '
labels: enhancement
---

**功能描述**

**使用场景**

**期望实现**
```

---

## 📊 成功指标

发布后关注以下指标：

- ⭐ Star 数量
- 🍴 Fork 数量
- 📥 Clone 次数
- 🐛 Issue 活跃度
- 🔀 PR 贡献数

祝发布顺利！🎉
