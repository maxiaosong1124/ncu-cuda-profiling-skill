# 我开发了一个 AI 插件，让 Claude/Kimi 自动帮你优化 CUDA 程序

> 作为一名 CUDA 开发者，你是否遇到过这样的场景：写了一个 kernel，不知道性能瓶颈在哪，只能对着 NCU 报告发呆？现在，只需安装这个 Skill，你的 AI 助手就能自动分析性能并给出优化建议。

---

## 📖 背景故事

上个月我在优化一个矩阵乘法 kernel 时，被 NCU（Nsight Compute）的复杂报告搞得头大。虽然 NCU 能提供详细的性能指标，但要从上百个指标中找出瓶颈、再给出针对性的优化建议，真的需要丰富的经验。

于是我想：**能不能让 AI 来帮我读 NCU 报告？**

就这样，**NCU CUDA Profiling Skill** 诞生了。

---

## 🤔 为什么要用这个 Skill？

### 传统方式的痛点

1. **NCU 报告太复杂**
   - 几十个指标，不知道哪个是关键
   - 需要经验才能判断瓶颈类型
   
2. **优化建议不直观**
   - 知道有 bank conflict，但不知道具体怎么改
   - 缺乏针对性的代码示例

3. **学习成本高**
   - 新手很难快速掌握性能分析
   - 每个 kernel 都要手动分析一遍

### 使用 Skill 的优势

| 传统方式 | 使用 Skill |
|---------|-----------|
| 手动运行 NCU | AI 自动帮你运行 |
| 自己读报告 | AI 直接告诉你瓶颈在哪 |
| 查文档找优化方法 | AI 给出具体代码修改建议 |
| 每次都要重复 | 一次安装，永久使用 |

**一句话总结**：安装后，你只需要对 AI 说"帮我分析这个 kernel"，剩下的它全包了。

---

## 🚀 使用过程演示

### 第一步：安装 Skill（30 秒）

```bash
git clone https://github.com/maxiaosong1124/ncu-cuda-profiling-skill.git
cd ncu-cuda-profiling-skill
./install.sh
```

> **[截图提示 1]**：这里放一张终端执行安装命令的截图，显示安装成功的提示

### 第二步：启动你的 AI 工具

我使用的是 Kimi Code CLI，安装完成后直接启动：

```bash
kimi
```

### 第三步：直接对话使用

现在，你只需要像平时一样和 AI 对话：

---

**我**：帮我分析这个 matmul kernel 的性能

**AI**：

> 我来帮你使用 NCU 分析这个矩阵乘法 kernel...
>
> **第一步**：运行 NCU 采集性能数据
> ```bash
> ncu --set full -o report ./matmul
> ```
> *[AI 自动执行 NCU 采集...]*
>
> **第二步**：诊断结果
> ```
> ┌─────────────────────────────────────┐
> │ 主要瓶颈: DRAM_MEMORY_BOUND         │
> │ L1 Hit Rate: 1.52% 🔴 (极低)        │
> │ SM 利用率: 30% 🟡 (偏低)            │
> │ 优化潜力: 4.2x                      │
> └─────────────────────────────────────┘
> ```
>
> **第三步**：根因分析
> - 共享内存存在 bank conflict，导致 L1 命中率极低
> - 没有使用 double buffering，无法隐藏延迟
>
> **第四步**：优化建议
> ```cpp
> // 修改 1: 添加 Padding 解决 bank conflict
> __shared__ float As[BM * (BK + 1)];  // +1 padding
> __shared__ float Bs[BK * (BN + 1)];  // +1 padding
> 
> // 修改 2: 双缓冲隐藏延迟
> __shared__ float As[2][BM * BK];
> ```
>
> **预期性能提升**: 3-5x 🚀

> **[截图提示 2]**：这里放一张 AI 输出完整分析报告的截图，突出显示诊断结果和优化建议部分

---

**我**：具体应该怎么改代码？

**AI**：

> 基于 NCU 分析，我建议按以下顺序优化：
>
> **1. 解决 Bank Conflict（预期 2x 提升）**
> ```cpp
> // 修改前
> __shared__ float As[BM * BK];
> __shared__ float Bs[BK * BN];
> 
> // 修改后
> __shared__ float As[BM][BK + 1];  // 二维数组 + padding
> __shared__ float Bs[BK][BN + 1];
> ```
>
> **2. 双缓冲优化（预期 1.5x 提升）**
> ```cpp
> __shared__ float As[2][BM * BK];
> __shared__ float Bs[2][BK * BN];
> 
> // 在计算当前 tile 时预取下一个 tile
> ```
>
> **3. 向量化加载（预期 1.3x 提升）**
> ```cpp
> // 使用 float4 加载
> float4 data = reinterpret_cast<float4*>(A)[idx];
> ```

> **[截图提示 3]**：这里放一张 AI 给出具体代码修改建议的截图，展示代码对比

---

**我**：改完之后效果怎么样？

**AI**：

> 让我对比一下优化前后的性能：
>
> ```
> 📈 性能对比报告
> ┌─────────────┬──────────┬─────────┬────────┐
> │ 指标        │ 优化前   │ 优化后  │ 提升   │
> ├─────────────┼──────────┼─────────┼────────┤
> │ 执行时间    │ 1200μs   │ 340μs   │ 3.5x   │
> │ L1 命中率   │ 2%       │ 78%     │ 39x    │
> │ SM 利用率   │ 30%      │ 71%     │ 2.4x   │
> │ DRAM 带宽   │ 85%      │ 25%     │ 🟢优化 │
> └─────────────┴──────────┴─────────┴────────┘
> ```
>
> **总结**：通过添加 padding 和双缓冲，性能提升了 **3.5 倍**，L1 命中率从 2% 提升到 78%。

> **[截图提示 4]**：这里放一张 AI 展示性能对比表格的截图，突出提升倍数

---

## 🛠️ 支持的 AI 编程工具

这个 Skill 目前支持以下主流 AI 编程助手：

| AI 工具 | 安装命令 | 特点 |
|---------|----------|------|
| **Kimi Code CLI** | `./install.sh --kimi` | ⭐ 推荐！启动即自动加载 |
| **Claude Code** | `./install.sh --claude` | 强大的代码理解能力 |
| **Cursor** | `./install.sh --cursor` | 编辑器内无缝集成 |
| **Codex** | `./install.sh --codex` | OpenAI 官方工具 |

### 一键安装到所有工具

```bash
./install.sh --all-agents
```

> **[截图提示 5]**：这里放一张展示安装命令和成功提示的截图

---

## 📋 系统要求

- **操作系统**：Linux (Ubuntu 18.04+)
- **CUDA**：11.0+
- **NCU**：Nsight Compute 2022.1+
- **GPU**：NVIDIA Volta 或更新架构

---

## 🔗 项目地址

**GitHub**: https://github.com/maxiaosong1124/ncu-cuda-profiling-skill

欢迎 Star ⭐ 和 Fork！有问题可以在 GitHub Issues 提出。

---

## 💡 总结

这个 Skill 的核心价值是**降低 CUDA 性能分析的门槛**：

- 新手不再需要学习复杂的 NCU 指标
- 专家可以快速获得优化建议
- 一次安装，所有项目都能用

如果你也经常写 CUDA 程序，强烈推荐试试！

---

## 📣 互动时间

你在 CUDA 优化过程中遇到过哪些坑？欢迎在评论区分享，我会用 Skill 帮你分析！

---

*本文使用 NCU CUDA Profiling Skill 辅助创作*
