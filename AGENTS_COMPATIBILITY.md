# 🤖 AI Agent 兼容性指南

本文档说明如何在各种 AI 编程工具中使用 NCU CUDA Profiling Skill。

---

## ✅ 已支持的工具

### 1. Kimi Code CLI ⭐ 原生支持

**识别方式**: 自动识别 `~/.config/agents/skills/` 目录下的 `SKILL.md`

**安装**:
```bash
./install.sh --target ~/.config/agents/skills/ncu-cuda-profiling
# 或简写
./install.sh  # 默认安装到 ~/.config/agents/skills/ncu-cuda-profiling
```

**使用**:
```bash
# 启动 Kimi Code CLI 后，skill 会自动加载
# 你可以通过以下方式调用：
@ncu-profile ./matmul
# 或询问：
"帮我分析这个 CUDA kernel 的性能"
```

**原理**: 
- Kimi Code CLI 启动时会扫描 `~/.config/agents/skills/*/` 目录
- 读取每个目录下的 `SKILL.md` 文件
- 将内容注入到系统 prompt 中

---

### 2. Claude Code ✨ 手动配置支持

**识别方式**: 通过 `.claude/skills/` 目录或自定义配置

**安装**:
```bash
# 方式 1: 项目级安装（推荐）
mkdir -p ~/.claude/skills/ncu-cuda-profiling
cp SKILL.md ~/.claude/skills/ncu-cuda-profiling/
cp -r examples ~/.claude/skills/ncu-cuda-profiling/

# 方式 2: 使用 install.sh
./install.sh --target ~/.claude/skills/ncu-cuda-profiling
```

**使用**:
```bash
# 在 Claude Code 中直接询问
"使用 ncu-cuda-profiling skill 分析这个 kernel"
```

**注意**: Claude Code 的 skill 系统仍在发展中，建议同时使用系统 prompt 方式。

---

### 3. Codex (OpenAI) 🔄 通过配置支持

**识别方式**: 通过 `.codex/` 配置目录

**安装**:
```bash
mkdir -p ~/.codex/skills/ncu-cuda-profiling
cp SKILL.md ~/.codex/skills/ncu-cuda-profiling/
```

**使用**:
Codex 会在处理 CUDA 相关问题时自动引用 skill 内容。

---

### 4. Cursor 📝 通过 Rules 支持

**识别方式**: 通过 `.cursor/rules/` 或 `.cursorrules` 文件

**安装**:
```bash
# 项目级安装
mkdir -p .cursor/rules
cp SKILL.md .cursor/rules/ncu-cuda-profiling.md

# 或全局安装
mkdir -p ~/.cursor/rules
cp SKILL.md ~/.cursor/rules/ncu-cuda-profiling.md
```

**使用**:
Cursor 会自动读取 rules 目录下的 markdown 文件作为上下文。

---

### 5. GitHub Copilot 🔧 通过 Prompt 支持

**识别方式**: 通过 VS Code 的 Copilot 自定义指令

**安装**:
```bash
# 复制 skill 内容到 VS Code 设置
# VS Code → Settings → Copilot → Custom Instructions
```

**使用**:
在代码注释中使用特定标记触发。

---

## 🚀 通用安装方案（推荐）

为了最大化兼容性，我们提供 **一键全平台安装**：

```bash
# 安装到所有支持的 agent 目录
./install.sh --all-agents

# 或分别安装
./install.sh --kimi      # Kimi Code CLI
./install.sh --claude    # Claude Code
./install.sh --cursor    # Cursor
./install.sh --codex     # Codex
```

---

## 📋 各工具配置细节

### Kimi Code CLI

**配置路径**: `~/.config/agents/skills/ncu-cuda-profiling/SKILL.md`

**验证安装**:
```bash
ls ~/.config/agents/skills/ncu-cuda-profiling/SKILL.md
# 输出: .../SKILL.md
```

**使用示例**:
```
用户: 分析这个 matmul kernel 的性能
Kimi: [自动加载 ncu-cuda-profiling skill]
      我来帮你使用 NCU 分析这个 CUDA kernel...
```

---

### Claude Code

**配置路径**: 
- 项目级: `.claude/skills/ncu-cuda-profiling/SKILL.md`
- 用户级: `~/.claude/skills/ncu-cuda-profiling/SKILL.md`

**手动加载**:
在 Claude Code 中，你可以通过以下方式显式引用 skill：
```
/claude load-skill ncu-cuda-profiling
```

**环境变量方式**:
```bash
export CLAUDE_SKILLS_PATH="~/.claude/skills"
```

---

### Cursor

**配置路径**:
- 项目级: `.cursorrules` 或 `.cursor/rules/`
- 用户级: `~/.cursor/rules/`

**创建 `.cursorrules` 文件**:
```bash
cat > .cursorrules << 'EOF'
# NCU CUDA Profiling Skill

当用户询问 CUDA 性能优化时：
1. 使用 ncu --set full 采集性能数据
2. 分析 DRAM/L1/SM 利用率
3. 识别瓶颈类型
4. 提供具体优化建议

## 诊断规则
...
EOF
```

---

## 🎯 使用场景示例

### 场景 1: Kimi Code CLI 自动识别

```bash
# 用户安装 skill
./install.sh

# 启动 Kimi Code CLI
kimi

# 在对话中使用
用户: 帮我优化这个 matmul.cu
Kimi: [自动识别 ncu-cuda-profiling skill]
      好的，我来帮你分析这个矩阵乘法 kernel。
      
      首先运行 NCU 采集：
      ```bash
      ncu --set full -o report ./matmul
      ```
      
      [分析结果...]
      [优化建议...]
```

### 场景 2: Claude Code 显式调用

```bash
# 安装 skill
./install.sh --claude

# 启动 Claude Code
claude

# 显式引用 skill
用户: /skill ncu-cuda-profiling
Claude: 已加载 NCU CUDA Profiling Skill

用户: 分析这个 kernel
Claude: [使用 skill 知识分析...]
```

### 场景 3: Cursor Rules 自动应用

```bash
# 在项目根目录创建 .cursorrules
cp SKILL.md .cursorrules

# 打开 Cursor，开始编辑 CUDA 文件
# Cursor 会自动应用 rules 中的知识
```

---

## ⚠️ 已知限制

| 工具 | 支持状态 | 限制 |
|------|----------|------|
| Kimi Code CLI | ✅ 完全支持 | 需正确放置到 skills 目录 |
| Claude Code | 🟡 部分支持 | 需手动配置或使用系统 prompt |
| Cursor | 🟡 部分支持 | 通过 rules 机制，非原生 skill |
| Codex | 🟡 部分支持 | 需自定义配置 |
| GitHub Copilot | 🔴 不支持 | 无 skill 机制，只能用自定义指令 |

---

## 🔮 未来计划

- [ ] 原生 Claude Code Skill 支持
- [ ] VS Code 扩展
- [ ] 独立的 CLI 工具
- [ ] Web UI 界面

---

## 💡 最佳实践

1. **多工具共存**: 同时安装到多个 agent 的 skills 目录
   ```bash
   ./install.sh --kimi --claude --cursor
   ```

2. **项目级配置**: 将 skill 放入项目目录，便于团队协作
   ```bash
   mkdir -p .skills/ncu-cuda-profiling
   cp SKILL.md .skills/ncu-cuda-profiling/
   ```

3. **版本管理**: 使用 git submodule 管理 skill 版本
   ```bash
   git submodule add https://github.com/maxiaosong1124/ncu-cuda-profiling-skill.git .skills/ncu-cuda-profiling
   ```

---

## 📚 参考

- [Kimi Code CLI Skills 文档](https://github.com/yourusername/kimi-cli-skills)
- [Claude Code Documentation](https://docs.anthropic.com/claude/docs)
- [Cursor Rules Documentation](https://cursor.sh/docs/rules)
