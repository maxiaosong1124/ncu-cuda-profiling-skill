#!/bin/bash
#
# NCU CUDA Profiling Skill - 一键发布脚本
# 使用方式: ./publish.sh [你的GitHub用户名]
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

USERNAME=${1:-""}

if [ -z "$USERNAME" ]; then
    echo -e "${RED}错误: 请提供 GitHub 用户名${NC}"
    echo "用法: ./publish.sh your_github_username"
    echo "示例: ./publish.sh hellofss"
    exit 1
fi

echo "========================================"
echo "  🚀 NCU CUDA Profiling Skill 发布"
echo "========================================"
echo ""
echo -e "GitHub 用户名: ${BLUE}$USERNAME${NC}"
echo ""

# 1. 替换用户名
echo -e "${YELLOW}[1/6]${NC} 替换配置中的用户名..."
sed -i "s/yourusername/$USERNAME/g" README.md SKILL.md RELEASE.md 2>/dev/null || true
echo -e "${GREEN}✓${NC} 完成"

# 2. 设置权限
echo -e "${YELLOW}[2/6]${NC} 设置脚本权限..."
chmod +x install.sh check_env.sh examples/auto_profile.sh
echo -e "${GREEN}✓${NC} 完成"

# 3. 初始化 git
echo -e "${YELLOW}[3/6]${NC} 初始化 Git 仓库..."
if [ ! -d ".git" ]; then
    git init
    git config user.email "you@example.com"
    git config user.name "Your Name"
fi
echo -e "${GREEN}✓${NC} 完成"

# 4. 提交代码
echo -e "${YELLOW}[4/6]${NC} 提交代码..."
git add .
git commit -m "🚀 Initial release: NCU CUDA Profiling Skill v1.0.0" || echo -e "${YELLOW}可能已经提交过${NC}"
echo -e "${GREEN}✓${NC} 完成"

# 5. 添加远程仓库
echo -e "${YELLOW}[5/6]${NC} 配置远程仓库..."
REPO_URL="https://github.com/$USERNAME/ncu-cuda-profiling-skill.git"

# 检查是否已有 remote
if git remote | grep -q "origin"; then
    git remote remove origin
fi

git remote add origin $REPO_URL
echo -e "${GREEN}✓${NC} 远程仓库: $REPO_URL"

# 6. 创建 GitHub 仓库（如果 gh CLI 可用）
echo -e "${YELLOW}[6/6]${NC} 创建 GitHub 仓库..."
if command -v gh &> /dev/null; then
    echo "使用 GitHub CLI 创建仓库..."
    gh repo create ncu-cuda-profiling-skill --public --source=. --push || {
        echo -e "${YELLOW}仓库可能已存在，尝试直接推送...${NC}"
    }
else
    echo -e "${YELLOW}GitHub CLI 未安装${NC}"
    echo "请在浏览器中手动创建仓库:"
    echo -e "${BLUE}https://github.com/new${NC}"
    echo ""
    echo "仓库名称: ncu-cuda-profiling-skill"
    echo "可见性: Public"
    echo "然后按回车继续..."
    read
fi

# 推送代码
echo ""
echo "推送代码到 GitHub..."
git branch -M main
git push -u origin main || {
    echo -e "${RED}推送失败${NC}"
    echo "请检查:"
    echo "1. GitHub 仓库是否已创建"
    echo "2. 是否有推送权限 (需要配置 SSH key 或输入密码)"
    echo ""
    echo "手动推送命令:"
    echo "  git push -u origin main"
    exit 1
}

# 创建标签
echo ""
echo "创建 Release 标签..."
git tag -a v1.0.0 -m "🎉 Release v1.0.0: Initial release" || echo -e "${YELLOW}标签已存在${NC}"
git push origin v1.0.0 || echo -e "${YELLOW}标签推送失败${NC}"

echo ""
echo "========================================"
echo -e "${GREEN}🎉 发布完成！${NC}"
echo "========================================"
echo ""
echo "📦 仓库地址:"
echo -e "   ${BLUE}https://github.com/$USERNAME/ncu-cuda-profiling-skill${NC}"
echo ""
echo "🏷️  Release:"
echo -e "   ${BLUE}https://github.com/$USERNAME/ncu-cuda-profiling-skill/releases${NC}"
echo ""
echo "📖 下一步:"
echo "   1. 在 GitHub 上创建 Release 说明"
echo "   2. 分享给社区使用"
echo "   3. 收集反馈持续改进"
echo ""
