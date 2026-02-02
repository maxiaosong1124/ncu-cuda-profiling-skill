# 🚀 最终发布指南

## 你需要提供的信息：

1. **GitHub 用户名**: _____________
2. **Git 邮箱**: _____________
3. **Git 姓名**: _____________

## 完整发布命令：

```bash
# 进入目录
cd /home/maxiaosong/work_space/cuda_learning/cuda_code/ncu-cuda-profiling-skill

# 替换用户名
sed -i 's/yourusername/YOUR_USERNAME/g' README.md SKILL.md RELEASE.md

# 配置 Git
git config user.email "YOUR_EMAIL"
git config user.name "YOUR_NAME"

# 提交
git add .
git commit -m "🚀 Initial release: NCU CUDA Profiling Skill v1.0.0"

# 推送到 GitHub
git remote add origin https://github.com/YOUR_USERNAME/ncu-cuda-profiling-skill.git
git branch -M main
git push -u origin main

# 创建标签
git tag -a v1.0.0 -m "🎉 Release v1.0.0"
git push origin v1.0.0
```

## 发布后：

1. 访问: https://github.com/YOUR_USERNAME/ncu-cuda-profiling-skill
2. 点击 Releases → Create a new release
3. 选择标签 v1.0.0
4. 填写发布说明（参考 RELEASE.md）
5. 点击 Publish release

