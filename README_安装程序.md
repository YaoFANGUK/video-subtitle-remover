# 视频字幕去除器 - 快速开始指南

## 📦 打包和分发

### 选项一：便携版本（推荐用于测试）

```
dist/VideoSubtitleRemover/VideoSubtitleRemover.exe
```

**特点**：
- ✅ 无需安装，解压即用
- ✅ 使用 `隐藏控制台启动.vbs` 完全隐藏控制台
- ✅ 适合个人使用和测试

### 选项二：安装程序版本（推荐用于分发）

#### 编译安装程序

1. **安装 Inno Setup Compiler**
   - 下载：https://jrsoftware.org/isdl.php
   - 选择 "Inno Setup 6.x" 版本

2. **先编译主程序**
   ```bash
   build_windows.bat
   ```

3. **编译安装程序**
   ```bash
   build_installer.bat
   ```

**生成的安装程序**：
```
Output/VideoSubtitleRemover-Setup.exe (约 1.8 GB)
```

#### 安装程序功能

✅ **创建桌面快捷方式**
✅ **添加到开始菜单** ("视频字幕去除器")
✅ **完整卸载程序**
✅ **管理员权限检查**
✅ **多语言支持** (简体中文、英文)

#### 分发方式

1. **直接分发**：发送 `VideoSubtitleRemover-Setup.exe` 给用户
2. **在线发布**：上传到 GitHub Releases
3. **分卷压缩**：使用 7z 创建分卷（每卷 2GB）

## 🚀 快速开始

### 方式一：使用便携版本

1. 下载或获取 `dist/VideoSubtitleRemover/` 目录
2. 双击 `隐藏控制台启动.vbs` 启动程序
3. 开始使用

### 方式二：使用安装程序

1. 双击 `VideoSubtitleRemover-Setup.exe`
2. 以管理员身份运行
3. 按照安装向导完成
4. 通过桌面快捷方式或开始菜单启动

## 📁 文件结构

```
dist/VideoSubtitleRemover/
├── VideoSubtitleRemover.exe              # 主程序
├── 隐藏控制台启动.vbs               # 无控制台启动器 ⭐
├── 启动程序.bat                     # 备选启动器
├── _internal/                           # 程序依赖
│   ├── backend/                        # 后端模块
│   │   ├── models/                   # AI 模型
│   │   ├── interface/                 # 多语言文件
│   │   └── ffmpeg/                   # FFmpeg
│   ├── config/                         # 配置文件
│   └── ...                         # 其他依赖
```

## 🎯 用户使用

### 第一次运行

1. 程序可能需要 1-2 分钟初始化 AI 模型
2. 首次运行时所有功能可能稍慢
3. 后续运行会更快

### 常见问题

**Q: 控制台窗口闪一下就消失？**
A: 这是正常的，某些库初始化时会短暂创建控制台

**Q: 程序启动慢？**
A: 首次运行需要加载 AI 模型，后续运行会快很多

**Q: 需要网络连接？**
A: 不需要，所有功能都在本地运行

## 🔧 高级配置

### 自定义安装位置

编辑 `VideoSubtitleRemover.iss` 中的：
```ini
DefaultDirName={autopf}\Program Files\视频字幕去除器
```

### 添加更多语言

在 `compiler/Languages/` 目录中添加新的 `.isl` 文件，然后在 `.iss` 中添加：
```ini
[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"
```

## 📝 版本管理

### 更新版本号

1. 修改 `backend/config.py` 中的 `VERSION`
2. 重新运行 `build_windows.bat`
3. 重新运行 `build_installer.bat`

### 版本发布流程

1. 更新版本号
2. 编译安装程序
3. 上传到 GitHub Releases
4. 创建 Release 标签和说明
5. 通知用户更新

---

**注意**: 安装程序文件较大 (~1.8 GB)，建议：
- 使用高速网络分发
- 提供分卷下载选项
- 确保用户有足够磁盘空间
