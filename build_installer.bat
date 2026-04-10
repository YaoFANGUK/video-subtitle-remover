@echo off
chcp 65001 >nul
echo ========================================
echo   视频字幕去除器 - 安装程序编译
echo ========================================
echo.

REM 检查 Inno Setup Compiler
where iscc >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Inno Setup Compiler
    echo.
    echo 下载地址: https://jrsoftware.org/isdl.php
    echo.
    echo 选择 "Inno Setup 6.x" 版本下载安装
    echo.
    pause
    exit /b 1
)

echo [1/2] 读取版本号...
for /f "tokens=2 delims='" %%a in ('findstr /C:"VERSION = " backend\config.py') do set VERSION=%%a
echo 版本号: %VERSION%
echo.

REM 检查 dist 目录
if not exist dist\VideoSubtitleRemover (
    echo 错误: 找到 dist\VideoSubtitleRemover 目录
    echo 请先运行 PyInstaller 打包程序
    pause
    exit /b 1
)

echo [2/2] 开始编译安装程序...
echo.
iscc VideoSubtitleRemover.iss
if errorlevel 1 (
    echo ✗ 编译失败！
    pause
    exit /b 1
)
echo ✓ 编译完成
echo.

REM 检查生成的安装程序
if exist Output\VideoSubtitleRemover-Setup.exe (
    echo ========================================
    echo   编译成功！
    echo ========================================
    echo.
    echo 安装程序位置: Output\VideoSubtitleRemover-Setup.exe
    for %%F in ("Output\VideoSubtitleRemover-Setup.exe") do echo 文件大小: %%~zF 字节
    echo.
    echo 功能特点:
    echo   - ✓ 创建桌面快捷方式
    echo   - ✓ 添加到开始菜单
    echo   - ✓ 完整的卸载程序
    echo   - ✓ 支持简体中文和英文界面
    echo.
    echo 下一步操作:
    echo   1. 测试安装: 右键点击安装程序 → 以管理员身份运行
    echo   2. 分发: Output\VideoSubtitleRemover-Setup.exe
) else (
    echo ✗ 编译成功但找不到输出文件
    pause
    exit /b 1
)

echo.
pause
