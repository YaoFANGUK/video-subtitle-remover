@echo off
chcp 65001 >nul
echo ========================================
echo   视频字幕去除器 - PyInstaller 打包脚本
echo ========================================
echo.

REM 读取版本号
for /f "tokens=2 delims='" %%a in ('findstr /C:"VERSION = " backend\config.py') do set VERSION=%%a
echo 版本号: %VERSION%
echo.

REM 检查是否安装了 PyInstaller
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo 错误: 未安装 PyInstaller
    echo 正在安装 PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo 安装失败，请手动运行: pip install pyinstaller
        pause
        exit /b 1
    )
)

REM 清理旧的构建文件
echo [1/5] 清理旧的构建文件...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
echo ✓ 清理完成
echo.

REM 执行 PyInstaller 打包
echo [2/5] 开始 PyInstaller 打包...
echo 这可能需要几分钟时间，请耐心等待...
echo.
pyinstaller --clean --noconfirm VideoSubtitleRemover.spec
if errorlevel 1 (
    echo ✗ 打包失败！
    pause
    exit /b 1
)
echo ✓ 打包完成
echo.

REM 重命名输出目录
echo [3/5] 重命名输出目录...
set OUTPUT_NAME=VideoSubtitleRemover-Windows-v%VERSION%
if exist dist\%OUTPUT_NAME% rmdir /s /q dist\%OUTPUT_NAME%
move dist\VideoSubtitleRemover dist\%OUTPUT_NAME%
if errorlevel 1 (
    echo ✗ 重命名失败！
    pause
    exit /b 1
)
echo ✓ 重命名完成: %OUTPUT_NAME%
echo.

REM 检查是否安装了 7z
echo [4/5] 检查 7z 压缩工具...
where 7z >nul 2>&1
if errorlevel 1 (
    echo 警告: 未找到 7z，跳过压缩步骤
    echo 请手动安装 7-Zip: https://www.7-zip.org/
    goto :skip_compression
)

REM 创建 7z 压缩包
echo 开始压缩（可能需要几分钟）...
cd dist\%OUTPUT_NAME%
7z a -t7z -mx=9 -m0=LZMA2 -ms=on -mfb=64 -md=32m -mmt=on -v2000m "..\vsr-v%VERSION%-windows-cpu.7z" *
cd ..\..

REM 检查是否只有一个分卷
if exist "vsr-v%VERSION%-windows-cpu.7z.001" (
    if not exist "vsr-v%VERSION%-windows-cpu.7z.002" (
        rename "vsr-v%VERSION%-windows-cpu.7z.001" "vsr-v%VERSION%-windows-cpu.7z"
        echo ✓ 压缩完成（单文件）
    ) else (
        echo ✓ 压缩完成（分卷）
    )
) else if exist "vsr-v%VERSION%-windows-cpu.7z" (
    echo ✓ 压缩完成（单文件）
) else (
    echo ✗ 压缩失败！
    goto :skip_compression
)

:skip_compression
echo.

REM 显示构建结果
echo [5/5] 构建结果摘要
echo ========================================
echo 输出目录: dist\%OUTPUT_NAME%
echo.

if exist "vsr-v%VERSION%-windows-cpu.7z" (
    echo 压缩包: vsr-v%VERSION%-windows-cpu.7z
    for %%F in ("vsr-v%VERSION%-windows-cpu.7z") do echo 文件大小: %%~zF 字节
) else if exist "vsr-v%VERSION%-windows-cpu.7z.001" (
    echo 压缩包: vsr-v%VERSION%-windows-cpu.7z.* (分卷)
)

echo.
echo ========================================
echo ✓ 构建成功完成！
echo ========================================
echo.
echo 下一步操作：
echo 1. 测试运行: dist\%OUTPUT_NAME%\VideoSubtitleRemover.exe
echo 2. 分发压缩包（如果生成了）
echo.

pause