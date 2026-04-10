; 视频字幕去除器 - Inno Setup 安装程序脚本
; 编译方法：下载 Inno Setup Compiler (https://jrsoftware.org/isdl.php)
; 或者使用命令：iscc VideoSubtitleRemover.iss

; 定义版本号
#define MyAppVersion "1.4.0"

[Setup]
AppName=视频字幕去除器
AppVersion={#MyAppVersion}
AppPublisher=YaoFANGUK
AppPublisherURL=https://github.com/YaoFANGUK/video-subtitle-remover
AppSupportURL=https://github.com/YaoFANGUK/video-subtitle-remover/issues
AppComments=基于AI的图片/视频硬字幕去除工具
DefaultDirName={autopf}\Program Files\视频字幕去除器
DefaultGroupName=视频字幕去除器
AllowNoIcons=yes
OutputBaseFilename=VideoSubtitleRemover-Setup-v{#MyAppVersion}
Compression=lzma
SolidCompression=yes
; 内部文件需要更多内存，禁用
InternalCompressLevel=none
SetupIconFile=design\vsr.ico
; 使用默认向导图片，避免bitmap格式错误
UninstallDisplayIcon={app}\VideoSubtitleRemover.exe
VersionInfoVersion=1.4.0
VersionInfoCompany=YaoFANGUK
VersionInfoDescription=视频字幕去除器
VersionInfoCopyright=© 2026 YaoFANGUK
VersionInfoProductName=视频字幕去除器

[Files]
Source: "dist\VideoSubtitleRemover\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "dist\VideoSubtitleRemover\VideoSubtitleRemover.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\视频字幕去除器"; Filename: "{app}\VideoSubtitleRemover.exe"
Name: "{userdesktop}\视频字幕去除器"; Filename: "{app}\VideoSubtitleRemover.exe"
Name: "{commonstartup}\视频字幕去除器"; Filename: "{app}\VideoSubtitleRemover.exe"

[Run]
Filename: "{app}\VideoSubtitleRemover.exe"; Description: "启动视频字幕去除器"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
