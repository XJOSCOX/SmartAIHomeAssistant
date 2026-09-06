; Build with Inno Setup 6: ISCC.exe packaging\Jake.iss
#define AppVersion "0.1.0"
[Setup]
AppId={{60E85E52-14BA-4855-B2DD-52D63EFC20E1}
AppName=Jake
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\Jake
DefaultGroupName=Jake
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installer
OutputBaseFilename=JakeSetup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
UninstallDisplayIcon={app}\Jake.exe
[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: unchecked
[Files]
Source: "..\dist\Jake\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{group}\Jake"; Filename: "{app}\Jake.exe"
Name: "{autodesktop}\Jake"; Filename: "{app}\Jake.exe"; Tasks: desktopicon
; Intentionally no Run section: installation does not launch a camera or the app.
; Uninstall does not delete %LOCALAPPDATA%\Jake or the user's vault keys.
