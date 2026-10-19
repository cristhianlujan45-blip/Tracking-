; Forja3D installer for Windows (Inno Setup 6). Built by .github/workflows/desktop.yml:
;   iscc /DAppVersion=1.0.0 desktop\windows\forja3d.iss
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{6F3C2A51-9D84-4B7E-A0F2-F0A3D3E1C7B9}
AppName=Forja3D
AppVersion={#AppVersion}
AppVerName=Forja3D {#AppVersion}
AppPublisher=Forja3D
DefaultDirName={autopf}\Forja3D
DefaultGroupName=Forja3D
DisableProgramGroupPage=yes
; installs for the current user without admin rights (can be changed in the first page)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\..\dist-installer
OutputBaseFilename=Forja3D-Setup-{#AppVersion}
SetupIconFile=..\assets\forja3d.ico
UninstallDisplayIcon={app}\Forja3D.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\..\dist\Forja3D\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "MicrosoftEdgeWebview2Setup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: NeedsWebView2

[Icons]
Name: "{group}\Forja3D"; Filename: "{app}\Forja3D.exe"
Name: "{group}\{cm:UninstallProgram,Forja3D}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Forja3D"; Filename: "{app}\Forja3D.exe"; Tasks: desktopicon

[Run]
Filename: "{tmp}\MicrosoftEdgeWebview2Setup.exe"; Parameters: "/silent /install"; StatusMsg: "WebView2…"; Check: NeedsWebView2; Flags: waituntilterminated
Filename: "{app}\Forja3D.exe"; Description: "{cm:LaunchProgram,Forja3D}"; Flags: nowait postinstall skipifsilent

[Code]
// The app window uses Microsoft Edge WebView2 (already present on Windows 11 and up-to-date Windows 10).
function NeedsWebView2(): Boolean;
var
  Version: String;
begin
  Result := not (
    RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) or
    RegQueryStringValue(HKLM, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) or
    RegQueryStringValue(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version))
    or (Version = '') or (Version = '0.0.0.0');
end;
