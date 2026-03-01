; Inno Setup script for Detector de Inventario
; Compile with: iscc installer\setup.iss
; Requires Inno Setup 6.x

#define AppName      "Detector de Inventario"
#define AppVersion   "1.0.0"
#define AppPublisher "Tu Empresa"
#define AppExeName   "DetectorInventario.exe"
#define BundleDir    "..\dist\DetectorInventario"

[Setup]
AppId={{A3F8C2D1-7B4E-4F9A-B6D2-0E5C1A8F3B7E}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL=https://github.com/tu-usuario/Object-Detection-Model
AppSupportURL=https://github.com/tu-usuario/Object-Detection-Model/issues
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
; Output
OutputDir=output
OutputBaseFilename=DetectorInventario_Setup
; Compression
Compression=lzma2/ultra64
SolidCompression=yes
; Architecture — only x64 Windows
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
; Appearance
WizardStyle=modern
; Minimum Windows version: Windows 10
MinVersion=10.0.17763
; Privileges — try to install without admin first, fall back to admin
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline dialog
; Icon (shown in Add/Remove Programs)
UninstallDisplayIcon={app}\{#AppExeName}
; Version info shown in Programs list
VersionInfoVersion={#AppVersion}
VersionInfoDescription={#AppName}

[Languages]
Name: "spanish";  MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english";  MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Crear acceso directo en el Escritorio"; GroupDescription: "Iconos adicionales:"; Flags: unchecked

[Files]
; Copy the entire PyInstaller onedir bundle
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Start Menu shortcut
Name: "{group}\{#AppName}";       Filename: "{app}\{#AppExeName}"
Name: "{group}\Desinstalar {#AppName}"; Filename: "{uninstallexe}"
; Desktop shortcut (optional, only if task selected)
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
; Offer to launch after install
Filename: "{app}\{#AppExeName}"; Description: "Iniciar {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Remove user data only if user confirms (we do NOT delete %APPDATA%\DetectorInventario automatically)
; The uninstaller removes only the program files; user data stays intact.

[Messages]
; Spanish overrides for important messages
spanish.WelcomeLabel2=Este asistente instalará {#AppName} {#AppVersion} en su equipo.%n%nSe recomienda cerrar todas las demás aplicaciones antes de continuar.
spanish.FinishedHeadingLabel=Instalación completada

[CustomMessages]
spanish.FirstRunNote=Nota: los modelos de inteligencia artificial (≈100 MB) se descargarán automáticamente la primera vez que use la aplicación. Asegúrese de tener conexión a internet.
english.FirstRunNote=Note: AI models (~100 MB) will be downloaded automatically the first time you launch the application. Please ensure you have an internet connection.

[Code]
procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpFinished then
    WizardForm.FinishedLabel.Caption :=
      WizardForm.FinishedLabel.Caption + #13#10 + #13#10 +
      CustomMessage('FirstRunNote');
end;
