; Hana Studio 설치 스크립트 - Inno Setup
; 파일명: setup.iss

[Setup]
; 기본 정보
AppName=Hana Studio
AppVersion=1.0.0
AppVerName=Hana Studio 1.0.0
AppPublisher=Hana Studio Team
AppPublisherURL=https://hanastudio.com
AppSupportURL=https://hanastudio.com/support
AppUpdatesURL=https://hanastudio.com/updates
AppCopyright=Copyright (C) 2025 Hana Studio Team

; 설치 경로 및 그룹
DefaultDirName={autopf}\Hana Studio
DefaultGroupName=Hana Studio
AllowNoIcons=yes

; 출력 설정
OutputDir=installer_output
OutputBaseFilename=HanaStudio_Setup_v1.0.0
SetupIconFile=hana.ico
Compression=lzma2/ultra64
SolidCompression=yes

; 호환성
WizardStyle=modern
MinVersion=10.0.17763
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

; 권한 및 기타
PrivilegesRequired=admin
DisableProgramGroupPage=yes

; 제거 프로그램
UninstallDisplayIcon={app}\HanaStudio.exe
UninstallDisplayName=Hana Studio

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "바탕화면에 바로가기 만들기"; GroupDescription: "추가 작업:"; Flags: unchecked
Name: "quicklaunchicon"; Description: "빠른 실행에 바로가기 만들기"; GroupDescription: "추가 작업:"; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
; 메인 실행파일 및 핵심 파일들
Source: "release_fast\HanaStudio.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "release_fast\*.dll"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "release_fast\*.pyd"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "release_fast\*.exe"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "release_fast\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; 프린터 관련 핵심 파일들 (dll 폴더에서)
Source: "dll\libDSRetransfer600App.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "dll\Retransfer600_SDKCfg.xml"; DestDir: "{app}"; Flags: ignoreversion
Source: "dll\EWL"; DestDir: "{app}"; Flags: ignoreversion
Source: "dll\R600StatusReference*"; DestDir: "{app}"; Flags: ignoreversion

; 설정 파일들
Source: "config.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "hana.ico"; DestDir: "{app}"; Flags: ignoreversion


; Visual C++ 재배포 가능 패키지 (필요한 경우)
; Source: "vcredist_x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
; 시작메뉴 바로가기
Name: "{group}\Hana Studio"; Filename: "{app}\HanaStudio.exe"; IconFilename: "{app}\hana.ico"
Name: "{group}\Hana Studio 제거"; Filename: "{uninstallexe}"

; 바탕화면 바로가기
Name: "{autodesktop}\Hana Studio"; Filename: "{app}\HanaStudio.exe"; IconFilename: "{app}\hana.ico"; Tasks: desktopicon

; 빠른 실행 바로가기 (Windows 10 이하)
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\Hana Studio"; Filename: "{app}\HanaStudio.exe"; IconFilename: "{app}\hana.ico"; Tasks: quicklaunchicon

[Run]
; 설치 완료 후 실행 옵션
Filename: "{app}\HanaStudio.exe"; Description: "Hana Studio 시작하기"; Flags: nowait postinstall skipifsilent

; Visual C++ 재배포 가능 패키지 설치 (필요한 경우)
; Filename: "{tmp}\vcredist_x64.exe"; Parameters: "/quiet"; StatusMsg: "Visual C++ 런타임 설치 중..."; Flags: waituntilterminated

[Registry]
; 파일 연결 및 프로그램 등록
Root: HKCU; Subkey: "Software\Hana Studio"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"
Root: HKCU; Subkey: "Software\Hana Studio"; ValueType: string; ValueName: "Version"; ValueData: "1.0.0"

[UninstallDelete]
; 제거 시 삭제할 추가 파일들 (사용자가 생성한 파일들)
Type: filesandordirs; Name: "{app}\temp"
Type: filesandordirs; Name: "{app}\output"
Type: filesandordirs; Name: "{app}\logs"

[Code]
// 사용자 정의 함수들

// .NET Framework 확인
function IsDotNetInstalled: Boolean;
begin
  Result := RegKeyExists(HKLM, 'SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full');
end;

// 이전 버전 확인 및 제거
function GetUninstallString: String;
var
  sUnInstPath: String;
  sUnInstallString: String;
begin
  sUnInstPath := ExpandConstant('Software\Microsoft\Windows\CurrentVersion\Uninstall\Hana Studio_is1');
  sUnInstallString := '';
  if not RegQueryStringValue(HKLM, sUnInstPath, 'UninstallString', sUnInstallString) then
    RegQueryStringValue(HKCU, sUnInstPath, 'UninstallString', sUnInstallString);
  Result := sUnInstallString;
end;

function IsUpgrade: Boolean;
begin
  Result := (GetUninstallString() <> '');
end;

function UnInstallOldVersion(): Integer;
var
  sUnInstallString: String;
  iResultCode: Integer;
begin
  Result := 0;
  sUnInstallString := GetUninstallString();
  if sUnInstallString <> '' then begin
    sUnInstallString := RemoveQuotes(sUnInstallString);
    if Exec(sUnInstallString, '/SILENT /NORESTART /SUPPRESSMSGBOXES','', SW_HIDE, ewWaitUntilTerminated, iResultCode) then
      Result := 3
    else
      Result := 2;
  end else
    Result := 1;
end;

// 설치 전 초기화
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep=ssInstall) then
  begin
    if (IsUpgrade()) then
    begin
      UnInstallOldVersion();
    end;
  end;
end;

// 설치 마법사 사용자 정의
procedure InitializeWizard;
begin
  // 설치 마법사 초기화 시 필요한 작업
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  
  // 시스템 요구사항 확인
  if CurPageID = wpWelcome then
  begin
    if not IsDotNetInstalled then
    begin
      MsgBox('이 프로그램을 실행하려면 .NET Framework 4.7.2 이상이 필요합니다.' + #13#10 + 
             'Microsoft 웹사이트에서 다운로드하여 설치해주세요.', mbInformation, MB_OK);
      Result := False;
    end;
  end;
end;

[Messages]
; 한국어 메시지 사용자 정의
WelcomeLabel2=컴퓨터에 [name/ver]을(를) 설치합니다.%n%n계속하기 전에 다른 모든 프로그램을 종료하는 것이 좋습니다.
ClickNext=계속하려면 [다음]을 클릭하거나, 설치를 종료하려면 [취소]를 클릭하세요.
SelectDirDesc=어디에 [name]을(를) 설치하시겠습니까?
SelectDirLabel3=설치 프로그램이 [name]을(를) 다음 폴더에 설치합니다.
DiskSpaceMBLabel=적어도 [mb] MB의 사용 가능한 디스크 공간이 필요합니다.
ToUNCPathname=유니코드 경로를 지원하지 않습니다. 다른 폴더를 선택해주세요.
InvalidPath=드라이브 문자를 포함한 전체 경로를 입력해야 합니다. 예:%n%nC:\APP%n%n또는 UNC 형식:%n%n\\server\share
InvalidDrive=선택한 드라이브나 UNC 공유가 없거나 액세스할 수 없습니다. 다른 것을 선택해주세요.
DiskSpaceWarning=설치하려면 적어도 %1 KB의 사용 가능한 공간이 필요하지만, 선택한 드라이브에는 %2 KB만 사용할 수 있습니다.%n%n그래도 계속하시겠습니까?
DirNameTooLong=폴더 이름이나 경로가 너무 깁니다.
InvalidDirName=폴더 이름이 올바르지 않습니다.
BadDirName32=폴더 이름에는 다음 문자를 사용할 수 없습니다:%n%n%1
DirExistsTitle=폴더가 있습니다
DirExists=다음 폴더:%n%n%1%n%n이(가) 이미 있습니다. 그래도 해당 폴더에 설치하시겠습니까?
DirDoesntExistTitle=폴더가 없습니다
DirDoesntExist=다음 폴더:%n%n%1%n%n이(가) 없습니다. 폴더를 만드시겠습니까?

[CustomMessages]
; 사용자 정의 메시지
AppIsRunning=Hana Studio가 현재 실행 중입니다. 프로그램을 종료한 후 설치를 계속하세요.
LaunchProgram=Hana Studio 시작하기
InstallationComplete=Hana Studio 설치가 완료되었습니다!
SystemRequirements=시스템 요구사항:%n- Windows 10 64bit 이상%n- 메모리: 4GB RAM 권장%n- 저장공간: 1GB 이상
InstallPath=설치 경로: {app}