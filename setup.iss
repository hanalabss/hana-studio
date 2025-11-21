; Hana Studio 설치 스크립트 - 컴파일 오류 해결
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
; 🔧 메인 실행파일 및 모든 종속 파일들 (recursesubdirs로 전체 복사)
; Changed from release_fast to dist/HanaStudio
Source: "dist\HanaStudio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; 🔧 핵심 DLL들을 명시적으로 추가 (간단하게 처리)
; Source: "dist\HanaStudio\libDSRetransfer600App.dll"; DestDir: "{app}"; Flags: ignoreversion
; Source: "dist\HanaStudio\Retransfer600_SDKCfg.xml"; DestDir: "{app}"; Flags: ignoreversion

; 🔧 원본 DLL 파일들도 백업으로 포함 (dll 폴더에서)
Source: "dll\libDSRetransfer600App.dll"; DestDir: "{app}\dll"; Flags: ignoreversion skipifsourcedoesntexist
Source: "dll\Retransfer600_SDKCfg.xml"; DestDir: "{app}\dll"; Flags: ignoreversion skipifsourcedoesntexist
Source: "dll\*.EWL"; DestDir: "{app}\dll"; Flags: ignoreversion skipifsourcedoesntexist
Source: "dll\R600StatusReference*"; DestDir: "{app}\dll"; Flags: ignoreversion skipifsourcedoesntexist

; 🔧 설정 파일들
Source: "config.json"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "hana.ico"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

; 🔧 Visual C++ 재배포 가능 패키지 (필요한 경우)
; Source: "vcredist_x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Dirs]
; 🔧 필요한 디렉토리들 미리 생성
Name: "{app}\dll"
Name: "{app}\temp"
Name: "{app}\output"
Name: "{app}\logs"
Name: "{app}\_internal"

[Icons]
; 시작메뉴 바로가기
Name: "{group}\Hana Studio"; Filename: "{app}\HanaStudio.exe"; IconFilename: "{app}\hana.ico"; WorkingDir: "{app}"
Name: "{group}\Hana Studio 제거"; Filename: "{uninstallexe}"

; 바탕화면 바로가기
Name: "{autodesktop}\Hana Studio"; Filename: "{app}\HanaStudio.exe"; IconFilename: "{app}\hana.ico"; Tasks: desktopicon; WorkingDir: "{app}"

; 빠른 실행 바로가기 (Windows 10 이하)
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\Hana Studio"; Filename: "{app}\HanaStudio.exe"; IconFilename: "{app}\hana.ico"; Tasks: quicklaunchicon; WorkingDir: "{app}"

[Run]
; 설치 완료 후 실행 옵션
Filename: "{app}\HanaStudio.exe"; Description: "Hana Studio 시작하기"; Flags: nowait postinstall skipifsilent; WorkingDir: "{app}"

; Visual C++ 재배포 가능 패키지 설치 (필요한 경우)
; Filename: "{tmp}\vcredist_x64.exe"; Parameters: "/quiet"; StatusMsg: "Visual C++ 런타임 설치 중..."; Flags: waituntilterminated

[Registry]
; 파일 연결 및 프로그램 등록
Root: HKCU; Subkey: "Software\Hana Studio"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"
Root: HKCU; Subkey: "Software\Hana Studio"; ValueType: string; ValueName: "Version"; ValueData: "1.0.0"
Root: HKCU; Subkey: "Software\Hana Studio"; ValueType: string; ValueName: "DLLPath"; ValueData: "{app}"

; 🔧 PATH 환경변수에 DLL 경로 추가 (현재 사용자만)
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "PATH"; ValueData: "{olddata};{app}"; Check: NeedsAddPath('{app}')

[UninstallDelete]
; 제거 시 삭제할 추가 파일들 (사용자가 생성한 파일들)
Type: filesandordirs; Name: "{app}\temp"
Type: filesandordirs; Name: "{app}\output"
Type: filesandordirs; Name: "{app}\logs"

[Code]
// 🔧 사용자 정의 함수들

// DLL이 이미 존재하는지 확인
function DLLAlreadyExists: Boolean;
begin
  Result := FileExists(ExpandConstant('{app}\libDSRetransfer600App.dll'));
end;

// Config 파일이 이미 존재하는지 확인
function ConfigAlreadyExists: Boolean;
begin
  Result := FileExists(ExpandConstant('{app}\Retransfer600_SDKCfg.xml'));
end;

// PATH에 경로 추가가 필요한지 확인
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'PATH', OrigPath)
  then begin
    Result := True;
    exit;
  end;
  // 이미 경로가 있는지 확인 (대소문자 무시)
  Result := Pos(';' + UpperCase(Param) + ';', ';' + UpperCase(OrigPath) + ';') = 0;
end;

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

// 🔧 DLL 파일 존재 확인 함수
function CheckDLLExists: Boolean;
var
  DLLPath: String;
begin
  DLLPath := ExpandConstant('{app}\libDSRetransfer600App.dll');
  Result := FileExists(DLLPath);
  if not Result then begin
    MsgBox('필수 DLL 파일이 없습니다: ' + DLLPath, mbError, MB_OK);
  end;
end;

// 설치 전/후 처리
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep=ssInstall) then
  begin
    if (IsUpgrade()) then
    begin
      UnInstallOldVersion();
    end;
  end;
  
  // 🔧 설치 완료 후 DLL 파일 확인
  if (CurStep=ssPostInstall) then
  begin
    if not CheckDLLExists then
    begin
      MsgBox('DLL 파일 설치에 문제가 있습니다. 기술 지원에 문의하세요.', mbError, MB_OK);
    end;
  end;
end;

// 설치 마법사 초기화
procedure InitializeWizard;
begin
  // 설치 마법사 초기화 시 필요한 작업
end;

// 다음 버튼 클릭 처리
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
      // 경고만 표시하고 계속 진행
    end;
  end;
end;

// 제거 프로그램 초기화
function InitializeUninstall(): Boolean;
begin
  Result := True;
  // 제거 시 필요한 초기화 작업
end;

[Messages]
; 한국어 메시지 사용자 정의
WelcomeLabel2=컴퓨터에 [name/ver]을(를) 설치합니다.%n%n계속하기 전에 다른 모든 프로그램을 종료하는 것이 좋습니다.
ClickNext=계속하려면 [다음]을 클릭하거나, 설치를 종료하려면 [취소]를 클릭하세요.
SelectDirDesc=어디에 [name]을(를) 설치하시겠습니까?
SelectDirLabel3=설치 프로그램이 [name]을(를) 다음 폴더에 설치합니다.
DiskSpaceMBLabel=적어도 [mb] MB의 사용 가능한 디스크 공간이 필요합니다.

[CustomMessages]
; 사용자 정의 메시지
AppIsRunning=Hana Studio가 현재 실행 중입니다. 프로그램을 종료한 후 설치를 계속하세요.
LaunchProgram=Hana Studio 시작하기
InstallationComplete=Hana Studio 설치가 완료되었습니다!
SystemRequirements=시스템 요구사항:%n- Windows 10 64bit 이상%n- 메모리: 4GB RAM 권장%n- 저장공간: 1GB 이상%n- 프린터: RTAI LUKA R600 호환
InstallPath=설치 경로: {app}
DLLInstallSuccess=프린터 DLL이 성공적으로 설치되었습니다.
DLLInstallFailed=프린터 DLL 설치에 실패했습니다. 수동으로 확인이 필요합니다.