# R600 SDK 함수 참조와 카드프레소 호출 방식

작성일: 2026-11-12

로드맵 5단계(SDK 기능 확장)에 필요한 함수 인자와 호출 순서를 정리한 문서다. 비교·로드맵 본문은 `card-software-comparison-and-roadmap.md`에 있다.

## 1. 근거와 신뢰 수준

| 내용 | 출처 | 신뢰 수준 |
|---|---|---|
| 함수 이름, 인자 개수·자료형·이름 (2장) | 하나스튜디오의 `dll/libDSRetransfer600App.dll`(64비트). 내보낸 함수마다 인자 이름과 자료형을 로그에 남기는 코드가 있어 그대로 읽었다. Ghidra로 87개 함수를 디컴파일했고, 로그상 인자 수와 디컴파일된 인자 수가 전부 일치했다. | 높음 |
| 카드프레소가 넘기는 값과 호출 순서 (3장) | `cardPresso1.7.70.exe`(32비트)의 RTAI 인쇄 코드(`cpRtaiPrintSdk`, `cpDascomRtaiPrintJobSession`)를 Ghidra로 디컴파일해 호출 지점의 인자와 주변 로그 문구를 읽었다. | 중간 (값은 코드에서 확인, 분기 조건 일부는 추정) |
| 값의 의미 (모드 번호, 위치 번호 등) | 인자 이름과 카드프레소 로그 문구로 해석했다. | 중간~낮음 (실기 확인 필요) |

주의할 점:

- 실제 프린터로 호출해 본 것은 없다. 3장의 값은 "카드프레소가 이렇게 부른다"는 사실이지, 동작을 보증하는 값이 아니다.
- 카드프레소는 210MB 전체가 아니라 R600/RTAI 관련 함수 약 180개만 디컴파일했다. 이 범위 밖에서 일어나는 호출은 보지 못했다.
- 반환값은 대부분 `unsigned int` 오류 코드(0이 성공)다. `R600GetErrorInfo`, `R600GetErrorOuterInfo`는 1바이트 값을, `R600GetPrivateCompanyInfo`는 문자열 포인터를 돌려준다.
- 카드프레소의 32비트 DLL과 하나스튜디오의 64비트 DLL이 같은 버전인지는 확인하지 못했다. 카드프레소가 이름으로 불러오는 함수 31개 중 30개가 하나스튜디오 DLL에 있고(`R600TranslateSDKErrCode`만 없다), 호출 지점을 확인한 함수는 인자 수가 모두 맞았다.

---

## 2. SDK 함수 목록 (87개)

"하나스튜디오" 열은 현재 `printer/` 코드에서 호출하는지 여부다. 87개 중 18개를 쓰고 있다.

### 초기화와 연결

| 함수 | 인자 | 하나스튜디오 |
|---|---|---|
| `R600LibInit` | 없음 | 사용 중 |
| `R600LibClear` | 없음 | 사용 중 |
| `R600EnumUsbPrt` | `char* szEnumList`, `unsigned int* pEnumListLen`, `int* pNum` | 사용 중 |
| `R600EnumTcpPrt` | `char* szEnumList`, `unsigned int* pEnumListLen`, `int* pNum` | 사용 중 |
| `R600SelectPrt` | `const char* Prt` | 사용 중 |
| `R600TcpSetTimeout` | `int nReadTimeout`, `int nWriteTimeout` | 사용 중 |
| `R600UsbSetTimeout` | `int nReadTimeout`, `int nWriteTimeout` | 미사용 |
| `R600EthernetSetup` | `int mode`, `unsigned char* PrinterID`, `unsigned char* IPAddressType`, `unsigned int* StaticIPAddress`, `unsigned int* GatewayIPAddress`, `unsigned int* SubnetMask` | 미사용 |

### 상태와 정보 조회

| 함수 | 인자 | 하나스튜디오 |
|---|---|---|
| `R600QueryPrtStatus` | `short* pChassisTemp`, `short* pPrintheadTemp`, `short* pHeaterTemp`, `unsigned int* pMainStatus`, `unsigned int* pSubStatus`, `unsigned int* pErrorStatus`, `unsigned int* pWarningStatus`, `unsigned char* pMainCode`, `unsigned char* pSubCode` | 미사용 |
| `R600GetErrorInfo` | `unsigned int errcode`, `char* outputstr`, `int* len` | 미사용 |
| `R600GetErrorOuterInfo` | `unsigned int errcode`, `char* outputstr`, `int* len` | 미사용 |
| `R600GetCheckStatusMsg` | `unsigned int statusCode`, `char* outputstr`, `int* len` | 미사용 |
| `R600GenErrFile` | `const char* fileName` | 미사용 |
| `R600GetPrtInfo` | `unsigned int* pPrintheadPrintCount`, `unsigned int* pCardPrintCount`, `unsigned int* pMainFWVer`, `unsigned int* pSubFWVer`, `unsigned int* pFPGAVer` | 미사용 |
| `R600GetPrtManufInfo` | `char* szPrtName`, `int* pPrtNameLen`, `char* szManuf`, `int* pManufLen`, `char* szPrtSN`, `int* pPrtSNLen`, `char* szDate`, `int* pDateLen`, `char* szMainFWVer`, `int* pMainFWVerLen` | 미사용 |
| `R600GetPrtOptionStatus` | `int nMode`, `unsigned char* p3In1Reader`, `unsigned char* pContactCardRW`, `unsigned char* pContactlessCardRW`, `unsigned char* pExternalBox`, `unsigned char* pMagneticReader`, `unsigned char* pLaminatorLeft`, `unsigned char* pLaminatorRight` | 미사용 |
| `R600GetRbnAndFilmRemaining` | `unsigned short* pRbnRemaining`, `unsigned short* pFilmRemaining` | 미사용 |
| `R600GetSN` | `char Dev`, `char* szSN`, `int* szSNLen` | 미사용 |
| `R600GetPrivateCompanyInfo` | 없음 | 미사용 |
| `R600IsFeederNoEmpty` | `int* pflag` | 미사용 |
| `R600IsPrtHaveCard` | `unsigned char* pFlag` | 미사용 |
| `R600ReaderHaveCard` | `int* pFlag` | 미사용 |
| `R600GetCardPos` | `int* pPos` | 미사용 |
| `R600GetCardType` | `int* pType` | 미사용 |

### 프린터·리본 설정

| 함수 | 인자 | 하나스튜디오 |
|---|---|---|
| `R600RibbonSettingsRW` | `int nMode`, `unsigned char* pRibbonType`, `unsigned char* pFilmType`, `unsigned char* pRibbonNearEnd`, `unsigned char* pFilmNearEnd` | 미사용 |
| `R600CardSettingsRW` | `int nMode`, `unsigned short* pCardType`, `unsigned short* pFontOrthTime`, `unsigned short* pBackOrthTime`, `unsigned short* pStdCardExitPos`, `unsigned short* pErrorCardExitPos`, `unsigned short* pStdCardExitPos`, `unsigned short* pCardEjectSide` | 미사용 |
| `R600PrtSettingsRW` | `int nMode`, `unsigned char* pBuzzer`, `unsigned char* pPowerOffDelayTime`, `unsigned char* pSleepTime`, `char* pYellow_Color_Shift`, `char* pMazenta_Color_Shift`, `char* pCyan_Color_Shift`, `char* pBlack_Color_Shift`, `char* pYellow_Color_Shift_Back`, `char* pMazenta_Color_Shift_Back`, `char* pCyan_Color_Shift_Back`, `char* pBlack_Color_Shift_Back`, `short* p2ndPrintingOffset` | 미사용 |
| `R600SetRibbonOpt` | `unsigned char isWrite`, `unsigned int key`, `char* value`, `unsigned int valueLen` | 사용 중 |
| `R600SetHalfRibbonOffset` | `double offset` | 미사용 |
| `R600SetImgUtilIsNeedSet_K_ColorRibbonSplitPrint` | `bool flag` | 미사용 |
| `R600OprRetryCtrl` | `unsigned int iMode`, `unsigned char cTryTimes` | 미사용 |
| `R600PrtReset` | 없음 | 미사용 |
| `R600PrtRestart` | 없음 | 미사용 |
| `R600CPtrClnMsg` | `char* upMsg`, `char* downMsg` | 미사용 |

### 카드 이동

| 함수 | 인자 | 하나스튜디오 |
|---|---|---|
| `R600CardInject` | `unsigned char ucDestPos` | 사용 중 |
| `R600CardEject` | `unsigned char ucDestPos` | 사용 중 |
| `R600CardMove` | `unsigned char ucDestPos` | 미사용 |
| `R600CardMoveImpr` | `unsigned char ucDestPos` | 미사용 |
| `R600CardTurnover` | 없음 | 사용 중 |

### 캔버스와 이미지

| 함수 | 인자 | 하나스튜디오 |
|---|---|---|
| `R600SetCanvasPortrait` | `int nPortrait` | 사용 중 |
| `R600PrepareCanvas` | `int nChromaticMode`, `int nMonoChroMode` | 사용 중 |
| `R600ClearCanvas` | 없음 | 사용 중 |
| `R600CommitCanvas` | `char* szImgInfo`, `int* pImgInfoLen` | 사용 중 |
| `R600PrintDraw` | `const char* szImgFront`, `const char* szImgBack` | 사용 중 |
| `R600GetPreviewImg` | `const char* szImgInfo`, `char* preview`, `int* previewLen` | 미사용 |
| `R600SetImagePara` | `int nWhiteTransparent`, `int nRotation`, `float fScale` | 사용 중 |
| `R600SetImgVisualParam` | `int nBrightness`, `int nContrast`, `int nSaturation` | 미사용 |
| `R600SetAddImageMode` | `int nMode`, `int nValueThreshold` | 미사용 |
| `R600SetAddImageMode_Rtai` | `int nMode`, `bool YMC`, `bool K`, `const char* ImagePath`, `int nValueThreshold` | 미사용 |
| `R600DrawImage` | `double dX`, `double dY`, `double dWidth`, `double dHeight`, `const char* szImgFilePath`, `int nSetNoAbsoluteBlack` | 사용 중 |
| `R600SetPrintRgnAndOffset` | `int xoffset`, `int yoffset`, `int cardSizeWidth`, `int cardSizeHeight`, `unsigned char isGlobal` | 미사용 |
| `R600SetYmcRgn` | `double x`, `double y`, `double width`, `double height`, `unsigned char isFront`, `unsigned char isMeansErase`, `unsigned char NotPrintMixBlack` | 미사용 |
| `R600SetKRgn` | `double x`, `double y`, `double width`, `double height`, `unsigned char isFront`, `unsigned char isMeansErase` | 미사용 |
| `R600SetCoatRgn` | `double x`, `double y`, `double width`, `double height`, `unsigned char isFront`, `unsigned char isMeansErase` | 미사용 |

### 특수 레이어 (화이트·UV·블랙·워터마크)

| 함수 | 인자 | 하나스튜디오 |
|---|---|---|
| `R600DrawWaterMark` | `double dX`, `double dY`, `double width`, `double height`, `const char* szImgFilePath` | 사용 중 |
| `R600SetWaterMarkParam` | `int rotation`, `unsigned char isNeedMirror` | 미사용 |
| `R600SetWaterMarkThreshold` | `int Threshold` | 미사용 |
| `R600DrawLayerWhite` | `double dX`, `double dY`, `double width`, `double height`, `const char* szImgFilePath` | 미사용 |
| `R600SetLayerWhiteThreshold` | `int Threshold` | 미사용 |
| `R600DrawLayerFluor` | `double dX`, `double dY`, `double width`, `double height`, `const char* szImgFilePath` | 미사용 |
| `R600SetLayerFluorThreshold` | `int Threshold` | 미사용 |
| `R600DrawLayerBlack` | `double dX`, `double dY`, `double width`, `double height`, `const char* szImgFilePath` | 미사용 |

### 텍스트·바코드·QR

| 함수 | 인자 | 하나스튜디오 |
|---|---|---|
| `R600DrawText` | `double x`, `double y`, `double width`, `double height`, `const wchar_t* szText`, `int nSetNoAbsoluteBlack` | 미사용 |
| `R600SetFont` | `const wchar_t* szFontName`, `float fSize` | 미사용 |
| `R600SetTextColor` | `int nR`, `int nG`, `int nB` | 미사용 |
| `R600SetTextDecorate` | `int nLandScape`, `int nReverseSequence`, `int nAutoLineFeed`, `int nLayDown` | 미사용 |
| `R600SetTextFontWeight` | `int nFontWeights` | 미사용 |
| `R600SetTextIsItalic` | `int nItalic` | 미사용 |
| `R600SetTextIsStrong` | `int nStrong` | 미사용 |
| `R600SetTextIsUnderline` | `int nUnderline` | 미사용 |
| `R600SetTextRotation` | `int nRotation` | 미사용 |
| `R600SetTextSpacing` | `double lineSpacing`, `double charSpacing` | 미사용 |
| `R600DrawBarCode` | `double dX`, `double dY`, `double dWidth`, `double dHeight`, `const char* szData`, `int nSetNoAbsoluteBlack` | 미사용 |
| `R600SetBarCodePara` | `int nSymbology`, `int nHumanRead`, `int nRotation` | 미사용 |
| `R600SetCodeColor` | `int nR`, `int nG`, `int nB` | 미사용 |
| `R600DrawQRCode` | `double dX`, `double dY`, `double dWidth`, `double dHeight`, `const char* szData`, `int nSetNoAbsoluteBlack` | 미사용 |
| `R600SetQRCodePara` | `int iErrorCorrectLevel` | 미사용 |

### 마그네틱·Mifare·접촉식 칩

| 함수 | 인자 | 하나스튜디오 |
|---|---|---|
| `R600SetMagCardOpt` | `unsigned char ucMode` | 미사용 |
| `R600MAGCardWrite` | `unsigned char ucTimeout`, `unsigned char* pTrack1Data`, `unsigned int pTrack1DataLen`, `unsigned char* pTrack2Data`, `unsigned int pTrack2DataLen`, `unsigned char* pTrack3Data`, `unsigned int pTrack3DataLen` | 미사용 |
| `R600MAGCardRead` | `unsigned char ucTimeout`, `unsigned char* pTrack1Data`, `unsigned int* pTrack1DataLen`, `unsigned char* pTrack2Data`, `unsigned int* pTrack2DataLen`, `unsigned char* pTrack3Data`, `unsigned int* pTrack3DataLen` | 미사용 |
| `R600M1CardGetSN` | `unsigned char ucMode`, `unsigned char* pSnrstr` | 미사용 |
| `R600M1CardPwdAuth` | `int nIsHexPwd`, `unsigned char ucPwdType`, `unsigned char ucAddr`, `const char* szPwd` | 미사용 |
| `R600M1CardRead` | `int nIsHexReading`, `unsigned char ucAdr`, `char* pData` | 미사용 |
| `R600M1CardWrite` | `int nIsHexWritting`, `unsigned char ucAdr`, `const char* pData` | 미사용 |
| `R600M1CardHalt` | 없음 | 미사용 |
| `R600CPUCardReset` | `int nSlotNum`, `char* pResetInfo`, `unsigned char* pResetInfolen`, `char* pUid`, `unsigned char* pUidLen` | 미사용 |
| `R600CPUCardExchangeAPDU` | `int nSlotNum`, `unsigned char* pSApdu`, `unsigned char ucApduLen`, `char* pRApdu`, `unsigned char* pRApduLen` | 미사용 |
현재 코드와 다른 점:

- `R600SetRibbonOpt`의 첫 인자는 리본 종류가 아니라 `isWrite`(쓰기 여부)다. `printer/r600_printer.py`의 `set_ribbon_option(ribbon_type=1, ...)`은 값은 맞지만 인자 이름이 실제 의미와 다르다.
- `R600DrawText`와 `R600SetFont`의 문자열은 `wchar_t`(UTF-16)다. 나머지 문자열 인자는 `char`다.

---

## 3. 카드프레소의 RTAI 호출 방식

### 3.1 인쇄 한 건의 순서

1. 리본 확인: `R600RibbonSettingsRW(0, &리본종류, &필름종류, &리본부족, &필름부족)`. 첫 인자 0은 읽기다. 리본 종류 1바이트를 보관해 뒤에서 쓴다.
2. K패널 분할 인쇄: `R600SetImgUtilIsNeedSet_K_ColorRibbonSplitPrint(flag)`. 리본이 지원하지 않거나 작업에 앞면과 뒷면이 함께 있지 않으면 `false`로 넘긴다.
3. `R600SetRibbonOpt(1, 0, "2", 2)`. 성공 로그가 "YMCS printing mode set as disabled"다. 하나스튜디오도 같은 값으로 호출하고 있다.
4. 면마다 캔버스 작업:
   1. `R600PrepareCanvas(0, 0)`
   2. 컬러(YMC) 레이어: `R600SetAddImageMode_Rtai(0, true, false, NULL, 220)` → `R600DrawImage(x, y, w, h, 경로, 1)`
   3. 검정(K) 레이어: `R600SetImagePara(1, 0, 0)` → `R600SetAddImageMode_Rtai(3, false, true, NULL, 220)` → `R600DrawImage(x, y, w, h, 경로, 0)`
   4. 특수 레이어(UV 또는 실버): 3.2 참고
   5. `R600CommitCanvas(버퍼, &길이)`. 버퍼 길이는 200이다.
5. `R600PrintDraw(앞면 캔버스 정보, 뒷면 캔버스 정보)`
6. 카드 배출

레이어 이미지는 종류별로 따로 렌더링한 파일이다. 하나스튜디오처럼 컬러 이미지 한 장과 마스크 한 장을 넘기는 구조와 같은 방식이다.

### 3.2 특수 레이어 (UV, 실버)

- 공통: `R600SetImagePara(1, 0, 0)`. 로그는 "Set white as transparent"다.
- 리본 종류 코드가 `0x18` 또는 `0x19`이면 실버로 처리한다.
  - `R600SetWaterMarkThreshold(임계값)`
  - `R600SetRibbonOpt(1, 1, "255", 4)`. 로그는 "Set watermark thickness to 255 for Silver layer"다.
  - `R600DrawWaterMark(x, y, w, h, 경로)`
- 그 외 리본은 UV로 처리한다.
  - `R600SetLayerFluorThreshold(임계값)`
  - `R600SetRibbonOpt(1, 1, "180", 4)`. 로그는 "Set watermark thickness to 180 for UV layer"다.
  - `R600DrawLayerFluor(x, y, w, h, 경로)`
- 그리기 직전에 `R600SetAddImageMode_Rtai(모드, false, false, 경로, 임계값)`을 부른다.
- 임계값 기본은 220이다.

확인하지 못한 부분:

- 임계값과 두께 설정은 모드가 3일 때만 실행된다. 인쇄 흐름에서 넘어오는 모드 값이 어떤 조건에서 3이 되는지는 끝까지 따라가지 못했다.
- `R600SetAddImageMode_Rtai`의 `nMode` 값 0과 3의 의미는 문서로 확인한 것이 아니다. 로그에 "halftone mode"라는 표현이 있어 이미지 이진화 방식으로 보인다.
- `0x18`, `0x19`가 어떤 리본 제품인지는 모른다.

### 3.3 `R600SetRibbonOpt`의 key

| key | 값 | 카드프레소 로그 |
|---|---|---|
| 0 | `"2"` | YMCS printing mode set as disabled |
| 1 | `"255"` (실버), `"180"` (UV) | Set watermark thickness to … |

다른 key는 카드프레소에서 쓰지 않는다.

### 3.4 마그네틱

1. 카드 이동: `R600CardMoveImpr(위치)`. 3.5 참고
2. `R600SetMagCardOpt(1)`. 실패 로그가 "Failed to Set Magnetic Coercivity to HIGH"이므로 1은 고보자력이다. 실패해도 무시하고 계속한다.
3. 쓰기: `R600MAGCardWrite(10, 트랙1, 길이1, 트랙2, 길이2, 트랙3, 길이3)`. 첫 인자 `ucTimeout`은 따로 지정하지 않으면 10이다.
4. 읽기: `R600MAGCardRead(10, 버퍼1, &길이1, 버퍼2, &길이2, 버퍼3, &길이3)`. 버퍼는 트랙마다 128바이트다.
5. 쓴 뒤 다시 읽어 검증한다. 공장 출하 상태의 빈 트랙 때문에 생기는 읽기 오류는 무시하는 예외 처리가 있다.

ISO 형식만 지원한다. 다른 형식을 요청하면 "Failed to write Tracks as it was no ISO"로 실패한다.

### 3.5 `R600CardMoveImpr`의 위치 번호

| 목적 | 3-in-1 인코더 | 그 외 |
|---|---|---|
| 마그네틱 | 5 | 6 |
| 스마트카드 | 3 또는 4 | 1 또는 2 |

스마트카드의 두 값은 접촉식과 비접촉식을 가르는 것으로 보이지만, 어느 쪽이 어느 값인지는 확인하지 못했다. 3-in-1 여부는 SDK 설정 파일의 `CardReader/Model`과 대응한다.

이동 전에 카드가 프린터 안에 있는지 확인하고, 없으면 먼저 투입한다. 이미 그 위치에 있으면 호출을 건너뛴다.

### 3.6 카드프레소가 쓰지 않는 것

- `R600OprRetryCtrl`: 주소는 불러오지만 분석 범위 안에서는 호출하지 않는다.
- Mifare(`R600M1Card*`), 접촉식 칩(`R600CPUCard*`): 불러오지 않는다. 카드프레소는 이 부분을 PC/SC 등 다른 경로로 처리한다.
- 화이트 레이어(`R600DrawLayerWhite`), 블랙 레이어(`R600DrawLayerBlack`), SDK 내장 텍스트·바코드·QR, 상태 조회(`R600QueryPrtStatus`), 리본 잔량(`R600GetRbnAndFilmRemaining`): 불러오지 않는다.

이 함수들은 하나스튜디오 DLL에 있으므로 쓸 수 있지만, 참고할 호출 예가 없어 실기 테스트로 값을 정해야 한다.

---

## 4. 하나스튜디오에 적용할 때

| 기능 | 필요한 함수 | 상태 |
|---|---|---|
| 프린터 상태·오류 문구 (0단계) | `R600QueryPrtStatus`, `R600GetErrorInfo`, `R600GetCheckStatusMsg` | 인자 확인됨. 호출 예 없음 |
| 리본·필름 잔량 (0단계) | `R600GetRbnAndFilmRemaining`, `R600RibbonSettingsRW` | 인자 확인됨. 읽기 모드 0은 카드프레소에서 확인 |
| 인코더 구성 확인 | `R600GetPrtOptionStatus` | 인자 확인됨. 호출 예 없음 |
| UV 레이어 (5단계) | `R600SetLayerFluorThreshold`, `R600DrawLayerFluor`, `R600SetRibbonOpt` | 호출 순서와 값 확인됨 (3.2) |
| K패널 분할 (5단계) | `R600SetImgUtilIsNeedSet_K_ColorRibbonSplitPrint` | 인자 확인됨 |
| 마그네틱 (5단계) | `R600CardMoveImpr`, `R600SetMagCardOpt`, `R600MAGCardWrite`, `R600MAGCardRead` | 호출 순서와 값 확인됨 (3.4) |
| Mifare (5단계) | `R600M1CardGetSN`, `R600M1CardPwdAuth`, `R600M1CardRead`, `R600M1CardWrite`, `R600M1CardHalt` | 인자만 확인됨. 값은 실기 테스트 필요 |
| 접촉식 칩 | `R600CPUCardReset`, `R600CPUCardExchangeAPDU` | 인자만 확인됨 |

실기 테스트로 먼저 정해야 할 것:

1. `R600QueryPrtStatus`가 돌려주는 상태 값과 `dll/R600StatusReference`의 대응. 이 파일은 평문이 아니라서 SDK가 직접 읽는 것으로 보인다. 문구는 `R600GetErrorInfo`나 `R600GetCheckStatusMsg`로 받는 쪽이 안전하다.
2. 보유 프린터의 리본 종류 코드와 UV 지원 여부.
3. `R600CardMoveImpr` 위치 번호와 보유 인코더 구성.
