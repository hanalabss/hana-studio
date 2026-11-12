"""printer 테스트 공용 도우미."""

IMG = "<img>"


def simplify(calls):
    """기록된 호출에서 임시 이미지 경로만 <img>로 바꿔 비교하기 쉽게 만든다."""
    result = []
    for name, args in calls:
        cleaned = []
        for arg in args:
            if isinstance(arg, bytes) and arg.lower().endswith((b".jpg", b".png", b".bmp")):
                cleaned.append(IMG)
            elif isinstance(arg, str) and arg.lower().endswith((".jpg", ".png", ".bmp")):
                cleaned.append(IMG)
            else:
                cleaned.append(arg)
        result.append((name, tuple(cleaned)))
    return result
