"""
스플래시 이미지 생성 스크립트
PyInstaller Splash용 PNG 이미지 생성
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_splash_image():
    """스플래시 이미지 생성"""
    # 300x200 크기의 이미지
    width, height = 400, 250
    img = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 배경 그라데이션 효과 (간단한 흰색 배경)
    draw.rectangle([0, 0, width, height], fill=(248, 249, 250, 255))

    # 테두리
    draw.rectangle([0, 0, width-1, height-1], outline=(74, 144, 226, 255), width=2)

    # 로고 원
    circle_x, circle_y = width // 2, 80
    circle_radius = 40
    draw.ellipse(
        [circle_x - circle_radius, circle_y - circle_radius,
         circle_x + circle_radius, circle_y + circle_radius],
        fill=(74, 144, 226, 255)
    )

    # H 텍스트
    try:
        font_h = ImageFont.truetype("arial.ttf", 48)
    except:
        font_h = ImageFont.load_default()

    draw.text((circle_x, circle_y), "H", fill=(255, 255, 255, 255),
              font=font_h, anchor="mm")

    # 타이틀
    try:
        font_title = ImageFont.truetype("arial.ttf", 20)
    except:
        font_title = ImageFont.load_default()

    draw.text((width // 2, 150), "Hana Studio", fill=(52, 58, 64, 255),
              font=font_title, anchor="mm")

    # 로딩 텍스트
    try:
        font_loading = ImageFont.truetype("arial.ttf", 12)
    except:
        font_loading = ImageFont.load_default()

    draw.text((width // 2, 190), "Loading...", fill=(108, 117, 125, 255),
              font=font_loading, anchor="mm")

    # 프로그레스바 배경
    bar_width = 200
    bar_height = 6
    bar_x = (width - bar_width) // 2
    bar_y = 210
    draw.rectangle([bar_x, bar_y, bar_x + bar_width, bar_y + bar_height],
                   fill=(233, 236, 239, 255))

    # 저장
    output_path = "splash.png"
    img.save(output_path, "PNG")
    print(f"스플래시 이미지 생성 완료: {output_path}")
    return output_path

if __name__ == "__main__":
    create_splash_image()
