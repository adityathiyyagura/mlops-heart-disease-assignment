from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
WIDTH = 1920
HEIGHT = 1080
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def build_intro():
    image = Image.new("RGB", (WIDTH, HEIGHT), "#101a22")
    draw = ImageDraw.Draw(image)

    for x in range(1030, WIDTH, 56):
        draw.line((x, 0, x, HEIGHT), fill="#17252e", width=1)
    for y in range(0, HEIGHT, 56):
        draw.line((1030, y, WIDTH, y), fill="#17252e", width=1)

    regular = ImageFont.truetype(FONT, 31)
    label = ImageFont.truetype(FONT, 24)
    title = ImageFont.truetype(FONT, 112)
    subtitle = ImageFont.truetype(FONT, 37)
    node_title = ImageFont.truetype(FONT, 24)
    node_detail = ImageFont.truetype(FONT, 17)
    count = ImageFont.truetype(FONT, 22)
    footer = ImageFont.truetype(FONT, 19)

    white = "#f4f1e8"
    muted = "#b3c1c5"
    teal = "#38c5b1"
    coral = "#ef8d70"

    draw.rounded_rectangle((128, 101, 182, 109), radius=4, fill=teal)
    draw.text((128, 137), "MACHINE LEARNING OPERATIONS  /  ASSIGNMENT 01", font=label, fill=teal)
    draw.text((120, 274), "Heart Disease", font=title, fill=white, stroke_width=2, stroke_fill=white)
    draw.text((120, 402), "Prediction", font=title, fill=white, stroke_width=2, stroke_fill=white)
    draw.line((126, 575, 844, 575), fill=coral, width=4)
    draw.text((126, 614), "From UCI data to a tested, monitored local service.", font=subtitle, fill=muted)

    draw.text((126, 785), "CLEAN MODELING DATA", font=label, fill=teal)
    draw.text((126, 833), "297", font=ImageFont.truetype(FONT, 66), fill=white)
    draw.text((260, 861), "complete records", font=count, fill=muted)
    draw.text((126, 920), "160", font=count, fill=white)
    draw.rounded_rectangle((185, 929, 455, 949), radius=8, fill="#263740")
    draw.rounded_rectangle((185, 929, 328, 949), radius=8, fill=teal)
    draw.text((480, 920), "137", font=count, fill=white)
    draw.rounded_rectangle((540, 929, 810, 949), radius=8, fill="#263740")
    draw.rounded_rectangle((540, 929, 663, 949), radius=8, fill=coral)
    draw.text((185, 963), "No disease", font=label, fill=muted)
    draw.text((540, 963), "Disease", font=label, fill=muted)

    nodes = [
        (1090, "DATA", "UCI · EDA"),
        (1285, "TRAIN", "5-fold CV"),
        (1480, "SERVE", "FastAPI"),
        (1675, "OBSERVE", "Prometheus"),
    ]
    node_y = 450
    node_w = 158
    node_h = 104
    for index, (x, heading, detail) in enumerate(nodes):
        edge = teal if index in (0, 3) else coral
        draw.rounded_rectangle((x, node_y, x + node_w, node_y + node_h), radius=8, outline="#536a72", width=2, fill="#14232b")
        draw.rounded_rectangle((x, node_y, x + node_w, node_y + 7), radius=3, fill=edge)
        heading_box = draw.textbbox((0, 0), heading, font=node_title)
        draw.text((x + (node_w - heading_box[2]) / 2, node_y + 28), heading, font=node_title, fill=white)
        detail_box = draw.textbbox((0, 0), detail, font=node_detail)
        draw.text((x + (node_w - detail_box[2]) / 2, node_y + 68), detail, font=node_detail, fill=muted)
        if index < len(nodes) - 1:
            arrow_x = x + node_w + 9
            arrow_y = node_y + node_h / 2
            draw.line((arrow_x, arrow_y, arrow_x + 17, arrow_y), fill=teal, width=3)
            draw.polygon([(arrow_x + 17, arrow_y), (arrow_x + 10, arrow_y - 6), (arrow_x + 10, arrow_y + 6)], fill=teal)

    draw.text((1090, 604), "CV-selected model  ·  Logistic Regression + Random Forest", font=label, fill=muted)
    draw.line((128, 1030, 1792, 1030), fill="#34464f", width=2)
    draw.text((128, 1044), "UCI HEART DISEASE  ·  REPRODUCIBLE MLOPS CASE STUDY", font=footer, fill=muted)
    draw.text((1534, 1044), "LOCAL DEMO  /  NOT A CLINICAL TOOL", font=footer, fill=muted)

    image_path = BASE / "video_intro.png"
    video_path = BASE / "video_intro.mp4"
    image.save(image_path, optimize=True)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-loop", "1", "-framerate", "30", "-i", str(image_path),
        "-t", "4", "-r", "30", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(video_path),
    ], check=True)
    print(f"Created {image_path}")
    print(f"Created {video_path}")


if __name__ == "__main__":
    build_intro()