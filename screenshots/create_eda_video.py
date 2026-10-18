from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent
WIDTH = 1920
HEIGHT = 1080
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"

SLIDES = [
    (
        "artifacts/eda/class_balance.png",
        "01 / 03     DATASET EXPLORATION",
        "Class balance",
        "Target-valid source records before complete-case filtering",
        "Counts: 164 no disease  /  139 disease",
    ),
    (
        "artifacts/eda/age_distribution.png",
        "02 / 03     DATASET EXPLORATION",
        "Age distribution",
        "The source cohort spans adult and older patients",
        "EDA describes the source cohort, not the held-out test split",
    ),
    (
        "artifacts/eda/correlation_heatmap.png",
        "03 / 03     DATASET EXPLORATION",
        "Feature correlation",
        "Pairwise correlations across the input features and target",
        "Descriptive relationships only; correlation does not establish causation",
    ),
]


def build_slide(source, index, eyebrow, title, subtitle, footer):
    image = Image.new("RGB", (WIDTH, HEIGHT), "#101a22")
    draw = ImageDraw.Draw(image)

    eyebrow_font = ImageFont.truetype(FONT, 23)
    title_font = ImageFont.truetype(FONT, 52)
    subtitle_font = ImageFont.truetype(FONT, 26)
    footer_font = ImageFont.truetype(FONT, 21)

    draw.rounded_rectangle((118, 57, 162, 64), radius=3, fill="#38c5b1")
    draw.text((118, 82), eyebrow, font=eyebrow_font, fill="#38c5b1")
    draw.text((118, 115), title, font=title_font, fill="#f4f1e8")
    draw.text((120, 184), subtitle, font=subtitle_font, fill="#b3c1c5")

    source_path = BASE / source
    with Image.open(source_path) as source_image:
        plot = source_image.convert("RGB")
    plot.thumbnail((1680, 750), Image.Resampling.LANCZOS)

    panel_margin = 14
    panel_x = (WIDTH - plot.width) // 2
    panel_y = 236 + (770 - plot.height) // 2
    draw.rounded_rectangle(
        (
            panel_x - panel_margin,
            panel_y - panel_margin,
            panel_x + plot.width + panel_margin,
            panel_y + plot.height + panel_margin,
        ),
        radius=5,
        fill="#f8f8f7",
    )
    image.paste(plot, (panel_x, panel_y))

    draw.line((118, 1030, WIDTH - 118, 1030), fill="#34464f", width=2)
    draw.text((120, 1045), footer, font=footer_font, fill="#b3c1c5")
    draw.text((1688, 1045), f"{index:02d} / 03", font=footer_font, fill="#38c5b1")

    slide_path = OUTPUT_DIR / f"eda_slide_{index:02d}.png"
    image.save(slide_path, optimize=True)
    return slide_path


def build_video():
    slides = [
        build_slide(source, index, eyebrow, title, subtitle, footer)
        for index, (source, eyebrow, title, subtitle, footer) in enumerate(SLIDES, start=1)
    ]
    video_path = OUTPUT_DIR / "video_eda.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for slide_path in slides:
        command.extend(["-loop", "1", "-framerate", "30", "-t", "4", "-i", str(slide_path)])
    command.extend([
        "-filter_complex",
        "[0:v]trim=duration=4,setpts=PTS-STARTPTS,format=yuv420p[v0];"
        "[1:v]trim=duration=4,setpts=PTS-STARTPTS,format=yuv420p[v1];"
        "[2:v]trim=duration=4,setpts=PTS-STARTPTS,format=yuv420p[v2];"
        "[v0][v1][v2]concat=n=3:v=1:a=0[outv]",
        "-map", "[outv]", "-r", "30", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path),
    ])
    subprocess.run(command, check=True)
    print(f"Created {video_path}")
    for slide_path in slides:
        print(f"Created {slide_path}")


if __name__ == "__main__":
    build_video()