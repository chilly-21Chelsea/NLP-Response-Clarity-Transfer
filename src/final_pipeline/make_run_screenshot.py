from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from final_pipeline.common import PROJECT_ROOT


def load_tail(path: Path, line_count: int = 18) -> list[str]:
    lines = [
        line.rstrip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    return lines[-line_count:]


def main() -> None:
    logs = [
        PROJECT_ROOT / "logs" / "step2_qevasion.log",
        PROJECT_ROOT / "logs" / "step3_qnli_transfer.log",
    ]
    body: list[str] = []
    for path in logs:
        body.append(f"PS> Get-Content {path.name} -Tail 18")
        body.extend(
            textwrap.shorten(line, width=150, placeholder=" ...")
            for line in load_tail(path)
        )
        body.append("")

    width, height = 1800, 1180
    image = Image.new("RGB", (width, height), "#0C0C0C")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 25)
    mono_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 21)

    draw.rectangle((0, 0, width, 58), fill="#202020")
    draw.ellipse((22, 18, 44, 40), fill="#FF5F56")
    draw.ellipse((52, 18, 74, 40), fill="#FFBD2E")
    draw.ellipse((82, 18, 104, 40), fill="#27C93F")
    draw.text(
        (126, 13),
        "PowerShell - NLP final experiment run",
        font=title_font,
        fill="#EDEDED",
    )

    y = 82
    for line in body:
        color = "#D4D4D4"
        if "PS>" in line:
            color = "#75BEFF"
        elif "best QEvasion" in line or "Transfer comparison" in line:
            color = "#B5CEA8"
        wrapped = textwrap.wrap(line, width=104) or [""]
        for part in wrapped:
            draw.text((28, y), part, font=mono_font, fill=color)
            y += 29
        if y > height - 70:
            break

    output = PROJECT_ROOT / "document_work" / "screenshots" / "run_result_screenshot.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=95)
    print(output)


if __name__ == "__main__":
    main()
