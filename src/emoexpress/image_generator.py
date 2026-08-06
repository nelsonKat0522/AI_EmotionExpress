from datetime import datetime
from pathlib import Path
import base64
import re
from typing import Any

from openai import OpenAI
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from emoexpress.config import (GENERATED_IMAGE_DIR,OPENAI_API_KEY)


IMAGE_MODEL = "gpt-image-2"
IMAGE_SIZE = "1024x1024"
IMAGE_QUALITY = "medium"

client = OpenAI(api_key=OPENAI_API_KEY)


def prepare_image_prompt(image_prompt: str,) -> str:
    """Add consistent visual and safety instructions."""

    if not isinstance(image_prompt, str):
        raise TypeError("image_prompt must be a string.")

    image_prompt = image_prompt.strip()

    if not image_prompt:
        raise ValueError("image_prompt cannot be empty.")

    return f"""
MAIN SCENE

{image_prompt}

ADDITIONAL REQUIREMENTS

Create a polished and emotionally supportive digital illustration.

- Show symbolic emotional progress and realistic hope.
- Avoid depicting a specific identifiable person.
- Use a calm and balanced composition.
- Do not include text, captions, logos, watermarks, or interface elements.
- Do not depict self-harm, violence, medication, medical treatment,
  or graphic distress.
- Do not suggest that every problem has been immediately resolved.
""".strip()


def create_safe_filename(caption: str) -> str:
    """Create a safe timestamped PNG filename."""

    safe_caption = re.sub(r"[^a-zA-Z0-9]+","_",caption.lower()).strip("_")

    if not safe_caption:
        safe_caption = "emoexpress_image"

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    return (f"{safe_caption[:50]}_{timestamp}.png")


def load_caption_font(font_size: int) -> Any:
    """Load a readable font with portable fallbacks."""

    font_candidates = [
        Path(
            "C:/Windows/Fonts/arialbd.ttf"
        ),
        Path(
            "C:/Windows/Fonts/arial.ttf"
        ),
        Path("/usr/share/fonts/truetype/"
            "dejavu/DejaVuSans-Bold.ttf")]

    for font_path in font_candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path),size=font_size)

    return ImageFont.load_default()


def wrap_caption_text(draw: ImageDraw.ImageDraw,caption: str,font: Any,maximum_width: int) -> str:
    """Wrap the caption according to rendered pixel width."""

    words = caption.split()

    if not words:
        return ""

    lines = []
    current_line = words[0]

    for word in words[1:]:
        candidate = (f"{current_line} {word}")

        text_box = draw.textbbox((0, 0),candidate,font=font)

        text_width = (text_box[2] - text_box[0])

        if text_width <= maximum_width:
            current_line = candidate
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)

    return "\n".join(lines)


def add_caption_to_image(image_path: str | Path,caption: str,output_path: str | Path | None = None) -> str:
    """Overlay the exact caption onto the generated image."""

    caption = caption.strip()

    if not caption:
        raise ValueError("caption cannot be empty.")

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Generated image not found: {image_path}")

    if output_path is None:
        output_path = image_path.with_name(f"{image_path.stem}_""with_caption.png")
    else:
        output_path = Path(output_path)

    image = PILImage.open(image_path).convert("RGBA")

    image_width, image_height = image.size

    font_size = max(28,int(image_width * 0.045))

    font = load_caption_font(font_size)

    overlay = PILImage.new("RGBA",image.size,(0, 0, 0, 0))

    draw = ImageDraw.Draw(overlay)

    horizontal_padding = int(image_width * 0.08)

    maximum_text_width = (image_width - 2 * horizontal_padding)

    wrapped_caption = wrap_caption_text(draw=draw,caption=caption,font=font,maximum_width=maximum_text_width)

    text_box = draw.multiline_textbbox((0, 0),wrapped_caption,font=font,spacing=10,align="center",stroke_width=2)

    text_width = (text_box[2]- text_box[0])

    text_height = (text_box[3]- text_box[1])

    vertical_padding = max(24,int(image_height * 0.025))

    banner_height = (text_height+ 2 * vertical_padding)

    banner_top = (image_height- banner_height- int(image_height * 0.035))

    banner_left = int(image_width * 0.045)

    banner_right = (image_width- banner_left)

    banner_bottom = (banner_top+ banner_height)

    draw.rounded_rectangle([banner_left,banner_top,banner_right,banner_bottom,],radius=28,fill=(0, 0, 0, 175))

    text_x = (image_width- text_width) / 2

    text_y = (banner_top+ vertical_padding- text_box[1])

    draw.multiline_text((text_x, text_y),
                    wrapped_caption,
                    font=font,
                    fill=(255, 255, 255, 255),
                    spacing=10,
                    align="center",
                    stroke_width=2,
                    stroke_fill=(0, 0, 0, 220))

    final_image = PILImage.alpha_composite(image,overlay).convert("RGB")

    final_image.save(output_path,format="PNG")

    return str(output_path)


def generate_image(image_prompt: str,caption: str) -> dict:
    """Generate an image and embed its caption."""

    final_prompt = prepare_image_prompt(image_prompt)

    response = client.images.generate(model=IMAGE_MODEL,
                                    prompt=final_prompt,
                                    size=IMAGE_SIZE,
                                    quality=IMAGE_QUALITY,
                                    n=1)

    if not response.data:
        raise RuntimeError("The image API returned no image.")

    encoded_image = (response.data[0].b64_json)

    if not encoded_image:
        raise RuntimeError("The image API did not return Base64 data.")

    image_bytes = base64.b64decode(encoded_image)

    original_image_path = (GENERATED_IMAGE_DIR/ create_safe_filename(caption))

    with open(original_image_path,"wb") as file:
        file.write(image_bytes)

    final_image_path = add_caption_to_image(image_path=original_image_path,caption=caption)

    return {"status": "success",
        "path": final_image_path,
        "original_path": str(original_image_path),
        "final_path": final_image_path,
        "caption": caption,
        "caption_embedded": True,
        "model": IMAGE_MODEL,
        "size": IMAGE_SIZE,
        "quality": IMAGE_QUALITY,
        "prompt": final_prompt,
        "error": None}


def generate_image_safely(image_prompt: str,caption: str) -> dict:
    """Generate the image without breaking the full pipeline."""

    try:
        return generate_image(image_prompt=image_prompt,caption=caption)

    except Exception as error:
        return {"status": "failed",
            "path": None,
            "original_path": None,
            "final_path": None,
            "caption": caption,
            "caption_embedded": False,
            "model": IMAGE_MODEL,
            "size": IMAGE_SIZE,
            "quality": IMAGE_QUALITY,
            "prompt": image_prompt,
            "error": str(error)}