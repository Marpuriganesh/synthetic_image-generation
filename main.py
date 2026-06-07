import os
import random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from src.font_collection import get_system_fonts
from fontTools.ttLib import TTFont
import pandas as pd
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

default_output_path = os.path.join(os.getcwd(), "output")
CHARS = r"""abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

# Fixed: Initialize as a list to prevent data overwriting on every iteration
train_output_data = []
test_output_data = []
# ==========================================
# TEXT GENERATION ENGINE
# ==========================================

def load_dictionary_words() -> list:
    """Attempts to load local system words, with a clean backup vocabulary."""
    system_dict = Path("/usr/share/dict/words")
    if system_dict.exists():
        with open(system_dict, "r", encoding="utf-8") as f:
            return [w.strip() for w in f.readlines() if 3 <= len(w.strip()) <= 12 and w.isascii()]
    
    return [
        "account", "balance", "invoice", "payment", "receipt", "statement", "amount", "total",
        "customer", "service", "business", "company", "system", "program", "developer", "language",
        "document", "processing", "network", "training", "matrix", "vector", "optical", "character"
    ]

WORD_POOL = load_dictionary_words()

def generate_text_snippet() -> str:
    """Randomly decides to generate a single word, a full sentence, or gibberish."""
    choice = random.choice(["word", "sentence", "gibberish"])
    
    if choice == "word":
        return random.choice(WORD_POOL)
        
    elif choice == "sentence":
        num_words = random.randint(3, 7)
        sentence_words = [random.choice(WORD_POOL) for _ in range(num_words)]
        sentence = " ".join(sentence_words).capitalize()
        return sentence + random.choice([".", "!", "?", ""])
        
    else:  
        length = random.randint(4, 15)
        return "".join(random.choice(CHARS) for _ in range(length))


# ==========================================
# COLOR & ACCESSIBILITY UTILITIES
# ==========================================

def calculate_brightness(rgb_tuple):
    R, G, B = rgb_tuple
    return (0.299 * R) + (0.587 * G) + (0.114 * B)

def generate_contrasting_colors():
    while True:
        bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        font_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        bg_brightness = calculate_brightness(bg_color)
        font_brightness = calculate_brightness(font_color)
        
        if abs(bg_brightness - font_brightness) > 100:
            return bg_color, font_color


def font_has_latin_support(font_path: str) -> bool:
    try:
        font = TTFont(font_path)
        cmap = font.getBestCmap()
        font.close()

        for c in CHARS:
            if cmap and ord(c) not in cmap:
                return False
                
        return True
    except Exception:
        return False

# ==========================================
# CORE AUGMENTATION FUNCTIONS (OpenCV/NumPy)
# ==========================================

def apply_ink_effects(
    pil_img: Image.Image, 
    mode: str = "bleed", 
    kernel_size: int = 2, 
    iterations: int = 1,
    blend_ratio: float = 1.0
) -> Image.Image:
    orig_np = np.array(pil_img)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if mode == "bleed":
        morphed_np = cv2.erode(orig_np, kernel, iterations=iterations)
    elif mode == "fade":
        morphed_np = cv2.dilate(orig_np, kernel, iterations=iterations)
    else:
        return pil_img

    if blend_ratio < 1.0:
        blended_np = cv2.addWeighted(
            src1=orig_np, 
            alpha=1.0 - blend_ratio, 
            src2=morphed_np, 
            beta=blend_ratio, 
            gamma=0
        )
        return Image.fromarray(blended_np)
        
    return Image.fromarray(morphed_np)


def apply_salt_and_pepper_noise(pil_img: Image.Image, amount: float = 0.02) -> Image.Image:
    img_np = np.array(pil_img)
    rows, cols, channels = img_np.shape
    
    num_pepper = int(img_np.size * amount / 2)
    coords_y = [random.randint(0, rows - 1) for _ in range(num_pepper)]
    coords_x = [random.randint(0, cols - 1) for _ in range(num_pepper)]
    img_np[coords_y, coords_x] = 0  
    
    num_salt = int(img_np.size * amount / 2)
    coords_y = [random.randint(0, rows - 1) for _ in range(num_salt)]
    coords_x = [random.randint(0, cols - 1) for _ in range(num_salt)]
    img_np[coords_y, coords_x] = 255  
    
    return Image.fromarray(img_np)


def apply_wave_distortion(pil_img: Image.Image, amplitude: float = 4.0, frequency: float = 0.05) -> Image.Image:
    img_np = np.array(pil_img)
    rows, cols = img_np.shape[:2]
    
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
    map_x = map_x + amplitude * np.sin(map_y * frequency)
    
    distorted_img = cv2.remap(
        img_np, 
        map_x.astype(np.float32), 
        map_y.astype(np.float32), 
        interpolation=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(255, 255, 255)
    )
    
    return Image.fromarray(distorted_img)


def apply_blur(pil_img: Image.Image, kernel_size: int = 3) -> Image.Image:
    img_np = np.array(pil_img)
    if kernel_size % 2 == 0:
        kernel_size += 1
    img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
    return Image.fromarray(img_np)

# ==========================================
# RENDERING PIPELINE & MAIN LOOP
# ==========================================

def get_text_dimensions(text: str, font_object: ImageFont.FreeTypeFont):
    bbox = font_object.getbbox(text)
    left, top, right, bottom = bbox
    exact_width = right - left
    exact_height = bottom - top
    return int(exact_width), int(exact_height), int(left), int(top)


def generate_single_text_image(text: str, font_path: str, output_path: str = default_output_path, mode:str = "train"):
    try:
        font = ImageFont.truetype(font_path, size=32)
    except Exception as e:
        print(f"Failed to load font: {e}")
        return

    e_w, e_h, left, top = get_text_dimensions(text, font)
    if e_w <= 0 or e_h <= 0:
        return

    padding = 20
    canvas_w = e_w + (padding * 2)
    canvas_h = e_h + (padding * 2)
    bg_color, text_color = generate_contrasting_colors()
    
    canvas = Image.new(mode="RGB", size=(canvas_w, canvas_h), color=bg_color)
    draw_tool = ImageDraw.Draw(canvas)

    draw_tool.text(font=font, text=text, fill=text_color, xy=(padding - left, padding - top))

    # Augmentations
    canvas = apply_wave_distortion(canvas, amplitude=random.uniform(2.5, 4.5), frequency=random.uniform(0.01, 0.1))
    canvas = apply_ink_effects(canvas, mode=random.choice(["bleed", "fade"]), blend_ratio=random.uniform(0.1, 1.0))
    canvas = apply_salt_and_pepper_noise(canvas, amount=0.005)
    canvas = apply_blur(canvas, kernel_size=3)

    stripped_font_name = Path(font_path).stem
    os.makedirs(os.path.join(output_path,mode), exist_ok=True)
    
    unique_id = random.randint(10000, 99999)
    output_file_path = os.path.join(output_path,mode, f"{stripped_font_name}_{unique_id}.png")
    relative_file_path = os.path.relpath(output_file_path, output_path)
    
    canvas.save(output_file_path)
    
    # Fixed: Append a new dictionary payload to the collection list
    if mode == 'train':
      train_output_data.append({
          "path": relative_file_path,
          "text": text
      })
    else:
      test_output_data.append({
          "path": relative_file_path,
          "text": text
      })
    


def generate(iterations: int = 100):
    fonts_paths = get_system_fonts()
    
    # 1. Pre-filter compatible fonts so we know the absolute maximum step size
    print("[System] Validating character maps across fonts...")
    valid_fonts = [str(p) for p in fonts_paths if font_has_latin_support(str(p))]
    
    if not valid_fonts:
        print("Error: No system fonts found that support the full character set layout.")
        return

    total_steps = iterations * len(valid_fonts)

    # 2. Configure and run the rich progress engine
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bright_green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        
        generation_task = progress.add_task(
            description=f"[cyan]Generating {total_steps} OCR tokens...", 
            total=total_steps
        )

        for _ in range(iterations):
            for path in valid_fonts:
                random_text = generate_text_snippet()
                generate_single_text_image(text=random_text, font_path=path, mode=random.choices(["train", "test"],weights=[80,20])[0])
                
                # Advance the bar forward by 1 step
                progress.update(generation_task, advance=1)

    # 3. Build and export file log metadata
    print("\n[System] Compilation finished. Formatting CSV log...")
    train_output_dataframe = pd.DataFrame(train_output_data)
    test_output_dataframe = pd.DataFrame(test_output_data)
    train_csv_output_path = os.path.join(default_output_path, "train.csv")
    test_csv_output_path = os.path.join(default_output_path, "test.csv")
    train_output_dataframe.to_csv(train_csv_output_path, index=False)
    test_output_dataframe.to_csv(test_csv_output_path, index=False)
    print(f"[Success] Saved {len(train_output_dataframe)} records to {train_csv_output_path}")
    print(f"[Success] Saved {len(test_output_dataframe)} records to {test_csv_output_path}")


def main():
    generate(500)


if __name__ == "__main__":
    main()