from pathlib import Path

# 1. Define the standard Mac font locations
font_dirs = ["/System/Library/Fonts", "/Library/Fonts"]

def get_system_fonts() -> list[Path]:
    all_font_paths = []
    # 2. Loop through directories and gather both .ttf and .otf files
    for dir_path in font_dirs:
        path_obj = Path(dir_path)
        if path_obj.exists():
            # Gather TrueType and OpenType fonts recursively
            all_font_paths.extend(list(path_obj.rglob("*.ttf")))
            all_font_paths.extend(list(path_obj.rglob("*.otf")))
    return all_font_paths

