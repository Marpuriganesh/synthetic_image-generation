from src.font_collection import get_system_fonts

for path in get_system_fonts():
    print(path)