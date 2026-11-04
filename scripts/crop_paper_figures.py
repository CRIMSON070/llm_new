import os
import sys
import fitz

sys.stdout.reconfigure(encoding='utf-8')

out_dir = r"outputs/Group_NUMBER/figures"
os.makedirs(out_dir, exist_ok=True)

doc = fitz.open("2504.14493v4.pdf")
dpi = 300
zoom = dpi / 72.0
mat = fitz.Matrix(zoom, zoom)

# 1. Crop Table 1 from Page 7 (0-indexed page 6)
# Coordinates: x0=50, y0=75, x1=562, y1=265
p7 = doc[6]
rect_t1 = fitz.Rect(50, 75, 562, 265)
pix_t1 = p7.get_pixmap(matrix=mat, clip=rect_t1)
path_t1 = os.path.join(out_dir, "paper_table1_crop.png")
pix_t1.save(path_t1)
print(f"Saved Table 1 crop: {path_t1} ({pix_t1.width}x{pix_t1.height})")

# 2. Crop Figure 5 from Page 8 (0-indexed page 7)
# Coordinates: x0=50, y0=75, x1=562, y1=265
p8 = doc[7]
rect_f5 = fitz.Rect(50, 75, 562, 265)
pix_f5 = p8.get_pixmap(matrix=mat, clip=rect_f5)
path_f5 = os.path.join(out_dir, "paper_figure5_crop.png")
pix_f5.save(path_f5)
print(f"Saved Figure 5 crop: {path_f5} ({pix_f5.width}x{pix_f5.height})")

# 3. Crop Table 2 from Page 8 (0-indexed page 7)
# Table 2 is in the left column: x0=50, y0=535, x1=305, y1=675
rect_t2 = fitz.Rect(50, 535, 305, 675)
pix_t2 = p8.get_pixmap(matrix=mat, clip=rect_t2)
path_t2 = os.path.join(out_dir, "paper_table2_crop.png")
pix_t2.save(path_t2)
print(f"Saved Table 2 crop: {path_t2} ({pix_t2.width}x{pix_t2.height})")

# 4. Crop Table 7 from Page 16 (0-indexed page 15) for optional appendix / extra details
# Table 7 on Page 16 spans across the page: x0=50, y0=75, x1=562, y1=320
p16 = doc[15]
rect_t7 = fitz.Rect(50, 75, 562, 320)
pix_t7 = p16.get_pixmap(matrix=mat, clip=rect_t7)
path_t7 = os.path.join(out_dir, "paper_table7_crop.png")
pix_t7.save(path_t7)
print(f"Saved Table 7 crop: {path_t7} ({pix_t7.width}x{pix_t7.height})")

print("All paper screenshots cropped successfully at 300 DPI.")
