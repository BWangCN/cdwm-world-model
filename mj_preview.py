"""Combined photoreal preview montage from the MuJoCo episode gifs (rows = objects, cols = phases)."""
from PIL import Image, ImageSequence, ImageDraw
OBJS = ["006_mustard_bottle", "021_bleach_cleanser", "022_windex_bottle", "048_hammer"]
COLS = [("grasp", 26), ("lift", 32), ("transport", 40), ("placed", 47)]
C = 330
rows = []
for o in OBJS:
    fr = [f.copy().convert("RGB") for f in ImageSequence.Iterator(Image.open(f"figures/mj_{o}.gif"))]
    strip = Image.new("RGB", (C * len(COLS), C), "black")
    for j, (nm, fi) in enumerate(COLS):
        strip.paste(fr[min(fi, len(fr) - 1)].resize((C, C)), (j * C, 0))
    d = ImageDraw.Draw(strip); d.rectangle([0, 0, 210, 20], fill="black")
    d.text((5, 5), " ".join(w for w in o.split("_") if not w[0].isdigit()), fill=(255, 230, 80))
    rows.append(strip)
sheet = Image.new("RGB", (C * len(COLS), C * len(rows)), "black")
for i, r in enumerate(rows): sheet.paste(r, (0, i * C))
sheet.save("figures/mj_preview.png"); print("wrote figures/mj_preview.png", sheet.size)
