from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

df = pd.read_csv("splits/random_test.csv")
clear = df[df.label=="clear"].head(3)["path"].tolist()
obst  = df[df.label=="obstructed"].head(3)["path"].tolist()
paths = clear + obst

fig, axes = plt.subplots(2, 3, figsize=(10,6))
for ax, p in zip(axes.ravel(), paths):
    img = Image.open(p).convert("RGB")
    ax.imshow(img); ax.set_title(Path(p).name[:28]); ax.axis("off")
Path("figures").mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig("figures/sample_grid.png", dpi=200)
print("saved figures/sample_grid.png")
