# view_misclassified.py - find which images the model got wrong
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd, torch
from train_baseline import CSVImageDataset, eval_tf, make_model
from torch.utils.data import DataLoader

CSV_PATH = "splits/loso_target/test.csv"
CKPT = "best_loso_target.pt"

ds = CSVImageDataset(CSV_PATH, eval_tf)
dl = DataLoader(ds, batch_size=1, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
m = make_model().to(device)
m.load_state_dict(torch.load(CKPT, map_location=device))
m.eval()

cls_map = {0: "clear", 1: "obstructed"}

results = []
with torch.no_grad():
    for i, (x, y) in enumerate(dl):
        x = x.to(device)
        pred = m(x).argmax(1).item()
        true = y.item()
        img_path = ds.df.iloc[i]["path"]
        if pred != true:
            results.append((img_path, cls_map[true], cls_map[pred]))

print(f"Total misclassified: {len(results)}")
for p, t, pr in results:
    print(f"TRUE={t:12s}  PRED={pr:12s}  FILE={p}")

pd.DataFrame(results, columns=["path","true","pred"]).to_csv("misclassified.csv", index=False)
print("Saved misclassified.csv")
