# tools/plot_confmat.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
from pathlib import Path

from train_baseline import CSVImageDataset, eval_tf, make_model

def predict_all(csv_path, ckpt):
    ds = CSVImageDataset(csv_path, eval_tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = make_model().to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x,y in dl:
            x = x.to(device)
            p = m(x).argmax(1).cpu().numpy()
            ys.extend(y.numpy()); ps.extend(p)
    return np.array(ys), np.array(ps)

# Generate all confusion matrices
configs = [
    ("splits/random_test.csv", "best_random.pt", "figures/confusion_random.png"),
    ("splits/loso_wrigley/test.csv", "best_loso_wrigley.pt", "figures/confusion_loso_wrigley.png"),
    ("splits/loso_amfam/test.csv", "best_loso_amfam.pt", "figures/confusion_loso_amfam.png"),
    ("splits/loso_fenway/test.csv", "best_loso_fenway.pt", "figures/confusion_loso_fenway.png"),
    ("splits/loso_guaranteedrate/test.csv", "best_loso_guaranteedrate.pt", "figures/confusion_loso_guaranteedrate.png"),
    ("splits/loso_oracle/test.csv", "best_loso_oracle.pt", "figures/confusion_loso_oracle.png"),
    ("splits/loso_target/test.csv", "best_loso_target.pt", "figures/confusion_loso_target.png"),
    ("splits/loso_tropicana/test.csv", "best_loso_tropicana.pt", "figures/confusion_loso_tropicana.png"),
    ("splits/loso_yankee/test.csv", "best_loso_yankee.pt", "figures/confusion_loso_yankee.png"),
]

Path("figures").mkdir(exist_ok=True)

for csv_path, ckpt, out in configs:
    ys, ps = predict_all(csv_path, ckpt)
    cm = confusion_matrix(ys, ps)
    
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["clear", "obstructed"])
    ax.set_yticklabels(["clear", "obstructed"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")