# make_splits.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(".")
df = pd.read_csv(ROOT/"master.csv")

# normalize
df["label"] = df["label"].str.lower().str.strip()
df["stadium"] = df["stadium"].str.lower().str.strip()

Path("splits").mkdir(exist_ok=True)

# ---------- A) 70/15/15 stratified by label ----------
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
tr_idx, tmp_idx = next(sss1.split(df, df["label"]))
tr = df.iloc[tr_idx].reset_index(drop=True)
tmp = df.iloc[tmp_idx].reset_index(drop=True)

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=43)
va_idx, te_idx = next(sss2.split(tmp, tmp["label"]))
va = tmp.iloc[va_idx].reset_index(drop=True)
te = tmp.iloc[te_idx].reset_index(drop=True)

tr.to_csv("splits/random_train.csv", index=False)
va.to_csv("splits/random_val.csv", index=False)
te.to_csv("splits/random_test.csv", index=False)

print("Random split:", len(tr), len(va), len(te))
print("Random train label counts:\n", tr["label"].value_counts(), "\n")

# ---------- B) LOSO (Leave-One-Stadium-Out) ----------
# Make one split per stadium, so you can pick any for the midterm.
for held_out in sorted(df["stadium"].unique()):
    if held_out in ("unknown", ""):  # skip unknowns if any
        continue
    te2 = df[df["stadium"]==held_out].reset_index(drop=True)
    tr_all = df[df["stadium"]!=held_out].reset_index(drop=True)
    # 15% of (non-held-out) for validation (stratified by label)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=45)
    tr2_idx, va2_idx = next(sss_val.split(tr_all, tr_all["label"]))
    tr2 = tr_all.iloc[tr2_idx].reset_index(drop=True)
    va2 = tr_all.iloc[va2_idx].reset_index(drop=True)

    outdir = Path("splits")/f"loso_{held_out}"
    outdir.mkdir(exist_ok=True)
    tr2.to_csv(outdir/"train.csv", index=False)
    va2.to_csv(outdir/"val.csv", index=False)
    te2.to_csv(outdir/"test.csv", index=False)

    print(f"LOSO='{held_out}' ->", len(tr2), len(va2), len(te2))
