from pathlib import Path
import pandas as pd
import re

root = Path("/Users/zsfleming/Downloads/566-Project-zfleming/data")

rows = []

for label_dir in root.iterdir():
    if not label_dir.is_dir():
        continue
    label = label_dir.name.lower()  # 'clear' or 'obstructed'
    for p in label_dir.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            fname = p.name.lower()
            match = re.match(r"([a-z0-9]+)_", fname)
            stadium = match.group(1) if match else "unknown"
            rows.append({
                "path": str(p),
                "label": label,
                "stadium": stadium
            })

df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
out_path = "/Users/zsfleming/Downloads/566-Project-zfleming/master.csv"
df.to_csv(out_path, index=False)

print(f"Wrote master.csv with {len(df)} rows")
print(df["stadium"].value_counts())
