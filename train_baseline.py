# train_baseline.py
import torch, torchvision as tv
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

IMSIZE = 224
MEAN, STD = (0.485,0.456,0.406), (0.229,0.224,0.225)

train_tf = tv.transforms.Compose([
    tv.transforms.RandomResizedCrop(IMSIZE, scale=(0.7,1.0)),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ColorJitter(0.1,0.1,0.1,0.05),
    tv.transforms.GaussianBlur(3, sigma=(0.1,1.0)),
    tv.transforms.RandomPerspective(distortion_scale=0.15, p=0.2),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(MEAN, STD),
])
eval_tf = tv.transforms.Compose([
    tv.transforms.Resize(256),
    tv.transforms.CenterCrop(IMSIZE),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(MEAN, STD),
])

class CSVImageDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.cls = {"clear":0, "obstructed":1}
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        x = self.transform(img)
        y = self.cls[str(r["label"]).lower()]
        return x, y

def make_loaders(split_dir="splits", kind="random", bs=32, nw=4):
    # Map 'kind' to actual csv paths
    if kind == "random":
        tr = f"{split_dir}/random_train.csv"
        va = f"{split_dir}/random_val.csv"
        te = f"{split_dir}/random_test.csv"
    elif kind.startswith("loso_"):
        base = f"{split_dir}/{kind}"
        tr = f"{base}/train.csv"
        va = f"{base}/val.csv"
        te = f"{base}/test.csv"
    else:
        raise ValueError(f"Unknown split kind: {kind}")

    # DataLoaders (pin_memory=False to avoid MPS warning on Mac)
    dl_tr = DataLoader(CSVImageDataset(tr, train_tf), batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=False)
    dl_va = DataLoader(CSVImageDataset(va, eval_tf),  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=False)
    dl_te = DataLoader(CSVImageDataset(te, eval_tf),  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=False)

    return dl_tr, dl_va, dl_te, tr  # <-- return the train CSV path too (for class weights)

def class_weights(csv_path):
    df = pd.read_csv(csv_path)
    vc = df["label"].str.lower().value_counts()
    total = len(df)
    w = [total/(2*vc.get("clear",1)), total/(2*vc.get("obstructed",1))]
    return torch.tensor(w, dtype=torch.float)

def make_model():
    m = tv.models.mobilenet_v3_small(weights=tv.models.MobileNet_V3_Small_Weights.DEFAULT)
    in_f = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_f, 2)
    return m

@torch.no_grad()
def eval_model(m, loader, device):
    m.eval()
    ys, ps = [], []
    for x,y in loader:
        x = x.to(device)
        logits = m(x)
        pred = logits.argmax(1).cpu().numpy()
        ys.extend(y.numpy()); ps.extend(pred)
    ys, ps = np.array(ys), np.array(ps)
    acc = (ys==ps).mean()
    return acc, ys, ps

def train(kind="random"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dl_tr, dl_va, dl_te, tr_csv = make_loaders(kind=kind)  # <-- accept 4 returns
    m = make_model().to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights(tr_csv).to(device))  # <-- use correct CSV for weights
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)

    best_acc, patience, bad = 0.0, 5, 0
    for epoch in range(25):
        m.train()
        for x,y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(m(x), y)
            loss.backward()
            opt.step()
        acc, _, _ = eval_model(m, dl_va, device)
        print(f"[{kind}] epoch {epoch}  val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc, bad = acc, 0
            torch.save(m.state_dict(), f"best_{kind}.pt")
        else:
            bad += 1
            if bad >= patience: break

    m.load_state_dict(torch.load(f"best_{kind}.pt", map_location=device))
    acc, ys, ps = eval_model(m, dl_te, device)
    print(f"[{kind}] test_acc={acc:.3f}")
    print(confusion_matrix(ys, ps))
    print(classification_report(ys, ps, target_names=["clear","obstructed"]))

if __name__ == "__main__":
    # train(kind="random")
    # train(kind="loso_wrigley")  # or loso_amfam / loso_oracle / loso_fenway

    for stadium in ["amfam", "fenway", "guaranteedrate", "oracle", "target", "tropicana", "yankee"]:
        train(kind=f"loso_{stadium}")