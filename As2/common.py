import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
LABELS_FILE = DATA / "labels.txt"
SCORER = ROOT / "raw" / "starter-kit" / "subtask1" / "scorer-subtask-1.py"

LABELS = [l.strip() for l in open(LABELS_FILE, encoding="utf-8") if l.strip()]
N_FOLDS = 5
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def load_paragraphs():
    df = pd.read_csv(DATA / "en_paragraphs.tsv", sep="\t", keep_default_na=False,
                     quoting=csv.QUOTE_MINIMAL)
    df["label_list"] = df["labels"].map(lambda s: s.split("|") if s else [])
    return df


def label_matrix(df):
    y = np.zeros((len(df), len(LABELS)), dtype=int)
    idx = {l: i for i, l in enumerate(LABELS)}
    for r, labs in enumerate(df["label_list"]):
        for l in labs:
            y[r, idx[l]] = 1
    return y


def fold_docs():
    return json.load(open(DATA / "splits.json"))["folds"]


# 测试折 k，开发折 k+1，其余三折训练
def split_for(df, k):
    folds = fold_docs()
    test = df["doc"].isin(set(folds[k])).to_numpy()
    dev = df["doc"].isin(set(folds[(k + 1) % N_FOLDS])).to_numpy()
    train = ~(test | dev)
    return train, dev, test


# 空标签行记为 None，与官方 scorer 一致
def _with_none(y):
    none = (y.sum(axis=1) == 0).astype(int)[:, None]
    return np.hstack([y, none])


def micro_macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    t, p = _with_none(y_true), _with_none(y_pred)
    micro = f1_score(t, p, average="micro", zero_division=1)
    macro = f1_score(t, p, average="macro", zero_division=1)
    return micro, macro


def per_label_f1(y_true, y_pred):
    from sklearn.metrics import precision_recall_fscore_support
    t, p = _with_none(y_true), _with_none(y_pred)
    pr, rc, f1, sup = precision_recall_fscore_support(t, p, average=None, zero_division=0)
    return pd.DataFrame({"label": LABELS + ["None"], "precision": pr, "recall": rc,
                         "f1": f1, "support": sup})


def apply_thresholds(probs, thr):
    return (probs >= np.asarray(thr)).astype(int)


def tune_global_threshold(y_true, probs):
    return float(max(GRID, key=lambda t: micro_macro_f1(y_true, apply_thresholds(probs, t))[0]))


# 从全局阈值出发，逐个标签调阈值
def tune_per_label_thresholds(y_true, probs, n_rounds=2):
    thr = np.full(probs.shape[1], tune_global_threshold(y_true, probs))
    for _ in range(n_rounds):
        for j in range(probs.shape[1]):
            def score(t):
                cand = thr.copy()
                cand[j] = t
                return micro_macro_f1(y_true, apply_thresholds(probs, cand))[0]
            thr[j] = max(GRID, key=score)
    return thr


def write_predictions(df, y_pred, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for (_, r), row in zip(df.iterrows(), y_pred):
            labs = [LABELS[j] for j in np.flatnonzero(row)]
            f.write("\t".join([r["doc"], str(r["start"]), str(r["end"])] + labs) + "\n")


# 调用官方 scorer
def official_score(pred_path, k):
    gold = DATA / "gold" / f"fold{k}.txt"
    out = subprocess.run(
        [sys.executable, "-I", str(SCORER), "-g", str(gold), "-p", str(pred_path),
         "-f", str(LABELS_FILE), "-o"],
        capture_output=True, text=True, check=True)
    last = [l for l in out.stdout.strip().splitlines() if "\t" in l][-1]
    micro, macro = map(float, last.split("\t"))
    return micro, macro


def save_results(system, rows, per_label):
    out = OUTPUTS / system
    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "scores.csv", index=False)
    pl = pd.concat(per_label).groupby("label", sort=False)[["precision", "recall", "f1", "support"]].mean()
    pl.to_csv(out / "per_label.csv")
    print(f"\n[{system}]")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"micro-F1 = {res.micro.mean():.4f} ± {res.micro.std():.4f}   "
          f"macro-F1 = {res.macro.mean():.4f} ± {res.macro.std():.4f}")
    return res