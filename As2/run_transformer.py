import argparse
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import common as C

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# s3：第二段输入为标题 + 前一段 + 后一段
def build_inputs(df, use_context):
    first = df["text"].tolist()
    if not use_context:
        return first, None
    prev_text, next_text = {}, {}
    for _, g in df.sort_values(["doc", "para_idx"]).groupby("doc"):
        texts = g["text"].tolist()
        for i, idx in enumerate(g.index):
            prev_text[idx] = texts[i - 1] if i > 0 else ""
            next_text[idx] = texts[i + 1] if i + 1 < len(texts) else ""
    second = [f"{r.title} | {prev_text[i]} | {next_text[i]}" for i, r in df.iterrows()]
    return first, second


def make_loader(tok, first, second, y, rows, args, shuffle):
    a = [first[i] for i in rows]
    b = [second[i] for i in rows] if second is not None else None
    enc = tok(a, b, truncation="longest_first", max_length=args.max_len)
    items = [{"input_ids": enc["input_ids"][j], "attention_mask": enc["attention_mask"][j],
              "labels": y[rows[j]].astype(np.float32)} for j in range(len(rows))]

    def collate(batch):
        out = tok.pad([{k: v for k, v in x.items() if k != "labels"} for x in batch], return_tensors="pt")
        out["labels"] = torch.tensor(np.stack([x["labels"] for x in batch]))
        return out

    return DataLoader(items, batch_size=args.bs if shuffle else args.bs * 2, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        probs.append(torch.sigmoid(model(**batch).logits).float().cpu().numpy())
    return np.concatenate(probs)


def run_fold(k, df, y, first, second, tok, args, device):
    tr, dv, te = C.split_for(df, k)
    tr_rows, dv_rows, te_rows = np.flatnonzero(tr), np.flatnonzero(dv), np.flatnonzero(te)
    set_seed(SEED)
    train_dl = make_loader(tok, first, second, y, tr_rows, args, True)
    dev_dl = make_loader(tok, first, second, y, dv_rows, args, False)
    test_dl = make_loader(tok, first, second, y, te_rows, args, False)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(C.LABELS), problem_type="multi_label_classification").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    # 每个 epoch 记录 dev/test 概率，只按 dev 选 epoch
    best = None
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, loss_sum = time.time(), 0.0
        for batch in train_dl:
            batch = {k_: v.to(device) for k_, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            loss_sum += loss.item()
        p_dev, p_test = predict(model, dev_dl, device), predict(model, test_dl, device)
        thr = C.tune_per_label_thresholds(y[dv], p_dev)
        dev_micro, _ = C.micro_macro_f1(y[dv], C.apply_thresholds(p_dev, thr))
        print(f"  fold{k} ep{ep} loss={loss_sum / len(train_dl):.4f} dev_micro={dev_micro:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if best is None or dev_micro > best["dev_micro"]:
            best = {"epoch": ep, "dev_micro": dev_micro, "thr": thr, "p_test": p_test}

    del model, opt
    if device.type == "mps":
        torch.mps.empty_cache()
    return tr, te, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["s2", "s3"], required=True)
    ap.add_argument("--model", default="FacebookAI/roberta-base")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(C.N_FOLDS)))
    args = ap.parse_args()
    use_context = args.system == "s3"
    if args.max_len is None:
        args.max_len = 384 if use_context else 256

    device = get_device()
    print(f"device={device} model={args.model} max_len={args.max_len}")
    df = C.load_paragraphs()
    y = C.label_matrix(df)
    first, second = build_inputs(df, use_context)
    tok = AutoTokenizer.from_pretrained(args.model)

    rows, per_label = [], []
    for k in args.folds:
        t0 = time.time()
        tr, te, best = run_fold(k, df, y, first, second, tok, args, device)
        y_pred = C.apply_thresholds(best["p_test"], best["thr"])
        pred_path = C.OUTPUTS / args.system / f"fold{k}.pred.txt"
        C.write_predictions(df.loc[te], y_pred, pred_path)
        np.save(C.OUTPUTS / args.system / f"fold{k}.probs.npy", best["p_test"])
        micro, macro = C.official_score(pred_path, k)
        rows.append({"fold": k, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                     "best_epoch": best["epoch"], "dev_micro": best["dev_micro"],
                     "micro": micro, "macro": macro,
                     "pred_empty_rate": float((y_pred.sum(1) == 0).mean()),
                     "mean_thr": float(np.mean(best["thr"])), "minutes": (time.time() - t0) / 60})
        per_label.append(C.per_label_f1(y[te], y_pred))
        print(f"fold{k}: micro={micro:.4f} macro={macro:.4f}", flush=True)

    C.save_results(args.system, rows, per_label)


if __name__ == "__main__":
    main()
