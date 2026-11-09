import csv
import json
import random
from pathlib import Path

SEED = 42
N_FOLDS = 5

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw" / "data"
EN = RAW / "training-second-batch-oct" / "EN"
INIT_B = RAW / "initial-training-data" / "subtask1" / "SEMEVAL23_FRAMING_INITIAL_TRAIN-B"
OUT = ROOT / "data"


def read_labels():
    labels = [l.strip() for l in open(INIT_B / "labels.txt", encoding="utf-8-sig") if l.strip()]
    assert len(labels) == 14, labels
    return labels


# "No framing" 不是官方标签，按空标签处理
def read_annotations(path, labels):
    rows = []
    for line in open(path, encoding="utf-8-sig"):
        line = line.rstrip("\r\n")
        if not line:
            continue
        f = line.split("\t")
        labs = [x.strip() for x in f[3:] if x.strip() and x.strip() != "No framing"]
        unknown = [x for x in labs if x not in labels]
        assert not unknown, (path, line, unknown)
        rows.append((f[0], int(f[1]), int(f[2]), sorted(set(labs))))
    return rows


def build(rows, doc_dir):
    out, by_doc = [], {}
    for r in rows:
        by_doc.setdefault(r[0], []).append(r)
    for doc, rs in by_doc.items():
        # 偏移量按 CRLF 计算，必须保留原始换行
        text = open(doc_dir / doc, encoding="utf-8", newline="").read()
        title = text.split("\n", 1)[0].strip()
        rs.sort(key=lambda r: r[1])
        for i, (_, s, e, labs) in enumerate(rs):
            out.append({
                "doc": doc, "start": s, "end": e, "para_idx": i, "n_paras": len(rs),
                "labels": "|".join(labs), "title": title,
                "text": " ".join(text[s:e].split()),
            })
    return out


def write_tsv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t",
                           quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "gold").mkdir(exist_ok=True)
    labels = read_labels()
    (OUT / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    paras = build(read_annotations(EN / "subtask-1-annotations.txt", labels), EN / "raw-documents")
    write_tsv(OUT / "en_paragraphs.tsv", paras)

    # 按文章划分 5 折
    docs = sorted({p["doc"] for p in paras})
    random.Random(SEED).shuffle(docs)
    folds = [sorted(docs[k::N_FOLDS]) for k in range(N_FOLDS)]
    json.dump({"seed": SEED, "n_folds": N_FOLDS, "folds": folds}, open(OUT / "splits.json", "w"), indent=1)

    # 清洗后的 gold：LF 换行，空标签行不带多余的 Tab
    for k, fold in enumerate(folds):
        fold = set(fold)
        with open(OUT / "gold" / f"fold{k}.txt", "w", encoding="utf-8", newline="\n") as f:
            for p in paras:
                if p["doc"] in fold:
                    labs = p["labels"].split("|") if p["labels"] else []
                    f.write("\t".join([p["doc"], str(p["start"]), str(p["end"])] + labs) + "\n")
        print(f"fold{k}: {len(fold)} docs, {sum(p['doc'] in fold for p in paras)} paragraphs")

    extra = build(read_annotations(INIT_B / "EN" / "subtask-1-annotations.txt", labels),
                  INIT_B / "EN" / "raw-documents")
    write_tsv(OUT / "extra_initialB_en.tsv", extra)
    print(f"EN paragraphs: {len(paras)} in {len(docs)} docs; extra initial-B spans: {len(extra)}")


if __name__ == "__main__":
    main()
