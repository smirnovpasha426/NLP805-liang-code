import sys, collections, pathlib
root = pathlib.Path(sys.argv[1])
for name, d in [("second-batch EN", root/"data/training-second-batch-oct/EN"),
                ("initial-B EN", root/"data/initial-training-data/subtask1/SEMEVAL23_FRAMING_INITIAL_TRAIN-B/EN")]:
    rows = []
    for line in open(d/"subtask-1-annotations.txt", encoding="utf-8-sig"):
        line = line.rstrip("\r\n")
        if not line: continue
        f = line.split("\t")
        rows.append((f[0], int(f[1]), int(f[2]), [x for x in f[3:] if x.strip()]))
    docs = collections.defaultdict(list)
    for r in rows: docs[r[0]].append(r)
    lab = collections.Counter(l for r in rows for l in r[3])
    nlab = collections.Counter(len(r[3]) for r in rows)
    # coverage: fraction of doc chars covered by annotated spans; offset sanity
    cov, bad = [], 0
    for doc, rs in docs.items():
        text = open(d/"raw-documents"/doc, encoding="utf-8").read()
        cov.append(sum(e-s for _,s,e,_ in rs)/max(1,len(text.strip())))
        for _,s,e,_ in rs:
            seg = text[s:e]
            if not seg.strip() or seg != seg.strip(): bad += 1
    print(f"== {name}: {len(docs)} docs, {len(rows)} rows, avg rows/doc {len(rows)/len(docs):.1f}")
    print(f"   span coverage of doc text: mean {sum(cov)/len(cov):.2f} min {min(cov):.2f}; spans w/ odd whitespace/empty: {bad}")
    print("   #labels per row:", dict(sorted(nlab.items())))
    for l,c in lab.most_common(): print(f"   {c:5d} {c/len(rows):6.1%}  {l}")
