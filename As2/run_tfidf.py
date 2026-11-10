import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

import common as C

SEED = 42


def build_model():
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                          sublinear_tf=True, strip_accents="unicode")
    clf = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))
    return vec, clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["s0", "s1"], required=True)
    args = ap.parse_args()

    df = C.load_paragraphs()
    y = C.label_matrix(df)
    rows, per_label = [], []

    for k in range(C.N_FOLDS):
        tr, dv, te = C.split_for(df, k)
        vec, clf = build_model()
        clf.fit(vec.fit_transform(df.loc[tr, "text"]), y[tr])
        p_dev = clf.predict_proba(vec.transform(df.loc[dv, "text"]))
        p_test = clf.predict_proba(vec.transform(df.loc[te, "text"]))

        # s0 固定 0.5，s1 在开发折上调阈值
        if args.system == "s0":
            thr = np.full(len(C.LABELS), 0.5)
        else:
            thr = C.tune_per_label_thresholds(y[dv], p_dev)

        y_pred = C.apply_thresholds(p_test, thr)
        pred_path = C.OUTPUTS / args.system / f"fold{k}.pred.txt"
        C.write_predictions(df.loc[te], y_pred, pred_path)

        micro, macro = C.official_score(pred_path, k)
        own_micro, _ = C.micro_macro_f1(y[te], y_pred)
        assert abs(own_micro - micro) < 1e-9, (own_micro, micro)
        dev_micro, _ = C.micro_macro_f1(y[dv], C.apply_thresholds(p_dev, thr))

        rows.append({"fold": k, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                     "dev_micro": dev_micro, "micro": micro, "macro": macro,
                     "pred_empty_rate": float((y_pred.sum(1) == 0).mean()),
                     "mean_thr": float(np.mean(thr))})
        per_label.append(C.per_label_f1(y[te], y_pred))

    C.save_results(args.system, rows, per_label)


if __name__ == "__main__":
    main()
