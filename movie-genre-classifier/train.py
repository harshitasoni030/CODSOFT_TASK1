"""Train, compare and save the best movie genre classifier.

Usage:  python train.py [--data PATH_TO_CSV] [--top-genres 10]
"""
import argparse, json, sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "genre_model.joblib"
META_PATH = ROOT / "models" / "metadata.json"
ALIASES = {"science fiction": "sci-fi", "sci fi": "sci-fi", "scifi": "sci-fi",
           "romantic": "romance", "romantic comedy": "comedy", "animated": "animation"}


def find_csv(arg):
    if arg:
        return Path(arg)
    found = sorted((ROOT / "data").glob("*.csv"))
    if not found:
        sys.exit("No CSV found. Put wiki_movie_plots_deduped.csv in ./data or pass --data PATH.")
    return found[0]


def load_data(path, top_genres):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"genre", "plot"} <= set(df.columns):
        sys.exit(f"CSV needs 'Genre' and 'Plot' columns, found: {list(df.columns)}")
    df = df[["plot", "genre"]].dropna()
    # Keep the first listed genre as the single label, then normalise it.
    df["genre"] = (df["genre"].str.lower().str.split(r"[,/;]", regex=True).str[0]
                   .str.strip().replace(ALIASES))
    df = df[~df["genre"].isin(["", "unknown"])]
    df = df[df["plot"].str.split().str.len() >= 10].drop_duplicates("plot")
    keep = df["genre"].value_counts().head(top_genres).index
    return df[df["genre"].isin(keep)].reset_index(drop=True)


def build(clf):
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3,
                            max_features=60000, sublinear_tf=True)
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--top-genres", type=int, default=10)
    args = ap.parse_args()

    path = find_csv(args.data)
    df = load_data(path, args.top_genres)
    print(f"Loaded {len(df)} movies from {path}\n{df['genre'].value_counts().to_string()}\n")

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["plot"], df["genre"], test_size=0.2, stratify=df["genre"], random_state=42)

    candidates = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(C=5, max_iter=1000, class_weight="balanced"),
        "Linear SVM": LinearSVC(C=0.5, class_weight="balanced"),
    }
    results, fitted = {}, {}
    for name, clf in candidates.items():
        model = build(clf).fit(X_tr, y_tr)
        pred = model.predict(X_te)
        results[name] = {
            "f1_macro": round(f1_score(y_te, pred, average="macro"), 4),
            "f1_weighted": round(f1_score(y_te, pred, average="weighted"), 4),
            "accuracy": round(float((pred == y_te).mean()), 4),
        }
        fitted[name] = model
        print(f"{name:20s} {results[name]}")

    best = max(results, key=lambda n: results[n]["f1_macro"])  # selection metric: macro F1
    print(f"\nBest model: {best}\n")
    print(classification_report(y_te, fitted[best].predict(X_te), zero_division=0))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(fitted[best], MODEL_PATH)
    META_PATH.write_text(json.dumps({
        "best_model": best, "selection_metric": "f1_macro", "results": results,
        "genres": list(fitted[best].classes_),
        "supports_confidence": hasattr(fitted[best], "predict_proba"),
        "train_size": len(X_tr), "test_size": len(X_te)}, indent=2))
    print(f"Saved {MODEL_PATH} and {META_PATH}")


if __name__ == "__main__":
    main()
