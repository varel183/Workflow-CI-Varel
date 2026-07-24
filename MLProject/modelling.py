import argparse
import os

import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def main(data_path: str):
    # Jangan panggil mlflow.set_experiment()/start_run(run_name=...) di sini -- `mlflow run` sudah menyiapkan run aktif lewat MLFLOW_RUN_ID, memanggilnya lagi bikin run ID conflict.
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
    )

    mlflow.sklearn.autolog()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("svm", SVC(kernel="linear", random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    test_acc = pipeline.score(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join("playstore_reviews_preprocessing", "clean_reviews.csv"),
    )
    args = parser.parse_args()
    main(args.data_path)
