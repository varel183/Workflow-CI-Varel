"""Retraining script used by the CI workflow (MLflow Project entry point).

Melatih ulang model TF-IDF + SVM dari dataset yang sudah dipreprocessing
(hasil Kriteria 1), mencatat run ke MLflow tracking lokal (mlruns/) di dalam
runner GitHub Actions, sehingga folder mlruns/ bisa diunggah sebagai artefak CI.
"""

import argparse
import os

import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def main(data_path: str):
    # Tracking URI, experiment, dan run sudah disiapkan oleh `mlflow run` (MLflow
    # Projects) lewat env var MLFLOW_TRACKING_URI / MLFLOW_RUN_ID / MLFLOW_EXPERIMENT_ID.
    # Jangan panggil mlflow.set_experiment()/mlflow.start_run(run_name=...) di sini,
    # karena itu akan bentrok dengan run yang sudah dibuat oleh `mlflow run` (error:
    # "active run ID does not match environment run ID").
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
