import pickle
import sys
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess_text

DATA_PATH = PROJECT_ROOT / "data" / "all_kindle_review.csv"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "best_sentiment_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"


def main() -> None:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    data = pd.read_csv(DATA_PATH)
    df = data[["reviewText", "rating"]].copy()
    df["rating"] = df["rating"].apply(lambda rating: 0 if rating < 3 else 1)

    lemmatizer = nltk.stem.WordNetLemmatizer()
    stopword_list = nltk.corpus.stopwords.words("english")
    preprocessor_bundle = {
        "lowercase": True,
        "special_characters_pattern": r"[^a-z A-z 0-9-]+",
        "url_pattern": r"(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?",
        "remove_html": True,
        "remove_extra_spaces": True,
        "remove_stopwords": True,
        "stopwords_language": "english",
        "stopwords": sorted(stopword_list),
        "lemmatize": True,
        "lemmatizer": lemmatizer,
        "embedding_type": "average_word2vec",
        "vector_size": 100,
    }

    df["reviewText"] = df["reviewText"].apply(
        lambda review: preprocess_text(review, preprocessor_bundle)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df["reviewText"],
        df["rating"],
        test_size=0.20,
        random_state=42,
    )

    train_tokens = [review.split() for review in X_train]
    word2vec_model = Word2Vec(
        sentences=train_tokens,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        seed=42,
    )

    def average_word2vec_vectors(tokenized_reviews):
        vectors = np.zeros((len(tokenized_reviews), word2vec_model.vector_size))
        for row_index, tokens in enumerate(tokenized_reviews):
            known_tokens = [
                token for token in tokens if token in word2vec_model.wv
            ]
            if known_tokens:
                vectors[row_index] = np.mean(
                    [word2vec_model.wv[token] for token in known_tokens],
                    axis=0,
                )
        return vectors

    X_train_avg_word2vec = average_word2vec_vectors(train_tokens)
    X_test_avg_word2vec = average_word2vec_vectors(
        [review.split() for review in X_test]
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "SVM": SVC(kernel="rbf", C=1.0),
        "Gaussian Naive Bayes": GaussianNB(),
    }

    results = {}
    for name, candidate in classifiers.items():
        candidate.fit(X_train_avg_word2vec, y_train)
        predictions = candidate.predict(X_test_avg_word2vec)
        results[name] = {
            "model": candidate,
            "accuracy": accuracy_score(y_test, predictions),
        }

    best_name = max(results, key=lambda name: results[name]["accuracy"])
    classifier = results[best_name]["model"]
    validation_accuracy = results[best_name]["accuracy"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with PREPROCESSOR_PATH.open("wb") as file:
        pickle.dump(preprocessor_bundle, file)

    model_bundle = {
        "model": classifier,
        "classifier_name": best_name,
        "word2vec_model": word2vec_model,
        "embedding_type": "average_word2vec",
        "vector_size": word2vec_model.vector_size,
        "classes": classifier.classes_.tolist(),
        "validation_accuracy": validation_accuracy,
    }
    with MODEL_PATH.open("wb") as file:
        pickle.dump(model_bundle, file)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved preprocessor to: {PREPROCESSOR_PATH}")
    print("Model comparison:")
    for name, result in results.items():
        print(f"  {name}: {result['accuracy']:.4f}")
    print(f"Selected model: {best_name}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")


if __name__ == "__main__":
    main()
