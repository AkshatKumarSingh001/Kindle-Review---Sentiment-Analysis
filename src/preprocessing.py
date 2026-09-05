import re
from typing import Any

import numpy as np


def preprocess_text(text: str, preprocessor_bundle: dict[str, Any]) -> str:
    """Apply the same cleaning and lemmatization used during training."""
    text = str(text)

    if preprocessor_bundle["lowercase"]:
        text = text.lower()

    text = re.sub(preprocessor_bundle["special_characters_pattern"], "", text)

    if preprocessor_bundle["remove_stopwords"]:
        stopword_set = set(preprocessor_bundle["stopwords"])
        text = " ".join(
            word for word in text.split() if word not in stopword_set
        )

    text = re.sub(preprocessor_bundle["url_pattern"], "", text)

    if preprocessor_bundle["remove_html"]:
        from bs4 import BeautifulSoup
        text = BeautifulSoup(text, "lxml").get_text()

    if preprocessor_bundle["remove_extra_spaces"]:
        text = " ".join(text.split())

    if preprocessor_bundle["lemmatize"]:
        lemmatizer = preprocessor_bundle["lemmatizer"]
        text = " ".join(lemmatizer.lemmatize(word) for word in text.split())

    return text


def average_word2vec_vector(
    text: str,
    word2vec_model: Any,
) -> np.ndarray:
    """Convert one cleaned review into an average Word2Vec vector."""
    tokens = text.split()
    known_tokens = [token for token in tokens if token in word2vec_model.wv]

    if not known_tokens:
        return np.zeros(word2vec_model.vector_size)

    return np.mean(
        [word2vec_model.wv[token] for token in known_tokens],
        axis=0,
    )
