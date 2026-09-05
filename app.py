import pickle
from pathlib import Path

import streamlit as st

from src.preprocessing import average_word2vec_vector, preprocess_text

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "best_sentiment_model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "model" / "preprocessor.pkl"

POSITIVE_EXAMPLE = (
    "This book completely pulled me in from the first page. "
    "I finished it in two sittings and immediately wanted more from this author."
)
NEGATIVE_EXAMPLE = (
    "The plot dragged on forever and the ending felt rushed. "
    "I was expecting a lot more given all the hype around this one."
)


@st.cache_resource
def load_artifacts():
    with MODEL_PATH.open("rb") as file:
        model_bundle = pickle.load(file)
    with PREPROCESSOR_PATH.open("rb") as file:
        preprocessor_bundle = pickle.load(file)
    return model_bundle, preprocessor_bundle


def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Source Sans 3', sans-serif;
        }

        .app-tagline {
            color: #5B6472;
            font-size: 1.05rem;
            margin-top: -0.6rem;
            margin-bottom: 1.5rem;
        }

        .result-box {
            padding: 1.1rem 1.4rem;
            border-radius: 10px;
            font-size: 1.15rem;
            font-weight: 600;
            margin-top: 0.75rem;
        }

        .result-positive {
            background-color: #E6F4EF;
            color: #0F6B4F;
            border: 1px solid #B7E4D4;
        }

        .result-negative {
            background-color: #FBEAEA;
            color: #A32A2A;
            border: 1px solid #F3C6C6;
        }

        .char-count {
            color: #8A94A3;
            font-size: 0.85rem;
            margin-top: -0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_example_text(text: str) -> None:
    st.session_state.review_text = text


st.set_page_config(
    page_title="Kindle Review Sentiment Analyzer",
    page_icon="📖",
    layout="centered",
)

inject_custom_css()

st.title("📖 Kindle Review Sentiment Analyzer")
st.markdown(
    '<p class="app-tagline">Paste a Kindle review below and see whether it reads as positive or negative.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About")
    st.write(
        "This app classifies the sentiment of Kindle product reviews using "
        "an average Word2Vec embedding and a trained Logistic Regression model."
    )
    st.markdown("**How it works**")
    st.markdown(
        "1. Clean the review text\n"
        "2. Convert it into an average Word2Vec vector\n"
        "3. Classify the vector as Positive or Negative"
    )
    st.markdown("**Model details**")
    st.markdown(
        "- Embedding: Word2Vec (size 100)\n"
        "- Classifier: Logistic Regression\n"
        "- Trained on 12,000 Kindle reviews"
    )
    st.divider()
    st.caption("Built by [Akshat Kumar Singh](https://github.com/AkshatKumarSingh001)")

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

try:
    model_bundle, preprocessor_bundle = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False
    st.error(
        "Model files could not be found. Make sure `model/best_sentiment_model.pkl` "
        "and `model/preprocessor.pkl` exist before running this app."
    )

with st.container(border=True):
    example_col1, example_col2 = st.columns(2)
    example_col1.button(
        "Try a positive example",
        on_click=set_example_text,
        args=(POSITIVE_EXAMPLE,),
        use_container_width=True,
    )
    example_col2.button(
        "Try a negative example",
        on_click=set_example_text,
        args=(NEGATIVE_EXAMPLE,),
        use_container_width=True,
    )

    review = st.text_area(
        "Your review",
        key="review_text",
        height=180,
        placeholder="Type or paste a Kindle review here...",
    )
    st.markdown(
        f'<p class="char-count">{len(review)} characters</p>',
        unsafe_allow_html=True,
    )

    action_col1, action_col2 = st.columns(2)
    analyze_clicked = action_col1.button(
        "Analyze sentiment",
        type="primary",
        use_container_width=True,
        disabled=not artifacts_loaded,
    )
    action_col2.button(
        "Clear",
        use_container_width=True,
        on_click=set_example_text,
        args=("",),
    )

if analyze_clicked:
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing sentiment..."):
            cleaned_review = preprocess_text(review, preprocessor_bundle)
            features = average_word2vec_vector(
                cleaned_review,
                model_bundle["word2vec_model"],
            ).reshape(1, -1)
            prediction = model_bundle["model"].predict(features)[0]

        if prediction == 1:
            st.markdown(
                '<div class="result-box result-positive">✅ Positive sentiment</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-box result-negative">❌ Negative sentiment</div>',
                unsafe_allow_html=True,
            )

        with st.expander("See processed text"):
            st.write(cleaned_review)