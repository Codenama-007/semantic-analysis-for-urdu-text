import streamlit as st
from predict import predict_categories

# Page configuration

st.set_page_config(
    page_title="Urdu Semantic Analysis",
    page_icon="🧠",
    layout="centered"
)

# Header

st.title("🧠 Urdu Text Semantic Analysis")

st.write(
    "Enter Urdu text and the model will predict "
    "its semantic categories."
)

# Text input

text = st.text_area(
    "Enter Urdu text",
    placeholder="مثال: اللہ بہت مہربان اور رحم کرنے والا ہے",
    height=150
)

# Analyze button

if st.button(
    "Analyze Text",
    type="primary"
):

    if not text.strip():

        st.warning(
            "Please enter some Urdu text."
        )

    else:

        with st.spinner(
            "Analyzing Urdu text..."
        ):

            processed_text, predictions, probabilities = (
                predict_categories(text)
            )

        # Preprocessed text

        st.subheader(
            "Preprocessed Text"
        )

        st.code(
            processed_text,
            language=None
        )

        # Predictions

        st.subheader(
            "Predicted Semantic Categories"
        )

        if predictions:

            for prediction in predictions:

                category = prediction[
                    "category"
                ]

                confidence = prediction[
                    "confidence"
                ]

                st.write(
                    f"**{category}**"
                )

                st.progress(
                    confidence,
                    text=(
                        f"Confidence: "
                        f"{confidence * 100:.2f}%"
                    )
                )

        else:

            st.info(
                "No semantic category crossed "
                "the confidence threshold."
            )

# Footer

st.divider()

st.caption(
    "Urdu Text Semantic Analysis • "
    "BiLSTM Multi-Label Classification"
)