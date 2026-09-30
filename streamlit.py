import streamlit as st

from src.demo import predict_abstract


# Page configuration
st.set_page_config(
    page_title="Medical Abstract Classifier",
    page_icon="🧬"
)


# Title
st.title("🧬 Medical Abstract Sentence Classifier")

st.write(
    "Enter a complete medical abstract and the model "
    "will classify each sentence into its rhetorical role."
)


# Abstract input
abstract = st.text_area(
    "Enter Medical Abstract",
    height=300,
    placeholder="Paste a medical research abstract here..."
)


# Predict button
if st.button("Classify Abstract"):

    if not abstract.strip():

        st.warning("Please enter a medical abstract.")

    else:

        results = predict_abstract(abstract)

        st.subheader("Classification Results")


        # Colors for each classification
        colors = {
            "BACKGROUND": "#D6EAF8",
            "OBJECTIVE": "#FCF3CF",
            "METHODS": "#D5F5E3",
            "RESULTS": "#FADBD8",
            "CONCLUSIONS": "#E8DAEF"
        }


        # Display each sentence
        for result in results:

            label = result["label"]
            sentence = result["sentence"]
            color = colors[label]

            st.markdown(
                f"""
                <div style="
                    background-color: {color};
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 8px;
                    border-left: 6px solid #555;
                ">

                    <div style="
                        font-weight: bold;
                        font-size: 18px;
                        margin-bottom: 8px;
                    ">
                        {label}
                    </div>

                    <div style="
                        font-size: 16px;
                        line-height: 1.6;
                    ">
                        {sentence}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )
