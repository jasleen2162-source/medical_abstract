import streamlit as st

from src.demo import predict_sentence


st.title(

 "Medical Abstract Classifier"

)

sentence = st.text_area(

 "Enter sentence"

)

if st.button(

 "Predict"

):

    predict_sentence(

      sentence

    )