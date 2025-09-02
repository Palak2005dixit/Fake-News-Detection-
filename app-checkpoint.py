import streamlit as st
import joblib
 
vectorizer=joblib.load("vectorizer.jb")
model=joblib.load("lr_model.jb")
st.title("Fake News Detection")
st.write("Enter a News Article below to check whether it is Real or Fake.")

news_input=st.text_area("Enter the News Article:","")
if st.button("Check News"):
    if news_input.strip():
       transformed_input=vectorizer.transform([news_input])
       prediction=model.predict(transformed_input)

       if prediction[0]==1:
           st.success("The News Article is Real.")
       else:
           st.error("The News Article is Fake.")
    else:
        st.warning("Please enter some text to analyze.")