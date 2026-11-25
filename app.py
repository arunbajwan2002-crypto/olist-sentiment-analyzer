import streamlit as st
import joblib


model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


st.set_page_config(page_title="Olist Sentiment Analyzer", page_icon="🛍️")
st.title("🛍️ E-Commerce Customer Sentiment Analyzer")
st.write("Enter a customer review below to predict whether the sentiment is Positive or Negative.")

user_review = st.text_area("Customer Review:", placeholder="Example: Produto excelente, chegou antes do prazo!")

if st.button("Predict Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        vec_text = vectorizer.transform([user_review])
        prediction = model.predict(vec_text)
        
        if prediction == 1:
            st.success("✅ **Positive Sentiment** (Customer is satisfied)")
        else:
            st.error("❌ **Negative Sentiment** (Customer is dissatisfied / At-Risk)")
