import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📩 Spam Email Classifier")

message = st.text_area("Enter your message:")

if st.button("Predict"):
    if message.strip() != "":
        data = vectorizer.transform([message])
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0]

        if prediction == 1:
            st.error(f"🚨 Spam (Confidence: {probability[1]*100:.2f}%)")
        else:
            st.success(f"✅ Not Spam (Confidence: {probability[0]*100:.2f}%)")
    else:
        st.warning("Please enter a message.")
