import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK data dependencies are loaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Initialize NLP tools
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
punct = set(string.punctuation)

def transform_text(text: str) -> str:
    """Preprocess text identically to training time."""
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    
    filtered = [
        ps.stem(token) for token in tokens
        if token.isalnum() and token not in stop_words and token not in punct
    ]
    return " ".join(filtered)

# Load saved vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

st.title("📩 SMS / Email Spam Classifier")
st.write("Enter any message below to check whether it's classified as **Spam** or **Legitimate (Ham)**.")

input_sms = st.text_area("Enter Message Here", height=150, placeholder="e.g., Congratulations! You won a $1,000 gift card...")

if st.button("Predict"):
    if not input_sms.strip():
        st.warning("Please enter a message first!")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        
        # 3. Predict
        result = model.predict(vector_input)[0]
        
        # 4. Display result
        if result == 1:
            st.error("🚨 **This is a SPAM message!**")
        else:
            st.success("✅ **This is NOT spam (Ham).**")