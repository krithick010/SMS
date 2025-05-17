import streamlit as st
import pickle
import nltk
import base64
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Download NLTK resources if not already present
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('punkt')

download_nltk_resources()
ps = PorterStemmer()

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if(i.isalnum()):
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# Custom CSS to improve the design
def add_bg_and_styling():
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7ff;
    }
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1px solid #cccccc;
        padding: 15px;
        font-size: 16px;
    }
    .spam-result {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        font-weight: bold;
        text-align: center;
    }
    .spam {
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #ef9a9a;
    }
    .not-spam {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #a5d6a7;
    }
    h1 {
        color: #1565c0;
        text-align: center;
        margin-bottom: 30px;
    }
    .description {
        padding: 15px;
        background-color: #e3f2fd;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #bbdefb;
    }
    </style>
    """, unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        tfidf = pickle.load(open('vectorizerl.pkl', 'rb'))
        model = pickle.load(open('modepl.pkl', 'rb'))
        return tfidf, model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def main():
    add_bg_and_styling()
    
    # App header
    st.title("SMS Spam Classifier")
    
    # App description
    st.markdown('<div class="description">This application analyzes text messages and classifies them as spam or legitimate. Enter your message below to check if it\'s spam.</div>', unsafe_allow_html=True)
    
    # Input area with improved styling
    input_text = st.text_area("Enter the message to analyze", height=150, key="message_input")
    
    # Create columns for button alignment
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("Analyze Message", use_container_width=True)
    
    # Load models
    tfidf, model = load_models()
    
    if tfidf is not None and model is not None and analyze_button and input_text:
        with st.spinner("Analyzing..."):
            # Preprocess and predict
            text = transform_text(input_text)
            vector = tfidf.transform([text])
            result = model.predict(vector)[0]
            
            # Display result with styling
            if result == 1:
                st.markdown('<div class="spam-result spam">SPAM DETECTED ⚠️</div>', unsafe_allow_html=True)
                st.warning("This message appears to be spam. Be cautious!")
            else:
                st.markdown('<div class="spam-result not-spam">NOT SPAM ✓</div>', unsafe_allow_html=True)
                st.success("This message appears to be legitimate.")
    
    # Footer information
    st.markdown("---")
    st.markdown("**How it works**: The classifier uses machine learning to analyze text patterns commonly found in spam messages.")

if __name__ == "__main__":
    main()
