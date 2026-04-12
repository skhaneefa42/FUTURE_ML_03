import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("📄 Resume Screener")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("Enter Job Description")

def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

if uploaded_file and job_desc:
    resume_text = extract_text(uploaded_file)

    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume_text, job_desc])

    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    st.success(f"Match Score: {round(score*100,2)}%")