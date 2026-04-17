import streamlit as st

st.title("Resume Screening System")

uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "txt"])

if uploaded_file is not None:
    st.success("File uploaded successfully 🎉")
    st.write("File name:", uploaded_file.name)