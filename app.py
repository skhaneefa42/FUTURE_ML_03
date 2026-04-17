import streamlit as st
import pdfplumber
import re

st.title("📄 AI Resume Screening System")

# ----------- INPUT: Job Description -----------
st.subheader("📝 Enter Job Description")
job_desc = st.text_area("Paste Job Description here")

# ----------- FILE UPLOAD -----------
uploaded_files = st.file_uploader(
    "Upload Resumes (Multiple Allowed)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# ----------- TEXT EXTRACTION -----------
def extract_text(file):
    if file.type == "application/pdf":
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    else:
        return file.read().decode("utf-8")

# ----------- SKILL EXTRACTION -----------
skills_db = [
    "python", "java", "sql", "machine learning", "deep learning",
    "data science", "excel", "power bi", "tableau", "nlp",
    "tensorflow", "pandas", "numpy"
]

def extract_skills(text):
    found = []
    text = text.lower()
    for skill in skills_db:
        if skill in text:
            found.append(skill)
    return list(set(found))

# ----------- EXPERIENCE EXTRACTION -----------
def extract_experience(text):
    matches = re.findall(r'(\d+)\+?\s*years', text.lower())
    if matches:
        return max([int(x) for x in matches])
    return 0

# ----------- SCORING FUNCTION -----------
def calculate_score(resume_text, jd_text):
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    matched = list(set(resume_skills) & set(jd_skills))
    missing = list(set(jd_skills) - set(resume_skills))

    skill_score = (len(matched) / len(jd_skills)) * 100 if jd_skills else 0

    experience = extract_experience(resume_text)
    exp_score = min(experience * 10, 100)

    final_score = (skill_score * 0.7) + (exp_score * 0.3)

    return {
        "matched": matched,
        "missing": missing,
        "skill_score": round(skill_score, 2),
        "experience": experience,
        "final_score": round(final_score, 2)
    }

# ----------- PROCESSING -----------
if uploaded_files and job_desc:

    results = []

    for file in uploaded_files:
        text = extract_text(file)
        result = calculate_score(text, job_desc)

        results.append({
            "name": file.name,
            **result
        })

    # ----------- RANKING -----------
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    st.subheader("🏆 Candidate Ranking")

    for i, res in enumerate(results, 1):
        st.markdown(f"### {i}. {res['name']}")

        st.write(f"✅ Final Score: **{res['final_score']} / 100**")
        st.write(f"📊 Skill Match Score: {res['skill_score']}%")
        st.write(f"💼 Experience: {res['experience']} years")

        st.write("🟢 Matched Skills:", res["matched"])
        st.write("🔴 Missing Skills:", res["missing"])

        if res["final_score"] > 70:
            st.success("Excellent Candidate 🚀")
        elif res["final_score"] > 50:
            st.info("Good Candidate 👍")
        else:
            st.warning("Needs Improvement ⚠️")

        st.divider()

else:
    st.info("Please upload resumes and enter job description")