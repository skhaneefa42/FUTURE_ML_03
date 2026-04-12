# 🧠 Resume Screening & Ranking System

An ML-powered system that reads resumes, extracts skills, compares them to job descriptions,
and ranks candidates using NLP + TF-IDF cosine similarity.

---

## 📁 Folder Structure

```
resume_screener/
├── README.md                  ← This file
├── requirements.txt           ← All Python dependencies
├── run.py                     ← Entry point to launch the app
│
├── backend/
│   ├── __init__.py
│   ├── app.py                 ← Flask REST API (routes)
│   ├── preprocessor.py        ← Text cleaning & NLP
│   ├── skill_extractor.py     ← Skill extraction logic
│   ├── similarity.py          ← TF-IDF + cosine similarity
│   ├── pdf_parser.py          ← PDF → text extraction
│   └── ranker.py              ← Ranking + skill gap logic
│
├── frontend/
│   └── index.html             ← Single-page UI (HTML/CSS/JS)
│
├── notebooks/
│   └── ml_development.ipynb  ← Jupyter notebook (exploration)
│
├── sample_data/
│   ├── sample_resume.txt      ← Example resume text
│   └── sample_jd.txt          ← Example job description
│
└── uploads/                   ← Uploaded resumes stored here (auto-created)
```

---

## 🚀 How to Run Locally

### Step 1: Clone or create the project folder
```bash
mkdir resume_screener && cd resume_screener
```

### Step 2: Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download spaCy English model
```bash
python -m spacy download en_core_web_sm
```

### Step 5: Run the app
```bash
python run.py
```

### Step 6: Open in browser
```
http://localhost:5000
```

---

## 🔌 API Endpoints

| Method | Endpoint            | Description                        |
|--------|---------------------|------------------------------------|
| POST   | `/upload_resume`    | Upload a PDF or TXT resume         |
| POST   | `/analyze`          | Analyze one resume vs a job desc   |
| POST   | `/rank_candidates`  | Rank multiple uploaded resumes     |

---

## 📦 Dataset

You can use the **Kaggle Resume Dataset**:
- URL: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
- Download `Resume.csv` and place it in `sample_data/`
- The notebook shows how to load and process it

---

## 🧠 How It Works

1. **PDF Parsing** → Extract raw text from uploaded PDFs
2. **Preprocessing** → Lowercase, remove stopwords & punctuation
3. **Skill Extraction** → Match against a curated skills keyword list
4. **TF-IDF Vectorization** → Convert text to numerical vectors
5. **Cosine Similarity** → Measure how close each resume is to the job description
6. **Ranking** → Sort candidates by similarity score
7. **Skill Gap** → Find skills in JD but missing from resume
