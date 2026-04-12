# ============================================================
# backend/skill_extractor.py
# ============================================================
# PURPOSE: Extract skills from resume and job description text.
#
# APPROACH: Keyword Matching
# We maintain a curated list of known technical and soft skills.
# We search the text for these keywords and return which ones appear.
#
# WHY NOT JUST USE NER (Named Entity Recognition)?
# spaCy's built-in NER is great for names, places, dates — but it
# doesn't specifically know "React.js" or "TensorFlow" are skills.
# Keyword matching is more reliable for technical skills.
#
# We also use spaCy for NOUN PHRASE extraction as a bonus
# to catch skills not in our list.
# ============================================================

import re
import spacy
from typing import Set, List, Dict

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None
    print("⚠️  spaCy model not loaded. Run: python -m spacy download en_core_web_sm")


# ============================================================
# 📚 SKILLS DATABASE
# A comprehensive dictionary of skills by category.
# Feel free to add more skills to any category!
# ============================================================

SKILLS_DATABASE = {

    # --- Programming Languages ---
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "c",
        "ruby", "php", "swift", "kotlin", "go", "golang", "rust", "scala",
        "r", "matlab", "perl", "bash", "shell", "powershell", "sql",
        "html", "css", "sass", "less"
    ],

    # --- Web Frameworks & Libraries ---
    "web_frameworks": [
        "react", "angular", "vue", "nextjs", "next.js", "nuxt",
        "django", "flask", "fastapi", "express", "nodejs", "node.js",
        "spring", "spring boot", "rails", "laravel", "asp.net",
        "jquery", "bootstrap", "tailwind", "webpack", "vite"
    ],

    # --- Machine Learning & AI ---
    "ml_ai": [
        "machine learning", "deep learning", "neural network",
        "natural language processing", "nlp", "computer vision",
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
        "xgboost", "lightgbm", "opencv", "huggingface", "transformers",
        "bert", "gpt", "llm", "reinforcement learning", "random forest",
        "gradient boosting", "svm", "support vector machine",
        "regression", "classification", "clustering", "pandas", "numpy"
    ],

    # --- Data Science & Analytics ---
    "data_science": [
        "data analysis", "data science", "data engineering",
        "data visualization", "statistics", "statistical analysis",
        "tableau", "power bi", "matplotlib", "seaborn", "plotly",
        "excel", "google analytics", "apache spark", "hadoop",
        "etl", "data warehouse", "data pipeline", "feature engineering",
        "a/b testing", "hypothesis testing"
    ],

    # --- Databases ---
    "databases": [
        "mysql", "postgresql", "postgres", "mongodb", "sqlite",
        "redis", "cassandra", "oracle", "sql server", "dynamodb",
        "elasticsearch", "firebase", "supabase", "prisma",
        "database design", "orm", "nosql", "relational database"
    ],

    # --- Cloud & DevOps ---
    "cloud_devops": [
        "aws", "amazon web services", "azure", "google cloud", "gcp",
        "docker", "kubernetes", "k8s", "terraform", "ansible",
        "jenkins", "github actions", "ci/cd", "continuous integration",
        "linux", "unix", "nginx", "apache", "microservices",
        "serverless", "lambda", "s3", "ec2", "cloud computing",
        "devops", "sre", "infrastructure"
    ],

    # --- Version Control & Tools ---
    "tools": [
        "git", "github", "gitlab", "bitbucket", "jira", "confluence",
        "slack", "vs code", "intellij", "eclipse", "postman",
        "swagger", "jupyter", "anaconda", "linux", "agile", "scrum",
        "kanban", "rest api", "graphql", "json", "xml", "yaml"
    ],

    # --- Soft Skills ---
    "soft_skills": [
        "leadership", "communication", "teamwork", "problem solving",
        "critical thinking", "time management", "collaboration",
        "project management", "mentoring", "agile methodology",
        "analytical", "detail oriented", "self motivated",
        "adaptability", "creativity", "presentation"
    ],

    # --- Security ---
    "security": [
        "cybersecurity", "penetration testing", "ethical hacking",
        "owasp", "encryption", "authentication", "oauth", "jwt",
        "ssl", "tls", "firewall", "network security", "siem"
    ],

    # --- Mobile Development ---
    "mobile": [
        "android", "ios", "react native", "flutter", "dart",
        "swift", "kotlin", "mobile development", "app development",
        "xcode", "android studio"
    ]
}

# Create a FLAT LIST of all skills (combining all categories)
# This makes searching easier — one list to check against
ALL_SKILLS = []
for category_skills in SKILLS_DATABASE.values():
    ALL_SKILLS.extend(category_skills)

# Remove duplicates and sort for consistency
ALL_SKILLS = sorted(set(ALL_SKILLS))


def extract_skills_by_keyword(text: str) -> Set[str]:
    """
    Extract skills by searching for known skill keywords in the text.

    METHOD: For each skill in our database, check if it appears in the text.
    We use word boundary matching to avoid false positives
    (e.g., "R" shouldn't match inside "for" or "her").

    Args:
        text: Raw or cleaned text from resume/job description

    Returns:
        Set of found skills (e.g., {"python", "machine learning", "docker"})
    """
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()

    found_skills = set()

    for skill in ALL_SKILLS:
        # Use regex word boundary (\b) to match whole words/phrases only
        # re.escape handles special chars in skill names like "c++" or "node.js"
        # re.IGNORECASE makes it case-insensitive (extra safety)
        pattern = r'\b' + re.escape(skill) + r'\b'

        if re.search(pattern, text_lower, re.IGNORECASE):
            found_skills.add(skill)

    return found_skills


def extract_noun_phrases_spacy(text: str) -> List[str]:
    """
    Use spaCy to extract noun phrases that might be skills not in our list.

    NOUN PHRASES = groups of words with a noun as the head
    Examples: "machine learning engineer", "cloud architecture", "data pipeline"

    This catches skills we might have missed in our keyword list.

    Args:
        text: Raw text

    Returns:
        List of noun phrases (potential skills)
    """
    if nlp is None:
        return []

    # Process with spaCy (limit to first 10,000 chars for speed)
    doc = nlp(text[:10000])

    # Extract noun chunks (built-in spaCy feature)
    noun_phrases = [
        chunk.text.lower().strip()
        for chunk in doc.noun_chunks
        if len(chunk.text.split()) <= 4  # Skip very long phrases
        and len(chunk.text) > 2          # Skip very short ones
    ]

    return noun_phrases


def get_skill_categories(skills: Set[str]) -> Dict[str, List[str]]:
    """
    Categorize found skills back into their groups.
    Useful for displaying a structured skills report.

    Args:
        skills: Set of found skill strings

    Returns:
        Dict mapping category name → list of skills in that category
    """
    categorized = {}

    for category, category_skills in SKILLS_DATABASE.items():
        # Find intersection — skills in both the found set and this category
        matched = [s for s in category_skills if s in skills]
        if matched:
            # Convert category name to readable format
            readable_name = category.replace("_", " ").title()
            categorized[readable_name] = matched

    return categorized


def extract_all_skills(text: str) -> Dict:
    """
    MAIN FUNCTION: Extract skills using all methods combined.

    Returns:
        Dict with:
        - "skills": Set of found skills
        - "categories": Skills organized by category
        - "count": Total number of skills found
    """
    # Method 1: Keyword matching (primary method — most reliable)
    keyword_skills = extract_skills_by_keyword(text)

    # Organize by category for display
    categorized = get_skill_categories(keyword_skills)

    return {
        "skills": keyword_skills,           # Set of all found skills
        "categories": categorized,          # Organized by type
        "count": len(keyword_skills)        # Total count
    }


def compute_skill_gap(resume_skills: Set[str], jd_skills: Set[str]) -> Dict:
    """
    Identify skills required by the job but missing from the resume.

    This is the SKILL GAP ANALYSIS — it tells the candidate:
    "You need to learn X, Y, Z to be more competitive for this role."

    Args:
        resume_skills: Skills found in the resume
        jd_skills: Skills required in the job description

    Returns:
        Dict with matching, missing, and extra skills
    """
    # Skills in JD that are also in resume (good matches!)
    matching_skills = resume_skills & jd_skills

    # Skills in JD but NOT in resume (the gap — what to learn)
    missing_skills = jd_skills - resume_skills

    # Skills in resume not mentioned in JD (extra — still good to have)
    extra_skills = resume_skills - jd_skills

    # Calculate match percentage
    if len(jd_skills) > 0:
        match_percentage = (len(matching_skills) / len(jd_skills)) * 100
    else:
        match_percentage = 0

    return {
        "matching": sorted(matching_skills),     # Skills the candidate has ✅
        "missing": sorted(missing_skills),       # Skills they need to learn ❌
        "extra": sorted(extra_skills),           # Bonus skills they have 🌟
        "match_percentage": round(match_percentage, 1)
    }


# ---- QUICK TEST ----
if __name__ == "__main__":
    sample_resume = """
    Python developer with 3 years of experience in machine learning and data analysis.
    Proficient in TensorFlow, scikit-learn, and PyTorch. Experience with Docker,
    Kubernetes, and AWS. Strong knowledge of SQL, MongoDB, and PostgreSQL.
    Built REST APIs using Flask and FastAPI. Familiar with Git, Jira, and Agile.
    """

    sample_jd = """
    We are looking for a Machine Learning Engineer with expertise in Python,
    TensorFlow, PyTorch, and deep learning. Must have experience with AWS,
    Docker, and Kubernetes. Knowledge of React and TypeScript is a plus.
    Strong communication skills required.
    """

    print("=== Skill Extraction Demo ===")

    resume_result = extract_all_skills(sample_resume)
    print(f"\n📄 Resume Skills ({resume_result['count']} found):")
    print(", ".join(sorted(resume_result["skills"])))

    jd_result = extract_all_skills(sample_jd)
    print(f"\n📋 JD Skills ({jd_result['count']} found):")
    print(", ".join(sorted(jd_result["skills"])))

    gap = compute_skill_gap(resume_result["skills"], jd_result["skills"])
    print(f"\n✅ Matching: {gap['matching']}")
    print(f"❌ Missing:  {gap['missing']}")
    print(f"📊 Match%:   {gap['match_percentage']}%")
