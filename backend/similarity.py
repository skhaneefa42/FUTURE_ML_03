# ============================================================
# backend/similarity.py
# ============================================================
# PURPOSE: Compute similarity between resumes and job descriptions.
#
# CORE CONCEPTS:
#
# 1. TF-IDF (Term Frequency - Inverse Document Frequency):
#    - Converts text into numerical vectors
#    - Words that appear often in a SPECIFIC document get high scores
#    - Words that appear in EVERY document (common) get low scores
#    - "Python" in a resume about Python is more meaningful than "the"
#
# 2. COSINE SIMILARITY:
#    - Measures the ANGLE between two vectors (not the length)
#    - Score of 1.0 = documents are identical
#    - Score of 0.0 = documents share no words
#    - Score of 0.7+ = strong match
#    - Think of it as: "how much do two documents point in the same direction?"
#
#    Visual analogy:
#    Job Description → [0.3, 0.8, 0.1, 0.5, ...]  (a vector in N-dimensional space)
#    Resume A       → [0.3, 0.7, 0.1, 0.4, ...]  (very similar direction)
#    Resume B       → [0.1, 0.1, 0.9, 0.0, ...]  (very different direction)
#    cos(JD, A) = 0.98  ← very close!
#    cos(JD, B) = 0.15  ← very different!
# ============================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple, Dict


# ---- Global vectorizer instance ----
# We create ONE vectorizer that we fit on all documents together.
# This ensures all documents use the same vocabulary/features.
_vectorizer = None


def build_tfidf_vectorizer(documents: List[str]) -> TfidfVectorizer:
    """
    Create and train (fit) a TF-IDF vectorizer on a collection of documents.

    WHAT IS FITTING?
    When we "fit" the vectorizer, it:
    1. Reads all the documents
    2. Builds a vocabulary of all unique words
    3. Computes IDF (how rare/common each word is across documents)

    After fitting, the vectorizer can convert any text into a numerical vector
    using that vocabulary.

    Args:
        documents: List of preprocessed text strings

    Returns:
        Fitted TfidfVectorizer object
    """
    vectorizer = TfidfVectorizer(
        # ngram_range=(1, 2) means we consider:
        # - single words ("python", "docker")
        # - word pairs ("machine learning", "deep learning")
        ngram_range=(1, 2),

        # Ignore words that appear in >85% of documents (too common)
        max_df=0.85,

        # Ignore words that appear in <1 document (too rare to be useful)
        min_df=1,

        # Use sublinear scaling: replaces TF with 1 + log(TF)
        # This prevents very frequent terms from dominating the score
        sublinear_tf=True,

        # Limit vocabulary to top 10,000 most important terms
        # (prevents memory issues with huge vocabularies)
        max_features=10000
    )

    # Fit the vectorizer to learn the vocabulary from our documents
    vectorizer.fit(documents)

    print(f"✅ TF-IDF vectorizer built with {len(vectorizer.vocabulary_)} unique terms")
    return vectorizer


def compute_similarity(resume_text: str, jd_text: str) -> float:
    """
    Compute cosine similarity between ONE resume and ONE job description.

    This is used when analyzing a single resume against a single JD.

    STEP BY STEP:
    1. Put both texts in a list
    2. Fit TF-IDF on both (so they share vocabulary)
    3. Transform both into vectors
    4. Compute cosine similarity
    5. Return as a score between 0 and 1

    Args:
        resume_text: Preprocessed resume text
        jd_text: Preprocessed job description text

    Returns:
        Float between 0.0 and 1.0 (similarity score)
    """
    if not resume_text or not jd_text:
        return 0.0

    # Put both texts in a list (vectorizer works on collections)
    documents = [jd_text, resume_text]

    # Build and fit vectorizer on both documents
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(documents)

    # tfidf_matrix[0] = JD vector
    # tfidf_matrix[1] = Resume vector
    # cosine_similarity returns a 2D array [[similarity]]
    # We get [0][1] to extract the single similarity score
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # Round to 4 decimal places for clean output
    return round(float(similarity_score), 4)


def rank_resumes(resumes: List[Dict], jd_text: str) -> List[Dict]:
    """
    Rank multiple resumes against a single job description.

    This is the MAIN RANKING FUNCTION.

    ALGORITHM:
    1. Combine JD + all resumes into one corpus
    2. Fit ONE TF-IDF vectorizer on the entire corpus
       (Important: all documents must share the same vocabulary!)
    3. Transform each document into a vector
    4. Compute cosine similarity of each resume vector vs JD vector
    5. Sort resumes by similarity score (highest first)
    6. Return ranked list with scores and rank positions

    Args:
        resumes: List of dicts, each with keys:
                 - "name": candidate name or filename
                 - "text": preprocessed resume text
                 - "raw_text": original resume text (for display)
        jd_text: Preprocessed job description text

    Returns:
        Sorted list of resume dicts with added "score" and "rank" fields
    """
    if not resumes:
        return []

    if not jd_text:
        print("⚠️  Empty job description provided")
        return resumes

    # ---- Step 1: Build the corpus ----
    # Corpus = collection of all text documents we'll analyze
    # First document is ALWAYS the job description (index 0)
    corpus = [jd_text] + [r["text"] for r in resumes]

    print(f"📊 Building TF-IDF matrix for {len(resumes)} resumes...")

    # ---- Step 2: Fit TF-IDF on the full corpus ----
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=10000,
        max_df=0.85,
        min_df=1
    )

    # fit_transform = fit + transform in one step
    # Returns a sparse matrix of shape (n_documents, n_features)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # ---- Step 3: Extract vectors ----
    # JD vector = first row of the matrix (index 0)
    jd_vector = tfidf_matrix[0:1]

    # Resume vectors = all rows from index 1 onwards
    resume_vectors = tfidf_matrix[1:]

    # ---- Step 4: Compute cosine similarities ----
    # cosine_similarity returns shape (1, n_resumes)
    # We flatten it to a 1D array for easier handling
    similarities = cosine_similarity(jd_vector, resume_vectors).flatten()

    print(f"✅ Similarity scores computed: {similarities}")

    # ---- Step 5: Attach scores to resume dicts ----
    scored_resumes = []
    for i, resume in enumerate(resumes):
        resume_with_score = resume.copy()  # Don't modify original
        resume_with_score["score"] = round(float(similarities[i]), 4)
        resume_with_score["score_percentage"] = round(float(similarities[i]) * 100, 1)
        scored_resumes.append(resume_with_score)

    # ---- Step 6: Sort by score (highest first) ----
    ranked = sorted(scored_resumes, key=lambda x: x["score"], reverse=True)

    # ---- Step 7: Add rank positions (1-indexed) ----
    for rank, resume in enumerate(ranked, start=1):
        resume["rank"] = rank

        # Assign a label based on score
        score = resume["score"]
        if score >= 0.7:
            resume["match_label"] = "Excellent Match 🏆"
        elif score >= 0.5:
            resume["match_label"] = "Good Match ✅"
        elif score >= 0.3:
            resume["match_label"] = "Partial Match 🔶"
        else:
            resume["match_label"] = "Low Match ❌"

    print(f"✅ Ranking complete! Top candidate: {ranked[0]['name']} ({ranked[0]['score_percentage']}%)")

    return ranked


def get_top_tfidf_terms(text: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Find the most important terms in a document using TF-IDF.
    Useful for summarizing what a resume or JD is focused on.

    Args:
        text: Preprocessed text
        top_n: How many top terms to return

    Returns:
        List of (term, score) tuples sorted by importance
    """
    if not text:
        return []

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform([text])

    # Get feature names (the vocabulary)
    feature_names = vectorizer.get_feature_names_out()

    # Get TF-IDF scores for our document
    scores = tfidf_matrix.toarray()[0]

    # Pair terms with their scores
    term_scores = list(zip(feature_names, scores))

    # Sort by score (highest first) and take top N
    top_terms = sorted(term_scores, key=lambda x: x[1], reverse=True)[:top_n]

    return [(term, round(score, 4)) for term, score in top_terms if score > 0]


# ---- QUICK TEST ----
if __name__ == "__main__":
    # Sample preprocessed texts
    jd = "python machine learning deep learning tensorflow pytorch docker kubernetes aws cloud"
    resume_a = "python tensorflow deep learning neural network pytorch docker kubernetes aws data science"
    resume_b = "java spring boot hibernate mysql backend developer microservices rest api"
    resume_c = "python data analysis pandas numpy matplotlib sql postgresql statistics"

    resumes = [
        {"name": "Alice (ML Engineer)", "text": resume_a, "raw_text": resume_a},
        {"name": "Bob (Java Dev)", "text": resume_b, "raw_text": resume_b},
        {"name": "Carol (Data Analyst)", "text": resume_c, "raw_text": resume_c},
    ]

    print("=== Similarity Ranking Demo ===\n")
    ranked = rank_resumes(resumes, jd)

    for r in ranked:
        print(f"Rank #{r['rank']}: {r['name']}")
        print(f"  Score: {r['score_percentage']}% — {r['match_label']}")
        print()

    # Show top TF-IDF terms
    print("Top terms in JD:", get_top_tfidf_terms(jd, 5))
