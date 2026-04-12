# ============================================================
# backend/ranker.py
# ============================================================
# PURPOSE: Orchestrate the full resume screening pipeline.
#
# This is the "conductor" module. It calls all other modules in order:
#   pdf_parser → preprocessor → skill_extractor → similarity
#
# Think of it like a factory pipeline:
#   📄 Raw Resume File
#       ↓  pdf_parser.py (extract text)
#   📝 Raw Text
#       ↓  preprocessor.py (clean & normalize)
#   🔤 Clean Text
#       ↓  skill_extractor.py (find skills)
#   🛠️  Skills List
#       ↓  similarity.py (compute score vs JD)
#   📊 Score + Ranking
# ============================================================

import os
from typing import List, Dict, Optional

# Import our own modules
from backend.pdf_parser import parse_resume
from backend.preprocessor import preprocess
from backend.skill_extractor import extract_all_skills, compute_skill_gap
from backend.similarity import rank_resumes, compute_similarity


def process_single_resume(file_path: str, candidate_name: Optional[str] = None) -> Dict:
    """
    Process one resume file through the complete pipeline.

    Args:
        file_path: Path to the resume file (PDF or TXT)
        candidate_name: Optional name override (default: use filename)

    Returns:
        Dict with all extracted information about the candidate:
        {
            "name": "John Doe",
            "file_path": "/uploads/john_doe.pdf",
            "raw_text": "Full original text...",
            "processed_text": "cleaned tokenized text...",
            "skills": {"python", "docker", ...},
            "skill_categories": {"Programming": ["python"], ...},
            "skill_count": 12,
            "status": "success" or "error",
            "error": "..." (only if status is "error")
        }
    """
    # Derive candidate name from filename if not provided
    if not candidate_name:
        # Get just the filename without path and extension
        # e.g., "/uploads/john_doe_resume.pdf" → "john_doe_resume"
        candidate_name = os.path.splitext(os.path.basename(file_path))[0]
        # Replace underscores/hyphens with spaces for cleaner display
        candidate_name = candidate_name.replace("_", " ").replace("-", " ").title()

    print(f"\n📄 Processing: {candidate_name}")

    # ---- Step 1: Extract raw text from file ----
    raw_text = parse_resume(file_path)

    if not raw_text:
        print(f"  ❌ Could not extract text from {file_path}")
        return {
            "name": candidate_name,
            "file_path": file_path,
            "status": "error",
            "error": "Could not extract text from resume file"
        }

    print(f"  ✅ Extracted {len(raw_text)} characters of text")

    # ---- Step 2: Preprocess the text ----
    processed_text = preprocess(raw_text)
    print(f"  ✅ Preprocessed to {len(processed_text.split())} tokens")

    # ---- Step 3: Extract skills ----
    skill_result = extract_all_skills(raw_text)  # Use raw text for skill extraction
    # (Raw text is better for skills because skill names might be removed during preprocessing)
    print(f"  ✅ Found {skill_result['count']} skills")

    return {
        "name": candidate_name,
        "file_path": file_path,
        "raw_text": raw_text,
        "text": processed_text,          # Used by similarity.py for vectorization
        "skills": skill_result["skills"],
        "categories": skill_result["categories"],
        "skill_count": skill_result["count"],
        "status": "success"
    }


def process_job_description(jd_text: str) -> Dict:
    """
    Process a job description through the pipeline.

    Args:
        jd_text: Raw job description text (pasted by user)

    Returns:
        Dict with processed JD data
    """
    print(f"\n📋 Processing Job Description ({len(jd_text)} chars)...")

    # Preprocess the JD text
    processed_jd = preprocess(jd_text)

    # Extract required skills from JD
    skill_result = extract_all_skills(jd_text)

    print(f"  ✅ JD requires {skill_result['count']} skills")

    return {
        "raw_text": jd_text,
        "text": processed_jd,            # Used for similarity computation
        "skills": skill_result["skills"],
        "categories": skill_result["categories"],
        "skill_count": skill_result["count"]
    }


def screen_and_rank(
    resume_file_paths: List[str],
    job_description: str,
    candidate_names: Optional[List[str]] = None
) -> Dict:
    """
    MAIN PIPELINE: Screen and rank multiple resumes against a job description.

    This is the TOP-LEVEL function called by the API.

    FULL WORKFLOW:
    1. Process each resume (parse → clean → extract skills)
    2. Process job description
    3. Rank resumes using TF-IDF cosine similarity
    4. Compute skill gaps for each candidate
    5. Return complete ranked results

    Args:
        resume_file_paths: List of file paths to resume files
        job_description: Raw job description text
        candidate_names: Optional list of names matching file_paths

    Returns:
        {
            "job_description": {...},
            "total_candidates": 5,
            "ranked_candidates": [
                {
                    "rank": 1,
                    "name": "Alice",
                    "score": 0.78,
                    "score_percentage": 78.0,
                    "match_label": "Excellent Match 🏆",
                    "skills": {...},
                    "skill_gap": {
                        "matching": [...],
                        "missing": [...],
                        "extra": [...],
                        "match_percentage": 75.0
                    }
                },
                ...
            ]
        }
    """
    print("\n" + "="*50)
    print("🚀 Starting Resume Screening Pipeline")
    print("="*50)

    # ---- Step 1: Process each resume ----
    processed_resumes = []

    for i, file_path in enumerate(resume_file_paths):
        # Get candidate name if provided
        name = candidate_names[i] if candidate_names and i < len(candidate_names) else None

        # Process the resume
        resume_data = process_single_resume(file_path, name)

        # Only include successfully processed resumes
        if resume_data["status"] == "success":
            processed_resumes.append(resume_data)
        else:
            print(f"  ⚠️  Skipping {file_path}: {resume_data.get('error', 'Unknown error')}")

    if not processed_resumes:
        return {
            "error": "No resumes could be processed",
            "total_candidates": 0,
            "ranked_candidates": []
        }

    print(f"\n✅ Successfully processed {len(processed_resumes)}/{len(resume_file_paths)} resumes")

    # ---- Step 2: Process job description ----
    jd_data = process_job_description(job_description)

    # ---- Step 3: Rank resumes using TF-IDF similarity ----
    print("\n📊 Computing similarity scores and ranking...")
    ranked_resumes = rank_resumes(processed_resumes, jd_data["text"])

    # ---- Step 4: Compute skill gaps ----
    print("\n🔍 Computing skill gaps...")
    for resume in ranked_resumes:
        # Skill gap = what's in JD but missing from resume
        skill_gap = compute_skill_gap(
            resume_skills=resume["skills"],    # Candidate's skills
            jd_skills=jd_data["skills"]        # Required skills
        )
        resume["skill_gap"] = skill_gap

        # Convert skills set to sorted list for JSON serialization
        # (JSON can't serialize Python sets)
        resume["skills"] = sorted(resume["skills"])

    print("\n" + "="*50)
    print("✅ Screening Complete!")
    print("="*50)

    return {
        "job_description": {
            "raw_text": jd_data["raw_text"][:500] + "...",  # Preview only
            "required_skills": sorted(jd_data["skills"]),
            "required_skills_count": jd_data["skill_count"],
            "categories": jd_data["categories"]
        },
        "total_candidates": len(ranked_resumes),
        "ranked_candidates": ranked_resumes
    }


def analyze_single_resume(file_path: str, job_description: str) -> Dict:
    """
    Analyze just ONE resume vs ONE job description.
    Used by the /analyze endpoint in the API.

    Args:
        file_path: Path to resume file
        job_description: Job description text

    Returns:
        Complete analysis dict for this one candidate
    """
    # Process resume and JD
    resume_data = process_single_resume(file_path)
    jd_data = process_job_description(job_description)

    if resume_data["status"] != "success":
        return {"error": resume_data.get("error", "Failed to process resume")}

    # Compute similarity score
    score = compute_similarity(resume_data["text"], jd_data["text"])

    # Compute skill gap
    skill_gap = compute_skill_gap(resume_data["skills"], jd_data["skills"])

    # Determine match label
    if score >= 0.7:
        match_label = "Excellent Match 🏆"
    elif score >= 0.5:
        match_label = "Good Match ✅"
    elif score >= 0.3:
        match_label = "Partial Match 🔶"
    else:
        match_label = "Low Match ❌"

    return {
        "candidate_name": resume_data["name"],
        "score": score,
        "score_percentage": round(score * 100, 1),
        "match_label": match_label,
        "candidate_skills": sorted(resume_data["skills"]),
        "skill_categories": resume_data["categories"],
        "required_skills": sorted(jd_data["skills"]),
        "skill_gap": skill_gap,
        "text_preview": resume_data["raw_text"][:300] + "..."
    }


# ---- QUICK TEST ----
if __name__ == "__main__":
    # Test with sample files
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    sample_jd = """
    We are looking for a Senior Python Developer with expertise in machine learning
    and deep learning. Requirements: Python, TensorFlow, PyTorch, scikit-learn,
    Docker, Kubernetes, AWS, REST API development, SQL, PostgreSQL.
    Nice to have: React, TypeScript, Spark, Hadoop.
    """

    sample_resumes = ["sample_data/sample_resume.txt"]

    results = screen_and_rank(sample_resumes, sample_jd)
    print(f"\n🏆 Results: {results['total_candidates']} candidates ranked")
