# ============================================================
# backend/app.py
# ============================================================
# PURPOSE: Flask REST API — the web server that connects our
#          ML backend to the frontend.
#
# WHAT IS A REST API?
# REST API = a set of URLs (endpoints) that accept requests and return data.
# Our frontend sends requests to these URLs, and Flask handles them.
#
# HOW FLASK WORKS:
# 1. We define "routes" using @app.route("/some/path")
# 2. Each route has a function that runs when that URL is visited
# 3. The function processes the request and returns a JSON response
#
# ENDPOINTS:
#   GET  /                      → Serve the HTML frontend
#   POST /upload_resume         → Upload a resume file
#   POST /analyze               → Analyze one resume vs JD
#   POST /rank_candidates       → Rank all uploaded resumes
#   GET  /list_uploads          → Show all uploaded files
#   DELETE /clear_uploads       → Delete all uploaded files
# ============================================================

import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import our ML pipeline modules
from backend.ranker import screen_and_rank, analyze_single_resume

# ============================================================
# ---- CONFIGURATION ----
# ============================================================

# Create Flask app instance
# __name__ tells Flask where to find templates and static files
app = Flask(__name__, static_folder="../frontend", static_url_path="")

# Enable CORS (Cross-Origin Resource Sharing)
# This allows our HTML frontend (running at localhost:5000) to make
# API calls to our Flask server. Without this, browsers block the requests.
CORS(app)

# ---- Upload Settings ----
UPLOAD_FOLDER = "uploads"                          # Where to save uploaded resumes
ALLOWED_EXTENSIONS = {"pdf", "txt"}                # Only allow these file types
MAX_FILE_SIZE = 10 * 1024 * 1024                   # 10 MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Tell Flask about our settings
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


# ============================================================
# ---- HELPER FUNCTIONS ----
# ============================================================

def allowed_file(filename: str) -> bool:
    """
    Check if an uploaded file has an allowed extension.

    Example:
        allowed_file("resume.pdf")  → True
        allowed_file("resume.docx") → False
        allowed_file("resume")      → False
    """
    # Check if filename has an extension and if that extension is in our allowed set
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def success_response(data: dict, message: str = "Success") -> tuple:
    """
    Standard success response format.
    Returns JSON with status, message, and data.
    """
    return jsonify({
        "status": "success",
        "message": message,
        "data": data
    }), 200


def error_response(message: str, code: int = 400) -> tuple:
    """
    Standard error response format.
    """
    return jsonify({
        "status": "error",
        "message": message
    }), code


# ============================================================
# ---- ROUTES (API ENDPOINTS) ----
# ============================================================

@app.route("/")
def serve_frontend():
    """
    Serve the main HTML page.
    When user visits http://localhost:5000, they get our frontend.
    """
    return send_from_directory("../frontend", "index.html")


# ------------------------------------------------------------
# ENDPOINT 1: Upload Resume
# POST /upload_resume
# ------------------------------------------------------------
@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    """
    Handle resume file uploads.

    HOW IT WORKS:
    1. Client sends a POST request with the file as form-data
    2. We validate the file (exists? right format? not too big?)
    3. We save it to the uploads/ folder
    4. We return the saved file path

    REQUEST FORMAT:
        POST /upload_resume
        Content-Type: multipart/form-data
        Body: { "file": <resume.pdf> }

    RESPONSE:
        {
            "status": "success",
            "data": {
                "filename": "john_doe_resume.pdf",
                "path": "uploads/john_doe_resume.pdf",
                "message": "File uploaded successfully"
            }
        }
    """
    # ---- Check if a file was included in the request ----
    if "file" not in request.files:
        return error_response("No file provided. Please attach a resume file.")

    file = request.files["file"]

    # ---- Check if a filename was provided ----
    if file.filename == "":
        return error_response("No file selected. Please choose a resume file.")

    # ---- Validate file extension ----
    if not allowed_file(file.filename):
        return error_response(
            f"File type not supported. Please upload a PDF or TXT file. "
            f"Got: .{file.filename.rsplit('.', 1)[-1]}"
        )

    # ---- Secure the filename ----
    # secure_filename() sanitizes the filename to prevent security issues
    # Example: "../../../etc/passwd" → "etc_passwd"
    filename = secure_filename(file.filename)

    # ---- Save the file ----
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    print(f"📤 File uploaded: {filename} ({os.path.getsize(file_path)} bytes)")

    return success_response({
        "filename": filename,
        "path": file_path,
        "size_bytes": os.path.getsize(file_path),
        "message": f"'{filename}' uploaded successfully!"
    })


# ------------------------------------------------------------
# ENDPOINT 2: Analyze Single Resume
# POST /analyze
# ------------------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Analyze a single resume against a job description.

    Use this when you want to see the detailed analysis for one resume.

    REQUEST FORMAT:
        POST /analyze
        Content-Type: application/json
        Body: {
            "resume_filename": "john_doe.pdf",
            "job_description": "We're looking for a Python developer..."
        }

    RESPONSE:
        {
            "status": "success",
            "data": {
                "candidate_name": "John Doe",
                "score": 0.72,
                "score_percentage": 72.0,
                "match_label": "Excellent Match 🏆",
                "candidate_skills": ["python", "docker", ...],
                "skill_gap": {
                    "matching": [...],
                    "missing": [...],
                    "extra": [...],
                    "match_percentage": 65.0
                }
            }
        }
    """
    # ---- Parse JSON request body ----
    data = request.get_json()

    if not data:
        return error_response("Request body must be JSON with 'resume_filename' and 'job_description'")

    resume_filename = data.get("resume_filename", "").strip()
    job_description = data.get("job_description", "").strip()

    # ---- Validate inputs ----
    if not resume_filename:
        return error_response("'resume_filename' is required")

    if not job_description or len(job_description) < 50:
        return error_response("'job_description' is required and must be at least 50 characters")

    # ---- Build file path ----
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(resume_filename))

    if not os.path.exists(file_path):
        return error_response(
            f"Resume file '{resume_filename}' not found. "
            f"Please upload it first using /upload_resume"
        )

    # ---- Run the analysis pipeline ----
    try:
        result = analyze_single_resume(file_path, job_description)

        if "error" in result:
            return error_response(result["error"])

        return success_response(result, "Analysis complete!")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return error_response(f"Analysis failed: {str(e)}", code=500)


# ------------------------------------------------------------
# ENDPOINT 3: Rank Multiple Candidates
# POST /rank_candidates
# ------------------------------------------------------------
@app.route("/rank_candidates", methods=["POST"])
def rank_candidates():
    """
    Rank ALL uploaded resumes against a job description.

    This is the MAIN FEATURE — takes all uploaded resumes and
    returns them sorted by relevance to the job description.

    REQUEST FORMAT:
        POST /rank_candidates
        Content-Type: application/json
        Body: {
            "job_description": "Looking for a Senior ML Engineer...",
            "filenames": ["alice.pdf", "bob.txt", "carol.pdf"]  ← optional
        }

    RESPONSE:
        {
            "status": "success",
            "data": {
                "job_description": {...},
                "total_candidates": 3,
                "ranked_candidates": [
                    {
                        "rank": 1,
                        "name": "Alice",
                        "score": 0.82,
                        "score_percentage": 82.0,
                        "match_label": "Excellent Match 🏆",
                        "skills": [...],
                        "skill_gap": {...}
                    },
                    ...
                ]
            }
        }
    """
    data = request.get_json()

    if not data:
        return error_response("Request body must be JSON with 'job_description'")

    job_description = data.get("job_description", "").strip()
    # Optional: specific filenames to rank (default: rank all uploaded)
    filenames = data.get("filenames", None)

    # ---- Validate job description ----
    if not job_description or len(job_description) < 50:
        return error_response(
            "'job_description' is required and must be at least 50 characters. "
            "Please provide a detailed job description."
        )

    # ---- Get list of resumes to rank ----
    if filenames:
        # Use the specified filenames
        resume_paths = []
        for fname in filenames:
            path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(fname))
            if os.path.exists(path):
                resume_paths.append(path)
            else:
                print(f"⚠️  File not found: {fname}")
    else:
        # Use ALL files in the uploads folder
        resume_paths = [
            os.path.join(app.config["UPLOAD_FOLDER"], f)
            for f in os.listdir(app.config["UPLOAD_FOLDER"])
            if allowed_file(f)
        ]

    # ---- Check we have resumes to rank ----
    if not resume_paths:
        return error_response(
            "No resumes found. Please upload resumes first using /upload_resume"
        )

    print(f"\n🎯 Ranking {len(resume_paths)} resumes...")

    # ---- Run the ranking pipeline ----
    try:
        results = screen_and_rank(resume_paths, job_description)

        if "error" in results:
            return error_response(results["error"])

        return success_response(results, f"Ranked {results['total_candidates']} candidates successfully!")

    except Exception as e:
        print(f"❌ Ranking failed: {e}")
        import traceback
        traceback.print_exc()
        return error_response(f"Ranking failed: {str(e)}", code=500)


# ------------------------------------------------------------
# ENDPOINT 4: List Uploaded Files
# GET /list_uploads
# ------------------------------------------------------------
@app.route("/list_uploads", methods=["GET"])
def list_uploads():
    """
    Return a list of all uploaded resume files.
    Used by the frontend to show which files are ready.
    """
    try:
        files = []
        upload_dir = app.config["UPLOAD_FOLDER"]

        for filename in os.listdir(upload_dir):
            if allowed_file(filename):
                file_path = os.path.join(upload_dir, filename)
                files.append({
                    "filename": filename,
                    "size_bytes": os.path.getsize(file_path),
                    "size_kb": round(os.path.getsize(file_path) / 1024, 1)
                })

        return success_response({
            "files": files,
            "count": len(files)
        })

    except Exception as e:
        return error_response(str(e), 500)


# ------------------------------------------------------------
# ENDPOINT 5: Clear All Uploads
# DELETE /clear_uploads
# ------------------------------------------------------------
@app.route("/clear_uploads", methods=["DELETE"])
def clear_uploads():
    """
    Delete all uploaded resume files.
    Used to reset the system for a new screening session.
    """
    try:
        upload_dir = app.config["UPLOAD_FOLDER"]
        deleted = []

        for filename in os.listdir(upload_dir):
            if allowed_file(filename):
                os.remove(os.path.join(upload_dir, filename))
                deleted.append(filename)

        return success_response({
            "deleted": deleted,
            "count": len(deleted),
            "message": f"Deleted {len(deleted)} files"
        })

    except Exception as e:
        return error_response(str(e), 500)


# ============================================================
# ---- ERROR HANDLERS ----
# ============================================================

@app.errorhandler(404)
def not_found(e):
    return error_response("Endpoint not found", 404)


@app.errorhandler(413)
def file_too_large(e):
    return error_response("File too large. Maximum size is 10MB", 413)


@app.errorhandler(500)
def server_error(e):
    return error_response("Internal server error", 500)
