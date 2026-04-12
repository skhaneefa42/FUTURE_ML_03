# ============================================================
# run.py — Application Entry Point
# ============================================================
# This is the file you run to start the entire application.
# Usage: python run.py
#
# It sets up the Flask development server and launches the app.
# ============================================================

import os
import sys

# Add the project root to Python path
# This ensures imports like "from backend.app import app" work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app import app

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🧠 Resume Screening & Ranking System")
    print("=" * 60)
    print("  📡 Server: http://localhost:5000")
    print("  📂 Uploads: ./uploads/")
    print("  🔗 API Docs: See README.md")
    print("=" * 60)
    print("\n  Open http://localhost:5000 in your browser!\n")

    # debug=True enables:
    # - Detailed error messages in browser
    # - Auto-reloading when you change code
    # - (Never use debug=True in production!)
    app.run(
        host="0.0.0.0",   # Accept connections from any network interface
        port=5000,         # Port number
        debug=True         # Development mode
    )
