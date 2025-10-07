import os
import sys
from pathlib import Path
import tempfile
import io

# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd

from src.models.user import db
from src.routes.user import user_bp
from src.routes.fairness_api import fairness_bp


# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(__file__), "dist"),
)
app.config["SECRET_KEY"] = "asdf#FGSgvasgf$5$WGT"
# Increase if you want to allow larger files (20 MB here)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# Enable CORS for all routes
CORS(app)

# -----------------------------------------------------------------------------
# Cross-platform temp directory for uploads (NO hardcoded '/tmp')
# -----------------------------------------------------------------------------
TMPDIR = Path(tempfile.gettempdir()) / "aift_uploads"
TMPDIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Blueprints (registered to handle all API routes including upload)
# -----------------------------------------------------------------------------
app.register_blueprint(user_bp, url_prefix="/api")
app.register_blueprint(fairness_bp, url_prefix="/api/fairness")

# -----------------------------------------------------------------------------
# (Optional) database (your original config)
# -----------------------------------------------------------------------------
# Create database directory if it doesn't exist
db_dir = os.path.join(os.path.dirname(__file__), 'database')
os.makedirs(db_dir, exist_ok=True)

app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(db_dir, 'app.db')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)
with app.app_context():
    db.create_all()


# -----------------------------------------------------------------------------
# Health check endpoint
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    return jsonify({
        "message": "AI Fairness Toolkit API",
        "status": "running",
        "endpoints": {
            "health": "/api/fairness/health",
            "demo_data": "/api/fairness/datasets/compas",
            "upload": "/api/fairness/upload",
            "analyze": "/api/fairness/analyze"
        }
    })


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 5000))
    # 0.0.0.0 makes it reachable from external requests
    app.run(host="0.0.0.0", port=port, debug=False)
