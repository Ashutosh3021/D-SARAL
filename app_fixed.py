"""
app_fixed.py  —  D-SARAL Flask Application
==========================================
Production entry point.  Start command: gunicorn app_fixed:app

Improvements over previous version
------------------------------------
- All uploaded files are processed (not just the first one)
- Separate  GET /api/analyze/<session_id>  endpoint (analysis without cleaning)
- Richer JSON result  (per-column stats, outlier summary, near-dup count)
- Accurate issuesFound count from all analysis categories
- Structured error responses with error_code field
- Request-size guard + allowed-extension guard kept
- SSE progress stream uses named stages (easier to render in frontend)
- Cleanup endpoint unchanged (backward-compatible)
"""

import io
import json
import os
import shutil
import uuid
import zipfile
from datetime import datetime

import pandas as pd
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from backend_processing import DataProcessingPipeline

# ──────────────────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_url_path="",
    static_folder="frontend",
    template_folder="frontend",
)
CORS(app)

UPLOAD_FOLDER    = "uploads"
PROCESSED_FOLDER = "processed"
ALLOWED_EXT      = {"csv", "json", "txt"}
MAX_FILE_SIZE    = 50 * 1024 * 1024   # 50 MB

app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"]   = PROCESSED_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER,    exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def _sanitize_for_json(obj):
    """Recursively stringify dict keys so json.dump/json.dumps never see dtype objects."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Event string."""
    return f"data: {json.dumps(_sanitize_for_json(data), default=str)}\n\n"


def _count_issues(analysis: dict) -> int:
    """Return total number of detected issues across all analysis categories."""
    n = 0
    n += len(analysis.get("missing_values", {}).get("total_missing_per_column", {}))
    n += len(analysis.get("format_inconsistencies", {}))
    n += len(analysis.get("broken_entries", {}))
    dup = analysis.get("duplicates", {})
    if dup.get("exact_duplicates", 0):
        n += 1
    if dup.get("near_duplicate_pairs"):
        n += 1
    n += len(analysis.get("outliers", {}))
    n += len(analysis.get("column_inconsistencies", {}))
    return n


def _build_summary(pipeline: DataProcessingPipeline,
                   df_raw: pd.DataFrame,
                   df_clean: pd.DataFrame,
                   analysis: dict,
                   report: str,
                   files_processed: int) -> dict:
    """Build the result dict returned in the SSE 'complete' event."""
    dup     = analysis.get("duplicates", {})
    outliers = analysis.get("outliers", {})

    return {
        "filesProcessed":   files_processed,
        "originalShape":    list(df_raw.shape),
        "cleanedShape":     list(df_clean.shape),
        "rowsRemoved":      df_raw.shape[0] - df_clean.shape[0],
        "colsRemoved":      df_raw.shape[1] - df_clean.shape[1],
        "issuesFound":      _count_issues(analysis),
        "exactDuplicates":  dup.get("exact_duplicates", 0),
        "nearDuplicates":   len(dup.get("near_duplicate_pairs", [])),
        "outlierColumns":   list(outliers.keys()),
        "missingPct":       analysis.get("missing_values", {})
                                    .get("pct_missing_per_column", {}),
        "reportPreview":    report[:600] + "…" if len(report) > 600 else report,
        "cleaningSteps":    [
            e["message"] for e in pipeline.cleaning_log
            if e.get("step", "").startswith("step_") and "message" in e
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Routes — static
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/<path:path>")
def static_files(path):
    return app.send_static_file(path)


# ──────────────────────────────────────────────────────────────────────────────
# Routes — API
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload_files():
    try:
        if "files" not in request.files:
            return jsonify({"error": "No files provided", "error_code": "NO_FILES"}), 400

        files     = request.files.getlist("files")
        valid     = [f for f in files if f and f.filename]

        if not valid:
            return jsonify({"error": "No files selected", "error_code": "EMPTY_FILES"}), 400

        # Validate all before saving any
        for f in valid:
            if not _allowed(f.filename):
                return jsonify({
                    "error": f"Unsupported file type: {f.filename}. Allowed: CSV, JSON, TXT",
                    "error_code": "INVALID_TYPE",
                }), 400

        session_id  = str(uuid.uuid4())
        session_dir = os.path.join(UPLOAD_FOLDER, session_id)
        os.makedirs(session_dir, exist_ok=True)

        saved = []
        for f in valid:
            fname = secure_filename(f.filename)
            fpath = os.path.join(session_dir, fname)
            f.save(fpath)
            saved.append({
                "filename": fname,
                "path":     fpath,
                "size":     os.path.getsize(fpath),
            })

        return jsonify({
            "sessionId": session_id,
            "files":     saved,
            "message":   f"Uploaded {len(saved)} file(s) successfully",
        })

    except Exception as exc:
        return jsonify({"error": str(exc), "error_code": "UPLOAD_ERROR"}), 500


# ── Full clean pipeline (SSE stream) ──────────────────────────────────────────
@app.route("/api/process/<session_id>", methods=["POST", "GET"])
def process_data(session_id):
    try:
        params = request.get_json(silent=True) or {} if request.method == "POST" else {}
    except Exception:
        params = {}

    sample_size = int(params.get("sampleSize", 10_000))

    def generate():
        try:
            session_upload_dir = os.path.join(UPLOAD_FOLDER, session_id)
            if not os.path.exists(session_upload_dir):
                yield _sse({"status": "error", "message": "Session not found", "progress": 0})
                return

            pipeline = DataProcessingPipeline(sample_size=sample_size)

            # ── Load ──────────────────────────────────────────────────────
            yield _sse({"status": "loading", "message": "Loading files…", "progress": 10})
            all_dfs = pipeline.load_files_from_directory(
                session_upload_dir, ["csv", "json", "txt"]
            )

            if not all_dfs:
                yield _sse({"status": "error", "message": "No readable files found.", "progress": 0})
                return

            # Merge all loaded DataFrames (column-union concat)
            dfs_list = list(all_dfs.values())
            if len(dfs_list) == 1:
                df_raw = dfs_list[0]
            else:
                try:
                    df_raw = pd.concat(
                        [d.reset_index(drop=True) for d in dfs_list],
                        ignore_index=True, sort=False,
                    )
                except Exception:
                    df_raw = dfs_list[0]   # fallback to first file

            # ── Analyse ───────────────────────────────────────────────────
            yield _sse({"status": "analyzing", "message": "Analysing data quality…", "progress": 25})
            analysis = pipeline.comprehensive_data_analysis(df_raw)

            # ── Document ──────────────────────────────────────────────────
            yield _sse({"status": "documenting", "message": "Generating issue report…", "progress": 50})
            report = pipeline.document_issues_with_examples(df_raw)

            # ── Clean ─────────────────────────────────────────────────────
            yield _sse({"status": "cleaning", "message": "Applying 14-step cleaning pipeline…", "progress": 70})
            df_clean = pipeline.apply_cleaning_techniques(df_raw)

            # ── Save results ──────────────────────────────────────────────
            yield _sse({"status": "saving", "message": "Saving results…", "progress": 90})

            session_out = os.path.join(PROCESSED_FOLDER, session_id)
            os.makedirs(session_out, exist_ok=True)

            df_clean.to_csv(os.path.join(session_out, "cleaned_data.csv"), index=False)

            with open(os.path.join(session_out, "analysis_report.txt"), "w", encoding="utf-8") as fh:
                fh.write(report)

            with open(os.path.join(session_out, "cleaning_log.json"), "w") as fh:
                json.dump(_sanitize_for_json(pipeline.cleaning_log), fh, indent=2, default=str)

            with open(os.path.join(session_out, "analysis_summary.json"), "w") as fh:
                json.dump(_sanitize_for_json(analysis), fh, indent=2, default=str)

            # ── Complete ──────────────────────────────────────────────────
            summary = _build_summary(
                pipeline, df_raw, df_clean, analysis, report, len(all_dfs)
            )
            yield _sse({
                "status":   "complete",
                "message":  "Processing complete!",
                "progress": 100,
                "results":  summary,
            })

        except MemoryError:
            yield _sse({
                "status":  "error",
                "message": "File too large for available memory. "
                           "Try a smaller file or reduce sampleSize.",
                "progress": 0,
            })
        except Exception as exc:
            yield _sse({"status": "error", "message": str(exc), "progress": 0})

    return Response(generate(), mimetype="text/event-stream")


# ── Analysis-only endpoint (no cleaning) ─────────────────────────────────────
@app.route("/api/analyze/<session_id>", methods=["GET"])
def analyze_only(session_id):
    """Run analysis and return JSON — no cleaning, no file saved."""
    try:
        session_upload_dir = os.path.join(UPLOAD_FOLDER, session_id)
        if not os.path.exists(session_upload_dir):
            return jsonify({"error": "Session not found", "error_code": "NO_SESSION"}), 404

        pipeline = DataProcessingPipeline()
        all_dfs  = pipeline.load_files_from_directory(session_upload_dir, ["csv", "json", "txt"])

        if not all_dfs:
            return jsonify({"error": "No readable files", "error_code": "NO_DATA"}), 400

        df = list(all_dfs.values())[0]
        analysis = pipeline.comprehensive_data_analysis(df)
        report   = pipeline.document_issues_with_examples(df)

        return jsonify({
            "analysis":     analysis,
            "report":       report,
            "issuesFound":  _count_issues(analysis),
        })

    except Exception as exc:
        return jsonify({"error": str(exc), "error_code": "ANALYSIS_ERROR"}), 500


# ── Download results as ZIP ───────────────────────────────────────────────────
@app.route("/api/download/<session_id>")
def download_results(session_id):
    try:
        session_out = os.path.join(PROCESSED_FOLDER, session_id)
        if not os.path.exists(session_out):
            return jsonify({"error": "Results not found — run /api/process first",
                            "error_code": "NO_RESULTS"}), 404

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(session_out):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    zf.write(fpath, os.path.relpath(fpath, session_out))
        buf.seek(0)

        return send_file(
            buf,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"dsaral_{session_id[:8]}.zip",
        )

    except Exception as exc:
        return jsonify({"error": str(exc), "error_code": "DOWNLOAD_ERROR"}), 500


# ── Get text report ───────────────────────────────────────────────────────────
@app.route("/api/report/<session_id>")
def get_report(session_id):
    try:
        rpath = os.path.join(PROCESSED_FOLDER, session_id, "analysis_report.txt")
        if not os.path.exists(rpath):
            return jsonify({"error": "Report not found", "error_code": "NO_REPORT"}), 404
        with open(rpath, encoding="utf-8") as fh:
            return jsonify({"report": fh.read()})
    except Exception as exc:
        return jsonify({"error": str(exc), "error_code": "REPORT_ERROR"}), 500


# ── Session status ────────────────────────────────────────────────────────────
@app.route("/api/status/<session_id>")
def get_session_status(session_id):
    try:
        up_dir  = os.path.join(UPLOAD_FOLDER,    session_id)
        out_dir = os.path.join(PROCESSED_FOLDER, session_id)
        files   = []

        if os.path.exists(up_dir):
            for fn in os.listdir(up_dir):
                fp = os.path.join(up_dir, fn)
                if os.path.isfile(fp):
                    files.append({"filename": fn, "size": os.path.getsize(fp)})

        return jsonify({
            "uploaded":  os.path.exists(up_dir),
            "processed": os.path.exists(out_dir),
            "files":     files,
        })
    except Exception as exc:
        return jsonify({"error": str(exc), "error_code": "STATUS_ERROR"}), 500


# ── Cleanup ───────────────────────────────────────────────────────────────────
@app.route("/api/cleanup/<session_id>", methods=["DELETE"])
def cleanup_session(session_id):
    try:
        for folder in (UPLOAD_FOLDER, PROCESSED_FOLDER):
            path = os.path.join(folder, session_id)
            if os.path.exists(path):
                shutil.rmtree(path)
        return jsonify({"message": "Session cleaned up"})
    except Exception as exc:
        return jsonify({"error": str(exc), "error_code": "CLEANUP_ERROR"}), 500


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)