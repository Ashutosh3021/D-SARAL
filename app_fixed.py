from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
import uuid
import zipfile
import io
from werkzeug.utils import secure_filename
import tempfile
import shutil
from datetime import datetime
import re

# Import our data processing class
from backend_processing import DataProcessingPipeline

app = Flask(__name__, static_url_path='', static_folder='frontend', template_folder='frontend')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'csv', 'json', 'txt'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    if '.' not in filename:
        return False
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_extension(filename):
    return filename.rsplit('.', 1)[1].lower()

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/<path:path>')
def static_files(path):
    return app.send_static_file(path)

@app.route('/api/upload', methods=['POST'])
def upload_files():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        file_type = request.form.get('fileType', 'csv')
        
        if not files or len(files) == 0 or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        # Create session directory
        session_id = str(uuid.uuid4())
        session_upload_dir = os.path.join(UPLOAD_FOLDER, session_id)
        
        # Ensure upload directory exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(session_upload_dir, exist_ok=True)
        
        uploaded_files = []
        
        # First validate all files before saving
        for file in files:
            if file and not allowed_file(file.filename):
                return jsonify({'error': f'File type not allowed: {file.filename}'}), 400
        
        # Now save all valid files
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(session_upload_dir, filename)
                file.save(file_path)
                uploaded_files.append({
                    'filename': filename,
                    'path': file_path,
                    'size': os.path.getsize(file_path)
                })
        
        return jsonify({
            'sessionId': session_id,
            'files': uploaded_files,
            'message': f'Successfully uploaded {len(uploaded_files)} files'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process/<session_id>', methods=['POST', 'GET'])
def process_data(session_id):
    # Get processing parameters from the request, but only for POST requests
    try:
        if request.method == 'POST':
            params = request.get_json() or {}
        else:
            params = {}
    except:
        params = {}
    sample_size = params.get('sampleSize', 10000)
    
    def generate():
        try:
            session_upload_dir = os.path.join(UPLOAD_FOLDER, session_id)
            if not os.path.exists(session_upload_dir):
                yield f'data: {json.dumps({"error": "Session not found", "progress": 0})}\n\n'
                return
            
            # Use the captured parameters
            sample_size = params.get('sampleSize', 10000)
            
            # Process files
            pipeline = DataProcessingPipeline(sample_size=sample_size)
            
            # Load files
            yield f'data: {json.dumps({"status": "loading", "message": "Loading files...", "progress": 10})}\n\n'
            
            # Load files from directory
            all_dataframes = pipeline.load_files_from_directory(session_upload_dir, ['csv', 'json', 'txt'])
            
            # Use the first dataframe for analysis
            if all_dataframes:
                df_to_analyze = list(all_dataframes.values())[0]
            else:
                # If no files were loaded, create an empty dataframe
                df_to_analyze = pd.DataFrame()
            
            yield f'data: {json.dumps({"status": "analyzing", "message": "Analyzing data quality...", "progress": 30})}\n\n'
            
            # Analyze data
            analysis_results = pipeline.comprehensive_data_analysis(df_to_analyze)
            yield f'data: {json.dumps({"status": "documenting", "message": "Documenting issues...", "progress": 60})}\n\n'
            
            # Generate report
            issue_report = pipeline.document_issues_with_examples(df_to_analyze)
            yield f'data: {json.dumps({"status": "cleaning", "message": "Cleaning data...", "progress": 80})}\n\n'
            
            # Clean data
            cleaned_df = pipeline.apply_cleaning_techniques(df_to_analyze)
            
            # Save results
            session_processed_dir = os.path.join(PROCESSED_FOLDER, session_id)
            os.makedirs(session_processed_dir, exist_ok=True)
            
            # Save cleaned data
            cleaned_file_path = os.path.join(session_processed_dir, 'cleaned_data.csv')
            cleaned_df.to_csv(cleaned_file_path, index=False)
            
            # Save analysis report
            report_path = os.path.join(session_processed_dir, 'analysis_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(issue_report)
            
            # Save cleaning log
            log_path = os.path.join(session_processed_dir, 'cleaning_log.json')
            with open(log_path, 'w') as f:
                json.dump(pipeline.cleaning_log, f, indent=2, default=str)
            
            result_data = {
                'status': 'complete',
                'message': 'Processing complete!',
                'progress': 100,
                'results': {
                    'filesProcessed': len(all_dataframes),
                    'originalShape': df_to_analyze.shape if df_to_analyze is not None and not df_to_analyze.empty else [0, 0],
                    'cleanedShape': cleaned_df.shape,
                    'issuesFound': len(analysis_results.get('missing_values', {}).get('total_missing_per_column', {})) +
                                  len(analysis_results.get('format_inconsistencies', {})) +
                                  len(analysis_results.get('broken_entries', {})),
                    'reportPreview': issue_report[:500] + '...' if len(issue_report) > 500 else issue_report
                }
            }
            yield f'data: {json.dumps(result_data)}\n\n'
        
        except Exception as e:
            yield f'data: {json.dumps({"status": "error", "message": str(e), "progress": 0})}\n\n'
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/download/<session_id>')
def download_results(session_id):
    try:
        session_processed_dir = os.path.join(PROCESSED_FOLDER, session_id)
        if not os.path.exists(session_processed_dir):
            return jsonify({'error': 'Results not found'}), 404
        
        # Create zip file in memory
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(session_processed_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, session_processed_dir)
                    zf.write(file_path, arc_name)
        
        memory_file.seek(0)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'dsaral_results_{session_id}.zip'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/report/<session_id>')
def get_report(session_id):
    try:
        session_processed_dir = os.path.join(PROCESSED_FOLDER, session_id)
        report_path = os.path.join(session_processed_dir, 'analysis_report.txt')
        
        if not os.path.exists(report_path):
            return jsonify({'error': 'Report not found'}), 404
        
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        return jsonify({'report': report_content})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<session_id>')
def get_session_status(session_id):
    try:
        session_upload_dir = os.path.join(UPLOAD_FOLDER, session_id)
        session_processed_dir = os.path.join(PROCESSED_FOLDER, session_id)
        
        status = {
            'uploaded': os.path.exists(session_upload_dir),
            'processed': os.path.exists(session_processed_dir),
            'files': []
        }
        
        if os.path.exists(session_upload_dir):
            for file in os.listdir(session_upload_dir):
                file_path = os.path.join(session_upload_dir, file)
                if os.path.isfile(file_path):
                    status['files'].append({
                        'filename': file,
                        'size': os.path.getsize(file_path)
                    })
        
        return jsonify(status)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup/<session_id>', methods=['DELETE'])
def cleanup_session(session_id):
    try:
        # Clean up upload directory
        session_upload_dir = os.path.join(UPLOAD_FOLDER, session_id)
        if os.path.exists(session_upload_dir):
            shutil.rmtree(session_upload_dir)
        
        # Clean up processed directory
        session_processed_dir = os.path.join(PROCESSED_FOLDER, session_id)
        if os.path.exists(session_processed_dir):
            shutil.rmtree(session_processed_dir)
        
        return jsonify({'message': 'Session cleaned up successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)