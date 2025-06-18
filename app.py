from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, make_response
import os
import main_web_processor  # Import the refactored processing logic
from werkzeug.utils import secure_filename
import pandas as pd # Needed for combining dataframes
import io # Needed for sending file data from memory
import glob # Import glob
from main_web_processor import (
    generate_reports, 
    generate_comparison_plot, 
    create_cross_comparison_plot,
    process_lv_file,
    process_gc_file,
    merge_with_interpolation,
    merge_overall_data,
    merge_step_data,
    create_dynamic_plot,
    plot_overall_merged_data,
    plot_per_step_data
)  # Import all required functions
from datetime import datetime
import json
import time
import math
# Import database modules
from Database.nh3_synth_database import NH3SynthDatabaseProcessor
from Database.database_models import init_db, get_session, Experiment, PlotSummary, ExperimentStep, StepDataPoint
from pathlib import Path
import tempfile

app = Flask(__name__, template_folder='templates', static_folder='static') # Create Flask application instance

# --- Configuration ---
# Define the path for uploaded files. Create the directory if it doesn't exist.
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the base directory for reports within the static folder
REPORTS_FOLDER = os.path.join(app.static_folder, 'reports') # Use app.static_folder
if not os.path.exists(REPORTS_FOLDER):
    os.makedirs(REPORTS_FOLDER)
# Explicitly add the path to the app config dictionary
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER 

# Define database folder
DATABASE_FOLDER = 'Database'
if not os.path.exists(DATABASE_FOLDER):
    os.makedirs(DATABASE_FOLDER)
app.config['DATABASE_FOLDER'] = DATABASE_FOLDER

# Initialize database for API
db_path = os.path.join(DATABASE_FOLDER, 'nh3_synth.db')
app.config['DATABASE_PATH'] = db_path

# Set recreate_db to False to avoid lock issues on startup
recreate_db = False

# Initialize database during startup
try:
    with app.app_context():
        processor = NH3SynthDatabaseProcessor(
            reports_dir=app.config['REPORTS_FOLDER'],
            db_path=app.config['DATABASE_PATH'],
            recreate_db=recreate_db
        )
    print(f"Database initialized at {app.config['DATABASE_PATH']}")
except Exception as e:
    print(f"Error initializing database: {e}")
    import traceback
    traceback.print_exc()
    print("The application will continue without database initialization.")

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'txt'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---
@app.route('/')
def index():
    """ Serves the main HTML page. """
    return render_template('index.html')

@app.route('/reports')
def reports_page():
    """ Serves the reports management page. """
    return render_template('reports.html')

@app.route('/upload')
def upload_page():
    """ Serves the upload page. """
    return render_template('upload.html')

@app.route('/visualize')
def visualize_page():
    """ Serves the visualization page. """
    return render_template('visualize.html')

# API Routes for the frontend

@app.route('/api/reports')
def api_reports():
    """Returns a list of available reports with basic metadata."""
    try:
        reports_base = app.config['REPORTS_FOLDER']
        # List only directories directly under the reports folder
        all_items = os.listdir(reports_base)
        report_folders = [d for d in all_items if os.path.isdir(os.path.join(reports_base, d))]
        # Sort by creation time, newest first
        report_folders.sort(key=lambda x: os.path.getctime(os.path.join(reports_base, x)), reverse=True)
        
        reports_data = []
        for folder in report_folders:
            # Get basic stats for each report
            folder_path = os.path.join(reports_base, folder)
            created_at = datetime.fromtimestamp(os.path.getctime(folder_path)).strftime('%Y-%m-%d %H:%M:%S')
            
            # Count number of stages by looking for step_* directories
            step_folders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d)) and d.startswith('step_')]
            
            reports_data.append({
                'name': folder,
                'created_at': created_at,
                'stages': len(step_folders)
            })
        
        return jsonify({'success': True, 'reports': reports_data})
    except Exception as e:
        print(f"Error listing reports for API: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error listing reports: {e}'}), 500

@app.route('/api/report-stats/<report_name>')
def api_report_stats(report_name):
    """Returns detailed statistics for a specific report."""
    try:
        report_folder = os.path.join(app.config['REPORTS_FOLDER'], report_name)
        
        if not os.path.isdir(report_folder):
            return jsonify({'success': False, 'message': 'Report not found'}), 404
            
        # Get basic stats
        created_at = datetime.fromtimestamp(os.path.getctime(report_folder)).strftime('%Y-%m-%d %H:%M:%S')
        
        # Get GC correction factor from metadata if available
        gc_correction_factor = None
        metadata_file_path = os.path.join(report_folder, 'catalyst_metadata.json')
        if os.path.exists(metadata_file_path):
            try:
                with open(metadata_file_path, 'r') as f:
                    metadata = json.load(f)
                    gc_correction_factor = metadata.get('gc_correction_factor')
            except Exception as e:
                print(f"Error reading metadata for {report_name}: {e}")
        
        # Count files and get total size
        total_files = 0
        total_size = 0
        for root, dirs, files in os.walk(report_folder):
            total_files += len(files)
            total_size += sum(os.path.getsize(os.path.join(root, file)) for file in files)
        
        # Count stages and data points
        step_folders = sorted(
            [d for d in os.listdir(report_folder) if os.path.isdir(os.path.join(report_folder, d)) and d.startswith('step_')],
            key=lambda x: int(x.split('_')[1])
        )
        
        stages_data = []
        total_data_points = 0
        
        for step_folder in step_folders:
            step_num = int(step_folder.split('_')[1])
            step_folder_path = os.path.join(report_folder, step_folder)
            
            # Try to read data from JSON file
            json_file = os.path.join(step_folder_path, f"step_{step_num}_data.json")
            data_points = 0
            max_temp = None
            duration = None
            
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    data_points = len(data)
                    total_data_points += data_points
                    
                    # Extract stage statistics
                    temps = []
                    dates = []
                    pressures = []
                    h2_flows = []
                    n2_flows = []
                    nh3_values = []
                    outlet_values = []
                    
                    for entry in data:
                        # Extract dates for duration calculation
                        if 'Date' in entry:
                            dates.append(entry['Date'])
                        
                        # Temperature data - prioritize "T Heater UP"
                        temperature_found = False
                        for temp_key in ['T Heater UP', 'T Heater 1', 'T Heater 1_LV', 'T_Heater_1', 'Temperature']:
                            if temp_key in entry and isinstance(entry[temp_key], (int, float)) and not pd.isna(entry[temp_key]):
                                temps.append(entry[temp_key])
                                temperature_found = True
                                break
                        
                        # Pressure data - prioritize "Pressure reading"
                        pressure_found = False
                        for pressure_key in ['Pressure reading', 'Pressure', 'Pressure_LV', 'Pressure (bar)', 'Pressure setpoint', 'Pressure-Setpoint', 'Pressure_SP', 'Pressure SP']:
                            if pressure_key in entry and isinstance(entry[pressure_key], (int, float)) and not pd.isna(entry[pressure_key]):
                                pressures.append(entry[pressure_key])
                                pressure_found = True
                                break
                        
                        # H2 Flow data
                        h2_found = False
                        for h2_key in ['H2 Actual Flow', 'H2_Flow', 'H2 Flow', 'H2 Flow (ml/min)']:
                            if h2_key in entry and isinstance(entry[h2_key], (int, float)) and not pd.isna(entry[h2_key]):
                                h2_flows.append(entry[h2_key])
                                h2_found = True
                                break
                        
                        # N2 Flow data
                        n2_found = False
                        for n2_key in ['N2 Actual Flow', 'N2_Flow', 'N2 Flow', 'N2 Flow (ml/min)']:
                            if n2_key in entry and isinstance(entry[n2_key], (int, float)) and not pd.isna(entry[n2_key]):
                                n2_flows.append(entry[n2_key])
                                n2_found = True
                                break
                        
                        # NH3 data
                        nh3_found = False
                        for nh3_key in ['NH3_GC', 'NH3_clean_GC', 'NH3 (%)', 'NH3']:
                            if nh3_key in entry and isinstance(entry[nh3_key], (int, float)) and not pd.isna(entry[nh3_key]):
                                nh3_values.append(entry[nh3_key])
                                nh3_found = True
                                break
                        
                        # Outlet mass flow data - prioritize "Outlet meas.flowrate"
                        outlet_found = False
                        for outlet_key in ['Outlet meas.flowrate', 'Outlet g/h', 'Outlet mass flowrate', 'Outlet_mass', 'Mass flowrate (g/h)']:
                            if outlet_key in entry and isinstance(entry[outlet_key], (int, float)) and not pd.isna(entry[outlet_key]):
                                outlet_values.append(entry[outlet_key])
                                outlet_found = True
                                break
                    
                    # Calculate statistics
                    if temps:
                        max_temp = round(max(t for t in temps if isinstance(t, (int, float))), 1)
                        avg_temp = round(sum(temps) / len(temps), 2)
                    else:
                        max_temp = None
                        avg_temp = None
                    
                    avg_pressure = round(sum(pressures) / len(pressures), 2) if pressures else None
                    avg_h2_flow = round(sum(h2_flows) / len(h2_flows), 2) if h2_flows else None
                    avg_n2_flow = round(sum(n2_flows) / len(n2_flows), 2) if n2_flows else None
                    avg_nh3 = round(sum(nh3_values) / len(nh3_values), 2) if nh3_values else None
                    avg_outlet = round(sum(outlet_values) / len(outlet_values), 2) if outlet_values else None
                    
                    # Calculate duration if dates are available
                    if dates and len(dates) >= 2:
                        try:
                            start_date = datetime.fromisoformat(dates[0].replace('Z', '+00:00'))
                            end_date = datetime.fromisoformat(dates[-1].replace('Z', '+00:00'))
                            duration_seconds = (end_date - start_date).total_seconds()
                            hours = int(duration_seconds // 3600)
                            minutes = int((duration_seconds % 3600) // 60)
                            duration = f"{hours}h {minutes}m"
                        except:
                            duration = "N/A"
                except Exception as e:
                    print(f"Error processing JSON for stage {step_num}: {e}")
                    
            stages_data.append({
                'number': step_num,
                'data_points': data_points,
                'max_temp': max_temp,
                'duration': duration,
                'avg_temp': avg_temp,
                'avg_pressure': avg_pressure,
                'avg_h2_flow': avg_h2_flow,
                'avg_n2_flow': avg_n2_flow,
                'avg_nh3': avg_nh3,
                'avg_outlet': avg_outlet
            })
        
        # Create export URL
        export_url = f"/static/reports/{report_name}/overall_merged_data.csv"
        
        return jsonify({
            'success': True,
            'created_at': created_at,
            'stages': len(step_folders),
            'data_points': total_data_points,
            'files_count': total_files,
            'total_size': total_size,
            'stages_data': stages_data,
            'gc_correction_factor': gc_correction_factor,
            'export_url': export_url if os.path.exists(os.path.join(app.static_folder, 'reports', report_name, 'overall_merged_data.csv')) else None
        })
    except Exception as e:
        print(f"Error getting report stats: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error getting report stats: {e}'}), 500

@app.route('/api/report-metadata/<report_name>', methods=['GET'])
def api_get_report_metadata(report_name):
    """Returns catalyst metadata for a specific report."""
    try:
        # Sanitize input to prevent directory traversal attacks
        report_name = secure_filename(report_name)
        report_folder = os.path.join(app.config['REPORTS_FOLDER'], report_name)
        
        if not os.path.isdir(report_folder):
            return jsonify({'success': False, 'message': 'Report not found'}), 404
        
        # Check if metadata file exists
        metadata_file_path = os.path.join(report_folder, 'catalyst_metadata.json')
        
        if os.path.exists(metadata_file_path):
            try:
                with open(metadata_file_path, 'r') as f:
                    metadata = json.load(f)
                return jsonify({'success': True, 'metadata': metadata})
            except Exception as e:
                print(f"Error reading metadata file: {e}")
                return jsonify({'success': False, 'message': f'Error reading metadata file: {e}'}), 500
        else:
            # Return empty metadata object if no file exists
            empty_metadata = {
                'experiment_number': '',
                'catalyst_batch': '',
                'reactor_number': '',
                'catalyst_weight': '',
                'catalyst_volume': '',
                'catalyst_state': '',
                'diluent_type': '',
                'catalyst_diluent_ratio': '',
                'catalyst_bed_length': '',
                'tc_top': '',
                'tc_bottom': '',
                'catalyst_notes': '',
                'gc_correction_factor': '0.86',
                'created_at': '',
                'updated_at': ''
            }
            return jsonify({'success': True, 'metadata': empty_metadata, 'file_exists': False})
    
    except Exception as e:
        print(f"Error getting report metadata: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error getting report metadata: {e}'}), 500

@app.route('/api/update-report-metadata/<report_name>', methods=['POST'])
def api_update_report_metadata(report_name):
    """Updates catalyst metadata for a specific report."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No metadata provided'}), 400
        
        # Sanitize input to prevent directory traversal attacks
        report_name = secure_filename(report_name)
        report_folder = os.path.join(app.config['REPORTS_FOLDER'], report_name)
        
        if not os.path.isdir(report_folder):
            return jsonify({'success': False, 'message': 'Report not found'}), 404
        
        # Check if metadata file exists and load existing data if it does
        metadata_file_path = os.path.join(report_folder, 'catalyst_metadata.json')
        existing_metadata = {}
        
        if os.path.exists(metadata_file_path):
            try:
                with open(metadata_file_path, 'r') as f:
                    existing_metadata = json.load(f)
            except Exception as e:
                print(f"Error reading existing metadata: {e}")
                # Continue with empty existing metadata
        
        # Update metadata with new values
        updated_metadata = {
            'experiment_number': data.get('experiment_number', existing_metadata.get('experiment_number', '')),
            'catalyst_batch': data.get('catalyst_batch', existing_metadata.get('catalyst_batch', '')),
            'reactor_number': data.get('reactor_number', existing_metadata.get('reactor_number', '')),
            'catalyst_weight': data.get('catalyst_weight', existing_metadata.get('catalyst_weight', '')),
            'catalyst_volume': data.get('catalyst_volume', existing_metadata.get('catalyst_volume', '')),
            'catalyst_state': data.get('catalyst_state', existing_metadata.get('catalyst_state', '')),
            'diluent_type': data.get('diluent_type', existing_metadata.get('diluent_type', '')),
            'catalyst_diluent_ratio': data.get('catalyst_diluent_ratio', existing_metadata.get('catalyst_diluent_ratio', '')),
            'catalyst_bed_length': data.get('catalyst_bed_length', existing_metadata.get('catalyst_bed_length', '')),
            'tc_top': data.get('tc_top', existing_metadata.get('tc_top', '')),
            'tc_bottom': data.get('tc_bottom', existing_metadata.get('tc_bottom', '')),
            'gc_correction_factor': data.get('gc_correction_factor', existing_metadata.get('gc_correction_factor', '0.86')),
            'catalyst_notes': data.get('catalyst_notes', existing_metadata.get('catalyst_notes', '')),
            'created_at': existing_metadata.get('created_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save updated metadata
        try:
            with open(metadata_file_path, 'w') as f:
                json.dump(updated_metadata, f, indent=2)
            return jsonify({'success': True, 'message': 'Metadata updated successfully', 'metadata': updated_metadata})
        except Exception as e:
            print(f"Error saving updated metadata: {e}")
            return jsonify({'success': False, 'message': f'Error saving metadata: {e}'}), 500
    
    except Exception as e:
        print(f"Error updating report metadata: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error updating report metadata: {e}'}), 500

@app.route('/api/report-contents/<report_name>')
def api_report_contents(report_name):
    """Returns the file/folder structure of a report."""
    try:
        report_folder = os.path.join(app.config['REPORTS_FOLDER'], report_name)
        
        if not os.path.isdir(report_folder):
            return jsonify({'success': False, 'message': 'Report not found'}), 404
            
        # Create a function to recursively build the structure
        def get_folder_structure(path, max_depth=2, current_depth=0):
            items = []
            
            if current_depth > max_depth:
                return items
                
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                
                if os.path.isdir(item_path):
                    children = get_folder_structure(item_path, max_depth, current_depth + 1) if current_depth < max_depth else []
                    items.append({
                        'name': item,
                        'type': 'folder',
                        'children': children
                    })
                else:
                    size = os.path.getsize(item_path)
                    size_str = f"{size} bytes"
                    if size > 1024:
                        size_str = f"{size/1024:.1f} KB"
                    if size > 1024*1024:
                        size_str = f"{size/(1024*1024):.1f} MB"
                        
                    items.append({
                        'name': item,
                        'type': 'file',
                        'size': size_str
                    })
            
            return sorted(items, key=lambda x: (0 if x['type'] == 'folder' else 1, x['name']))
            
        contents = get_folder_structure(report_folder)
        
        return jsonify({
            'success': True,
            'contents': contents
        })
    except Exception as e:
        print(f"Error getting report contents: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error getting report contents: {e}'}), 500

@app.route('/api/rename-report', methods=['POST'])
def api_rename_report():
    """Renames a report folder."""
    try:
        data = request.get_json()
        old_name = data.get('old_name')
        new_name = data.get('new_name')
        
        if not old_name or not new_name:
            return jsonify({'success': False, 'message': 'Missing old or new report name.'}), 400
        
        # Sanitize input to prevent directory traversal attacks
        from werkzeug.utils import secure_filename
        old_name = secure_filename(old_name)
        new_name = secure_filename(new_name)
        
        old_path = os.path.join(app.config['REPORTS_FOLDER'], old_name)
        new_path = os.path.join(app.config['REPORTS_FOLDER'], new_name)
        
        # Check if old folder exists
        if not os.path.isdir(old_path):
            return jsonify({'success': False, 'message': f'Report folder not found: {old_name}'}), 404
        
        # Check if new folder name already exists
        if os.path.exists(new_path):
            return jsonify({'success': False, 'message': f'A report with the name {new_name} already exists.'}), 409
        
        # Rename the folder
        os.rename(old_path, new_path)
        
        return jsonify({'success': True, 'message': f'Report renamed from {old_name} to {new_name}'})
    except Exception as e:
        print(f"Error renaming report: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error renaming report: {e}'}), 500

@app.route('/api/delete-report', methods=['POST'])
def api_delete_report():
    """Deletes a report folder and all its contents."""
    try:
        data = request.get_json()
        report_name = data.get('report_name')
        
        if not report_name:
            return jsonify({'success': False, 'message': 'Missing report name.'}), 400
        
        # Sanitize input to prevent directory traversal attacks
        from werkzeug.utils import secure_filename
        report_name = secure_filename(report_name)
        
        report_path = os.path.join(app.config['REPORTS_FOLDER'], report_name)
        
        # Check if folder exists
        if not os.path.isdir(report_path):
            return jsonify({'success': False, 'message': f'Report folder not found: {report_name}'}), 404
        
        # Use shutil.rmtree to remove the directory and all its contents
        import shutil
        shutil.rmtree(report_path)
        
        return jsonify({'success': True, 'message': f'Report {report_name} deleted successfully'})
    except Exception as e:
        print(f"Error deleting report: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error deleting report: {e}'}), 500

@app.route('/api/dashboard-data')
def api_dashboard_data():
    """Returns basic statistics for the dashboard."""
    try:
        reports_base = app.config['REPORTS_FOLDER']
        
        # List only directories directly under the reports folder
        all_items = os.listdir(reports_base)
        report_folders = [d for d in all_items if os.path.isdir(os.path.join(reports_base, d)) and d != 'cross_comparisons']
        
        # Sort by creation time, newest first
        report_folders.sort(key=lambda x: os.path.getctime(os.path.join(reports_base, x)), reverse=True)
        
        # Count total reports
        total_reports = len(report_folders)
        
        # Count recent uploads (last 7 days)
        one_week_ago = time.time() - (7 * 24 * 60 * 60)
        recent_uploads = sum(1 for folder in report_folders if os.path.getctime(os.path.join(reports_base, folder)) > one_week_ago)
        
        # Get information for the most recent reports (up to 5)
        recent_reports = []
        reports_stats = []
        
        for folder in report_folders:
            folder_path = os.path.join(reports_base, folder)
            created_at = datetime.fromtimestamp(os.path.getctime(folder_path)).strftime('%Y-%m-%d %H:%M:%S')
            
            # Count number of stages by looking for step_* directories
            step_folders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d)) and d.startswith('step_')]
            
            # Only add to recent_reports if it's one of the first 5
            if len(recent_reports) < 5:
                recent_reports.append({
                    'name': folder,
                    'created_at': created_at,
                    'stages': len(step_folders)
                })
            
            # Calculate total data points
            data_points = 0
            overall_csv_path = os.path.join(folder_path, 'overall_merged_data.csv')
            if os.path.exists(overall_csv_path):
                try:
                    df = pd.read_csv(overall_csv_path)
                    data_points = len(df)
                except Exception as e:
                    print(f"Error reading overall CSV for {folder}: {e}")
            
            reports_stats.append({
                'name': folder,
                'data_points': data_points
            })
        
        return jsonify({
            'success': True,
            'total_reports': total_reports,
            'recent_uploads': recent_uploads,
            'recent_reports': recent_reports,
            'reports_stats': reports_stats
        })
    except Exception as e:
        print(f"Error getting dashboard data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error getting dashboard data: {e}'}), 500

@app.route('/api/list-cross-report-folders')
def api_list_cross_report_folders():
    """Returns a list of report folders for cross-report comparison."""
    try:
        reports_base = app.config['REPORTS_FOLDER']
        
        # List only directories directly under the reports folder
        all_items = os.listdir(reports_base)
        report_folders = [d for d in all_items if os.path.isdir(os.path.join(reports_base, d))]
        
        # Sort by name
        report_folders.sort()
        
        return jsonify({
            'success': True,
            'folders': report_folders
        })
    except Exception as e:
        print(f"Error listing cross-report folders: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error listing folders: {e}'}), 500

@app.route('/process', methods=['POST'])
def process_files():
    """ 
    Handles file uploads, triggers processing, and returns results as JSON.
    """
    # --- File Upload Handling ---
    if 'lv_file' not in request.files or 'gc_file' not in request.files:
        return jsonify({'success': False, 'message': 'Missing LV or GC file in request.'}), 400

    lv_file = request.files['lv_file']
    gc_file = request.files['gc_file']

    if lv_file.filename == '' or gc_file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file(s).'}), 400

    if lv_file and allowed_file(lv_file.filename) and gc_file and allowed_file(gc_file.filename):
        # Secure filenames and save uploaded files temporarily
        lv_filename = secure_filename(lv_file.filename)
        gc_filename = secure_filename(gc_file.filename)
        lv_filepath = os.path.join(app.config['UPLOAD_FOLDER'], lv_filename)
        gc_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gc_filename)

        try:
            lv_file.save(lv_filepath)
            gc_file.save(gc_filepath)
            print(f"Files saved: {lv_filepath}, {gc_filepath}")
        except Exception as e:
            print(f"Error saving files: {e}")
            return jsonify({'success': False, 'message': f'Error saving uploaded files: {e}'}), 500

        # --- Data Processing ---
        try:
            # Get the processing mode (new or add)
            processing_mode = request.form.get('processing_mode', 'new')
            
            # Path to existing report if in 'add' mode
            existing_report_path = None
            if processing_mode == 'add':
                existing_report_name = request.form.get('existing_report')
                if not existing_report_name:
                    return jsonify({'success': False, 'message': 'No existing report specified for add mode.'}), 400
                
                # Sanitize to prevent directory traversal attacks
                existing_report_name = secure_filename(existing_report_name)
                existing_report_path = os.path.join(REPORTS_FOLDER, existing_report_name)
                
                if not os.path.isdir(existing_report_path):
                    return jsonify({'success': False, 'message': f'Existing report not found: {existing_report_name}'}), 404
            
            # Get the metadata fields to generate the report prefix for new mode
            report_prefix_text = None
            if processing_mode == 'new':
                experiment_number = request.form.get('experiment_number', '').strip()
                catalyst_batch = request.form.get('catalyst_batch', '').strip()
                reactor_number = request.form.get('reactor_number', '').strip()
                
                if experiment_number and catalyst_batch and reactor_number:
                    # Generate the prefix from metadata fields
                    report_prefix_text = f"{experiment_number}_{catalyst_batch}_{reactor_number}"
                else:
                    # If metadata is missing, use a generic timestamp
                    report_prefix_text = None
            
            # Get interpolation parameters from the form
            use_interpolation = request.form.get('use_interpolation', 'false').lower() == 'true'
            interpolation_kind = request.form.get('interpolation_kind', 'cubic')
            
            # Get uniform grid parameters
            use_uniform_grid = request.form.get('use_uniform_grid', 'true').lower() == 'true'
            grid_freq = request.form.get('grid_freq', '1min')
            
            # Get GC correction factor
            gc_correction_factor = float(request.form.get('gc_correction_factor', '0.86'))
            print(f"Using GC correction factor: {gc_correction_factor}")
            
            # Validate interpolation_kind to prevent injection
            valid_interpolation_kinds = ['linear', 'cubic', 'quadratic', 'nearest']
            if interpolation_kind not in valid_interpolation_kinds:
                interpolation_kind = 'cubic'  # Default to cubic if invalid
                
            # Validate grid_freq to prevent injection
            valid_grid_freqs = ['30s', '1min', '2min', '5min', '10min']
            if grid_freq not in valid_grid_freqs:
                grid_freq = '1min'  # Default to 1min if invalid

            # Call the main processing function from the refactored module
            # Pass the static reports folder as the base output folder and the prefix
            results = main_web_processor.generate_reports(
                lv_filepath,
                gc_filepath,
                REPORTS_FOLDER,
                report_prefix_text=report_prefix_text,
                use_interpolation=use_interpolation,
                interpolation_kind=interpolation_kind,
                use_uniform_grid=use_uniform_grid,
                grid_freq=grid_freq,
                existing_report_path=existing_report_path,
                gc_correction_factor=gc_correction_factor
            )
            
            # --- Save Catalyst Metadata ---
            if results.get('success') and results.get('timestamp_prefix'):
                report_folder = os.path.join(REPORTS_FOLDER, results['timestamp_prefix'])
                
                # For new reports, create metadata from form
                if processing_mode == 'new':
                    # Get all the catalyst metadata from the form
                    catalyst_metadata = {
                        'experiment_number': request.form.get('experiment_number', ''),
                        'catalyst_batch': request.form.get('catalyst_batch', ''),
                        'reactor_number': request.form.get('reactor_number', ''),
                        'catalyst_weight': request.form.get('catalyst_weight', ''),
                        'catalyst_volume': request.form.get('catalyst_volume', ''),
                        'catalyst_state': request.form.get('catalyst_state', ''),
                        'particle_size': request.form.get('particle_size', ''),
                        'diluent_type': request.form.get('diluent_type', ''),
                        'catalyst_diluent_ratio': request.form.get('catalyst_diluent_ratio', ''),
                        'catalyst_bed_length': request.form.get('catalyst_bed_length', ''),
                        'tc_top': request.form.get('tc_top', ''),
                        'tc_bottom': request.form.get('tc_bottom', ''),
                        'catalyst_notes': request.form.get('catalyst_notes', ''),
                        'gc_correction_factor': gc_correction_factor,
                        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Create the metadata file in the report folder
                    metadata_file_path = os.path.join(report_folder, 'catalyst_metadata.json')
                    
                    try:
                        with open(metadata_file_path, 'w') as f:
                            json.dump(catalyst_metadata, f, indent=2)
                        print(f"Catalyst metadata saved to: {metadata_file_path}")
                        # Add metadata saved flag to results
                        results['metadata_saved'] = True
                    except Exception as e:
                        print(f"Error saving catalyst metadata: {e}")
                        results['metadata_saved'] = False
                    
                    # --- Save Original Uploaded Files ---
                    # Create proper filenames using metadata
                    experiment_number = catalyst_metadata.get('experiment_number', 'Unknown')
                    catalyst_batch = catalyst_metadata.get('catalyst_batch', 'Unknown')
                    reactor_number = catalyst_metadata.get('reactor_number', 'Unknown')
                    
                    # Generate filenames: Exp_Batch_Reactor_LV.txt and Exp_Batch_Reactor_GC.txt
                    base_filename = f"{experiment_number}_{catalyst_batch}_{reactor_number}"
                    lv_stored_filename = f"{base_filename}_LV.txt"
                    gc_stored_filename = f"{base_filename}_GC.txt"
                    
                    # Copy uploaded files to report folder with new names
                    try:
                        import shutil
                        lv_stored_path = os.path.join(report_folder, lv_stored_filename)
                        gc_stored_path = os.path.join(report_folder, gc_stored_filename)
                        
                        shutil.copy2(lv_filepath, lv_stored_path)
                        shutil.copy2(gc_filepath, gc_stored_path)
                        
                        print(f"Original files saved to report folder:")
                        print(f"  LV: {lv_stored_path}")
                        print(f"  GC: {gc_stored_path}")
                        
                        # Add file storage info to results
                        results['original_files_saved'] = True
                        results['lv_stored_filename'] = lv_stored_filename
                        results['gc_stored_filename'] = gc_stored_filename
                    except Exception as e:
                        print(f"Error saving original files to report folder: {e}")
                        results['original_files_saved'] = False
                # For existing reports, the metadata file should already exist
                # No need to update it when adding new data points
                else:
                    # Check if metadata file exists and if not, create an empty one
                    metadata_file_path = os.path.join(report_folder, 'catalyst_metadata.json')
                    if not os.path.exists(metadata_file_path):
                        # Create a default metadata file with empty values
                        default_metadata = {
                            'experiment_number': '',
                            'catalyst_batch': '',
                            'reactor_number': '',
                            'catalyst_weight': '',
                            'catalyst_volume': '',
                            'catalyst_state': '',
                            'particle_size': '',
                            'diluent_type': '',
                            'catalyst_diluent_ratio': '',
                            'catalyst_bed_length': '',
                            'tc_top': '',
                            'tc_bottom': '',
                            'catalyst_notes': '',
                            'gc_correction_factor': '0.86',
                            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        try:
                            with open(metadata_file_path, 'w') as f:
                                json.dump(default_metadata, f, indent=2)
                            print(f"Default metadata created for existing report: {metadata_file_path}")
                        except Exception as e:
                            print(f"Error creating default metadata: {e}")
                    
                    # In 'add' mode, we're reusing the same report directory
                    # Just inform the user metadata is preserved
                    results['metadata_saved'] = True
                    results['metadata_preserved'] = True
                    
                    # --- Save Latest Version of Original Files for Existing Reports ---
                    # Load existing metadata to get naming convention
                    try:
                        with open(metadata_file_path, 'r') as f:
                            existing_metadata = json.load(f)
                        
                        experiment_number = existing_metadata.get('experiment_number', 'Unknown')
                        catalyst_batch = existing_metadata.get('catalyst_batch', 'Unknown')
                        reactor_number = existing_metadata.get('reactor_number', 'Unknown')
                        
                        # Generate filenames using existing metadata
                        base_filename = f"{experiment_number}_{catalyst_batch}_{reactor_number}"
                        lv_stored_filename = f"{base_filename}_LV.txt"
                        gc_stored_filename = f"{base_filename}_GC.txt"
                        
                        # Copy uploaded files to report folder, overwriting existing ones
                        import shutil
                        lv_stored_path = os.path.join(report_folder, lv_stored_filename)
                        gc_stored_path = os.path.join(report_folder, gc_stored_filename)
                        
                        shutil.copy2(lv_filepath, lv_stored_path)
                        shutil.copy2(gc_filepath, gc_stored_path)
                        
                        print(f"Latest original files saved to existing report folder:")
                        print(f"  LV: {lv_stored_path}")
                        print(f"  GC: {gc_stored_path}")
                        
                        # Update metadata with latest update timestamp
                        existing_metadata['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        with open(metadata_file_path, 'w') as f:
                            json.dump(existing_metadata, f, indent=2)
                        
                        # Add file storage info to results
                        results['original_files_saved'] = True
                        results['lv_stored_filename'] = lv_stored_filename
                        results['gc_stored_filename'] = gc_stored_filename
                        results['files_updated'] = True
                    except Exception as e:
                        print(f"Error saving latest original files to existing report: {e}")
                        results['original_files_saved'] = False
            
            # --- Cleanup Uploaded Files (Optional) ---
            # You might want to keep these for debugging or remove them
            # try:
            #     os.remove(lv_filepath)
            #     os.remove(gc_filepath)
            #     print(f"Cleaned up uploaded files: {lv_filepath}, {gc_filepath}")
            # except OSError as e:
            #     print(f"Error removing uploaded files: {e}")
            
            # Adjust paths in results to be relative to static folder root
            # Ensure the paths start with 'static/' for web access
            static_folder_name = os.path.basename(app.static_folder)

            # Determine the actual folder name used (prefix + timestamp or just timestamp)
            # This is a bit more complex now, need to get it from one of the paths if available
            # or reconstruct it if no paths were generated but processing was considered successful for the folder creation part.
            actual_report_folder_name = None
            if results.get('overall_plot_path'):
                # Path is like: static/reports/PREFIX_TIMESTAMP/overall_plot.json
                actual_report_folder_name = os.path.basename(os.path.dirname(results['overall_plot_path']))
            elif results.get('overall_csv_path'):
                actual_report_folder_name = os.path.basename(os.path.dirname(results['overall_csv_path']))
            elif results.get('step_reports') and results['step_reports'][0].get('plot_path'):
                actual_report_folder_name = os.path.basename(os.path.dirname(os.path.dirname(results['step_reports'][0]['plot_path'])))
            # Fallback if no files generated but folder might exist (e.g. empty data)
            elif results.get('timestamp_prefix'):
                actual_report_folder_name = results['timestamp_prefix']
            
            if results.get('overall_plot_path'):
                relative_path = os.path.relpath(results['overall_plot_path'], app.static_folder)
                results['overall_plot_path'] = os.path.join(static_folder_name, relative_path).replace('\\', '/')
            if results.get('overall_csv_path'):
                relative_path = os.path.relpath(results['overall_csv_path'], app.static_folder)
                results['overall_csv_path'] = os.path.join(static_folder_name, relative_path).replace('\\', '/')
            
            for step_report in results.get('step_reports', []):
                if step_report.get('plot_path'):
                    relative_path = os.path.relpath(step_report['plot_path'], app.static_folder)
                    step_report['plot_path'] = os.path.join(static_folder_name, relative_path).replace('\\', '/')
                if step_report.get('csv_path'):
                    relative_path = os.path.relpath(step_report['csv_path'], app.static_folder)
                    step_report['csv_path'] = os.path.join(static_folder_name, relative_path).replace('\\', '/')
                if step_report.get('json_path'):
                    relative_path = os.path.relpath(step_report['json_path'], app.static_folder)
                    step_report['json_path'] = os.path.join(static_folder_name, relative_path).replace('\\', '/')

            # Add the timestamp prefix to the response
            if results.get('success'):
                # The timestamp_prefix in results should now be the actual folder name (prefix_timestamp or just timestamp)
                if actual_report_folder_name and actual_report_folder_name != 'reports': 
                    results['timestamp_prefix'] = actual_report_folder_name
                elif report_prefix_text: # if processing made the folder but no files, try to construct it
                    # This is a less ideal fallback, assumes current time if files were not made
                    current_timestamp_for_fallback = datetime.now().strftime("%Y%m%d_%H%M%S")
                    results['timestamp_prefix'] = f"{report_prefix_text}_{current_timestamp_for_fallback}"
                # else: # if no prefix and no files, the original logic for timestamp_prefix might need to be re-evaluated
                    # For now, if actual_report_folder_name is None, timestamp_prefix might remain None or an old value

            return jsonify(results)

        except Exception as e:
            print(f"Error during main processing: {e}")
            import traceback
            traceback.print_exc() # Print detailed error to Flask console
            return jsonify({'success': False, 'message': f'Processing failed: {e}'}), 500
    else:
        return jsonify({'success': False, 'message': 'Invalid file type'}), 400

# --- New Route for Downloading Selected Stages --- 
@app.route('/download_selected_stages', methods=['POST'])
def download_selected_stages():
    """ 
    Receives a list of JSON file paths for selected stages, 
    reads them, combines the data, and sends back a combined CSV file.
    """
    try:
        data = request.get_json()
        json_paths_relative = data.get('json_paths')

        if not json_paths_relative or not isinstance(json_paths_relative, list):
            return jsonify({'success': False, 'message': 'Invalid or missing list of JSON paths.'}), 400
        
        combined_df = pd.DataFrame()
        dfs_to_combine = []

        # Base directory is the project root where app.py is
        base_dir = os.path.dirname(os.path.abspath(__file__))

        for rel_path in json_paths_relative:
            # Construct absolute path and sanitize it
            # Ensure the path stays within the intended static/reports directory
            abs_path = os.path.abspath(os.path.join(base_dir, rel_path))
            allowed_dir = os.path.abspath(os.path.join(base_dir, 'static', 'reports'))
            
            if not abs_path.startswith(allowed_dir):
                 print(f"Warning: Access denied for path outside allowed directory: {rel_path}")
                 continue # Skip potentially malicious paths
            
            if os.path.exists(abs_path):
                try:
                    # Read JSON into DataFrame
                    # orient='records' matches how we saved it in main_web_processor
                    df_step = pd.read_json(abs_path, orient='records') 
                    dfs_to_combine.append(df_step)
                except Exception as e:
                    print(f"Error reading or processing JSON file {abs_path}: {e}")
                    # Optionally inform the user about specific file errors
            else:
                print(f"Warning: JSON file not found: {abs_path}")

        if not dfs_to_combine:
            return jsonify({'success': False, 'message': 'No valid stage data found for selected paths.'}), 404

        # Combine all valid dataframes
        combined_df = pd.concat(dfs_to_combine, ignore_index=True)
        # Optional: Sort by date if needed after combining
        if 'Date' in combined_df.columns:
             try:
                 # Convert Date back to datetime if it was stringified in JSON
                 combined_df['Date'] = pd.to_datetime(combined_df['Date'])
                 combined_df = combined_df.sort_values(by='Date')
             except Exception as e:
                 print(f"Warning: Could not sort combined data by Date: {e}")

        # Convert combined DataFrame to CSV in memory
        csv_buffer = io.StringIO()
        combined_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Send the CSV data as a file download
        response = make_response(csv_buffer.getvalue())
        response.headers['Content-Disposition'] = 'attachment; filename=selected_stages_data.csv'
        response.headers['Content-Type'] = 'text/csv'
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error creating combined CSV: {e}'}), 500

# --- Route to serve generated static files (reports) ---
# This is handled implicitly by Flask if files are in the 'static' folder.
# If your reports were outside 'static', you'd need a route like this:
# @app.route('/reports/<path:filename>')
# def serve_report(filename):
#     return send_from_directory(REPORTS_FOLDER_ABSOLUTE, filename)

# --- New Routes for Loading Previous Reports ---

@app.route('/list_reports', methods=['GET'])
def list_reports():
    """Lists the available timestamped report folders."""
    try:
        reports_base = app.config['REPORTS_FOLDER']
        # List only directories directly under the reports folder
        all_items = os.listdir(reports_base)
        report_folders = [d for d in all_items if os.path.isdir(os.path.join(reports_base, d))]
        # Simple sorting, assumes YYYYMMDD_HHMMSS format
        report_folders.sort(reverse=True) 
        return jsonify({'success': True, 'reports': report_folders})
    except Exception as e:
        print(f"Error listing reports: {e}")
        return jsonify({'success': False, 'message': f'Error listing reports: {e}'}), 500

@app.route('/load_report/<timestamp>', methods=['GET'])
def load_report(timestamp):
    """Load a specific report for visualization."""
    try:
        # Construct report path
        report_folder = os.path.join(app.config['REPORTS_FOLDER'], timestamp)
        
        if not os.path.exists(report_folder):
            return jsonify({'success': False, 'message': 'Report not found.'}), 404
            
        # Get overall plot paths
        overall_plot_path = None
        overall_csv_path = None
        
        # Check for overall plot JSON
        overall_plot_json = os.path.join(report_folder, 'overall_plot.json')
        if os.path.exists(overall_plot_json):
            overall_plot_path = os.path.join('static', 'reports', timestamp, 'overall_plot.json')
        
        # Check for overall CSV
        overall_csv = os.path.join(report_folder, 'overall_merged_data.csv')
        if os.path.exists(overall_csv):
            overall_csv_path = os.path.join('static', 'reports', timestamp, 'overall_merged_data.csv')
        
        # Count stages
        stage_folders = sorted(
            [d for d in os.listdir(report_folder) if os.path.isdir(os.path.join(report_folder, d)) and d.startswith('step_')],
            key=lambda x: int(x.split('_')[1])
        )
        num_stages = len(stage_folders)
        
        # Get step report data
        step_reports = []
        for step_dir in stage_folders:
            step_num = int(step_dir.split('_')[1])
            step_folder = os.path.join(report_folder, step_dir)
            
            # Look for step-specific files
            step_plot_json = os.path.join(step_folder, f"step_{step_num}_plot.json")
            step_csv = os.path.join(step_folder, f"step_{step_num}_data.csv")
            step_json = os.path.join(step_folder, f"step_{step_num}_data.json")
            
            step_data = {
                'step_number': step_num,
                'plot_path': None,
                'csv_path': None,
                'json_path': None
            }
            
            if os.path.exists(step_plot_json):
                step_data['plot_path'] = os.path.join('static', 'reports', timestamp, step_dir, f"step_{step_num}_plot.json")
            
            if os.path.exists(step_csv):
                step_data['csv_path'] = os.path.join('static', 'reports', timestamp, step_dir, f"step_{step_num}_data.csv")
                
            if os.path.exists(step_json):
                step_data['json_path'] = os.path.join('static', 'reports', timestamp, step_dir, f"step_{step_num}_data.json")
            
            step_reports.append(step_data)
        
        # Get comparison plots
        comparison_plots_path = os.path.join(report_folder, 'comparison_plots')
        comparison_plots = []
        
        if os.path.exists(comparison_plots_path):
            comparison_files = [f for f in os.listdir(comparison_plots_path) if f.endswith('.json')]
            
            for comp_file in comparison_files:
                comp_path = os.path.join('static', 'reports', timestamp, 'comparison_plots', comp_file)
                comparison_plots.append({
                    'name': comp_file,
                    'path': comp_path
                })
        
        return jsonify({
            'success': True,
            'timestamp_prefix': timestamp,
            'overall_plot_path': overall_plot_path,
            'overall_csv_path': overall_csv_path,
            'num_stages': num_stages,
            'step_reports': step_reports,
            'comparison_plots': comparison_plots
        })
    except Exception as e:
        print(f"Error loading report: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error loading report: {e}'}), 500

# --- New Route for Stage Comparison ---
@app.route('/compare_stages', methods=['POST'])
def compare_stages():
    """Generates a comparison plot for selected stages."""
    try:
        data = request.get_json()
        timestamp = data.get('timestamp')
        stage_numbers = data.get('stages')
        comparison_prefix = data.get('comparison_prefix', '').strip() # Get the prefix

        print(f"Received request to compare stages in report {timestamp}: {stage_numbers}")
        print(f"Using comparison prefix: '{comparison_prefix}'")

        if not timestamp or not stage_numbers or not isinstance(stage_numbers, list) or len(stage_numbers) < 1:
            return jsonify({'success': False, 'message': 'Missing or invalid timestamp or stage numbers.'}), 400

        # Construct the base path for the report
        report_folder_abs = os.path.join(app.static_folder, 'reports', timestamp)
        if not os.path.isdir(report_folder_abs):
             return jsonify({'success': False, 'message': f'Report folder not found: {timestamp}'}), 404

        # Find the JSON data files for the selected stages
        stage_data_paths = []
        for stage_num in stage_numbers:
            json_file = os.path.join(report_folder_abs, f'step_{stage_num}', f'step_{stage_num}_data.json')
            if os.path.exists(json_file):
                stage_data_paths.append(json_file)
                print(f"Found JSON data file for stage {stage_num}: {json_file}")
            else:
                print(f"Warning: Data file not found for stage {stage_num} in report {timestamp}: {json_file}")
                # Decide if you want to fail or just proceed with available data
                # return jsonify({'success': False, 'message': f'Data file missing for stage {stage_num}'}), 404
        
        if not stage_data_paths:
             return jsonify({'success': False, 'message': 'No data files found for selected stages.'}), 404

        # Call the new function to generate the comparison plot
        # It needs the base output folder to save the plot
        try:
            print(f"Calling generate_comparison_plot with {len(stage_data_paths)} files and prefix '{comparison_prefix}'")
            comparison_plot_path = generate_comparison_plot(stage_data_paths, report_folder_abs, comparison_prefix) # Pass the prefix
        except Exception as plot_error:
            print(f"Error in generate_comparison_plot function: {plot_error}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'message': f'Error generating comparison plot: {plot_error}'}), 500

        if comparison_plot_path:
            print(f"Successfully generated comparison plot: {comparison_plot_path}")
            # Convert the absolute path to a web-accessible path (relative to static)
            static_folder_name = os.path.basename(app.static_folder)
            relative_path = os.path.relpath(comparison_plot_path, app.static_folder)
            web_path = os.path.join(static_folder_name, relative_path).replace('\\', '/')
            print(f"Web-accessible path: {web_path}")
            return jsonify({'success': True, 'comparison_plot_path': web_path})
        else:
            print("generate_comparison_plot returned None")
            return jsonify({'success': False, 'message': 'Failed to generate comparison plot.'}), 500

    except Exception as e:
        print(f"Error comparing stages: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error comparing stages: {e}'}), 500

# --- New Endpoint to List Comparison Plots in a Report Folder ---
@app.route('/list_comparison_plots/<timestamp>', methods=['GET'])
def list_comparison_plots_in_folder(timestamp):
    """Lists available *_stages_comparison_plot_*.json files within a specific report's comparison_plots directory."""
    try:
        report_folder_rel = os.path.join('reports', timestamp, 'comparison_plots')
        report_folder_abs = os.path.join(app.static_folder, report_folder_rel)

        if not os.path.isdir(report_folder_abs):
            return jsonify({'success': False, 'message': f'Comparison plots folder not found for report: {timestamp}'}), 404

        comparison_plot_files = []
        for filename in os.listdir(report_folder_abs):
            # Updated condition to correctly identify comparison plots with optional prefixes
            if '_stages_comparison_plot_' in filename and filename.endswith('.json'):
                # Store path relative to the specific timestamp's folder for easier reconstruction later
                # e.g., comparison_plots/stages_comparison_plot_XYZ.json
                relative_path_in_timestamp_folder = os.path.join('comparison_plots', filename)
                comparison_plot_files.append({
                    'name': filename, 
                    'path': relative_path_in_timestamp_folder.replace('\\', '/')
                })
        
        return jsonify({'success': True, 'comparison_plots': comparison_plot_files})
    except Exception as e:
        print(f"Error listing comparison plots for {timestamp}: {e}")
        return jsonify({'success': False, 'message': f'Error listing comparison plots: {e}'}), 500

# --- New Endpoint for Generating Cross-Report Comparison Plot ---
@app.route('/generate_cross_comparison', methods=['POST'])
def generate_cross_report_comparison():
    """Generates a cross-report comparison plot from selected sources."""
    try:
        data = request.get_json()
        selected_json_rel_paths = data.get('selected_comparison_json_paths', []) # e.g. ["static/reports/TS1/comp_plots/file.json", ...]
        current_report_ts = data.get('current_report_timestamp')
        current_selected_stages = data.get('current_report_selected_stages', [])

        print(f"Received cross-comparison request:")
        print(f"- Selected JSON paths: {selected_json_rel_paths}")
        print(f"- Current report: {current_report_ts}")
        print(f"- Selected stages: {current_selected_stages}")

        if not selected_json_rel_paths and not (current_report_ts and current_selected_stages):
            return jsonify({'success': False, 'message': 'No data sources provided for cross-comparison.'}), 400

        base_dir = os.path.dirname(os.path.abspath(__file__)) # Project root
        static_reports_folder_abs = os.path.join(base_dir, app.static_folder, 'reports') # Absolute path to static/reports

        absolute_selected_json_paths = []
        if selected_json_rel_paths:
            for rel_path in selected_json_rel_paths:
                # Clean up any URL artifacts in the rel_path
                rel_path = rel_path.replace('%20', ' ').replace('\\', '/')
                
                # rel_path is like "static/reports/TIMESTAMP/comparison_plots/FILENAME.json"
                # We need path from project root. os.path.join will handle this if base_dir is project root.
                abs_path = os.path.normpath(os.path.join(base_dir, rel_path))
                
                # Security check: ensure it's within the static_reports_folder_abs
                if not abs_path.startswith(static_reports_folder_abs):
                    print(f"Security Warning: Attempt to access path outside allowed reports directory: {rel_path}")
                    continue 
                
                if os.path.exists(abs_path):
                    absolute_selected_json_paths.append(abs_path)
                    print(f"Added JSON file for cross-comparison: {abs_path}")
                else:
                    print(f"Warning: Selected comparison JSON not found at {abs_path} (from relative {rel_path})")

        # Call the processing function from main_web_processor
        try:
            print(f"Calling create_cross_comparison_plot with {len(absolute_selected_json_paths)} files")
            cross_plot_abs_path = create_cross_comparison_plot(
                selected_comparison_json_file_paths=absolute_selected_json_paths,
                current_report_timestamp=current_report_ts,
                current_report_selected_stages=current_selected_stages,
                base_reports_folder_abs=static_reports_folder_abs 
            )
        except Exception as plot_error:
            print(f"Error in create_cross_comparison_plot function: {plot_error}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'message': f'Error generating cross-comparison plot: {plot_error}'}), 500

        if cross_plot_abs_path:
            print(f"Successfully generated cross-comparison plot: {cross_plot_abs_path}")
            # Convert the absolute path of the new plot to a web-accessible relative path (from project root)
            # e.g., static/reports/cross_comparisons/cross_comp_TIMESTAMP/plot.json
            web_path = os.path.relpath(cross_plot_abs_path, base_dir).replace('\\', '/')
            print(f"Web-accessible path: {web_path}")
            return jsonify({'success': True, 'cross_comparison_plot_path': web_path})
        else:
            print("create_cross_comparison_plot returned None")
            return jsonify({'success': False, 'message': 'Failed to generate cross-comparison plot.'}), 500

    except Exception as e:
        print(f"Error generating cross-comparison plot: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error generating cross-comparison plot: {e}'}), 500

# --- New Endpoint for Downloading Processed Cross-Comparison Data as CSV ---
@app.route('/download_cross_comparison_csv', methods=['POST'])
def download_cross_comparison_csv_data():
    """
    Receives processed and filtered data from the frontend (visible points in the cross-comparison plot)
    and sends back a combined CSV file.
    """
    try:
        data_rows = request.get_json() # Expects a list of objects

        if not data_rows or not isinstance(data_rows, list) or len(data_rows) == 0:
            return jsonify({'success': False, 'message': 'No data provided for CSV export.'}), 400
        
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(data_rows)

        if df.empty:
            return jsonify({'success': False, 'message': 'No data to export after processing.'}), 404

        # Convert DataFrame to CSV in memory
        csv_buffer = io.StringIO()
        # Define a specific order for columns if desired, otherwise it will be alphabetical or insertion order based on dict keys
        # For example: columns = ['Source', 'Stage', 'Parameter', 'RelativeTime', 'Value']
        # df.to_csv(csv_buffer, index=False, columns=columns) 
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Send the CSV data as a file download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        response = make_response(csv_buffer.getvalue())
        response.headers['Content-Disposition'] = f'attachment; filename=cross_comparison_visible_data_{timestamp}.csv'
        response.headers['Content-Type'] = 'text/csv'
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error creating cross-comparison CSV: {e}'}), 500

# --- New Endpoint to List All Comparison Plots ---
@app.route('/list_all_comparison_plots', methods=['GET'])
def list_all_comparison_plots():
    """Lists all available comparison plots from all report folders for cross-comparison."""
    try:
        reports_folder_abs = app.config['REPORTS_FOLDER']
        if not os.path.isdir(reports_folder_abs):
            return jsonify({'success': False, 'message': 'Reports folder not found.'}), 404

        all_comparison_plots = []
        
        # Get all report folders
        report_folders = [d for d in os.listdir(reports_folder_abs) 
                         if os.path.isdir(os.path.join(reports_folder_abs, d)) and d != 'cross_comparisons']
        
        # Check each report folder for comparison_plots
        for report_folder in report_folders:
            comparison_plots_dir = os.path.join(reports_folder_abs, report_folder, 'comparison_plots')
            if os.path.isdir(comparison_plots_dir):
                for filename in os.listdir(comparison_plots_dir):
                    if '_stages_comparison_plot_' in filename and filename.endswith('.json'):
                        # Store relative path from static folder for frontend use
                        static_folder_name = os.path.basename(app.static_folder)
                        rel_path = os.path.join(static_folder_name, 'reports', report_folder, 'comparison_plots', filename)
                        rel_path = rel_path.replace('\\', '/')
                        
                        all_comparison_plots.append({
                            'name': filename,
                            'report_folder': report_folder,
                            'path': rel_path,
                            'display_name': f"{report_folder} - {filename}"
                        })
        
        return jsonify({'success': True, 'comparison_plots': all_comparison_plots})
    except Exception as e:
        print(f"Error listing all comparison plots: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error listing all comparison plots: {e}'}), 500

# --- Routes for Report Management ---
@app.route('/rename_report', methods=['POST'])
def rename_report():
    """Renames a report folder."""
    try:
        data = request.get_json()
        old_name = data.get('old_name')
        new_name = data.get('new_name')
        
        if not old_name or not new_name:
            return jsonify({'success': False, 'message': 'Missing old or new report name.'}), 400
        
        # Sanitize input to prevent directory traversal attacks
        old_name = secure_filename(old_name)
        new_name = secure_filename(new_name)
        
        old_path = os.path.join(app.config['REPORTS_FOLDER'], old_name)
        new_path = os.path.join(app.config['REPORTS_FOLDER'], new_name)
        
        # Check if old folder exists
        if not os.path.isdir(old_path):
            return jsonify({'success': False, 'message': f'Report folder not found: {old_name}'}), 404
        
        # Check if new folder name already exists
        if os.path.exists(new_path):
            return jsonify({'success': False, 'message': f'A report with the name {new_name} already exists.'}), 409
        
        # Rename the folder
        os.rename(old_path, new_path)
        
        return jsonify({'success': True, 'message': f'Report renamed from {old_name} to {new_name}'})
    
    except Exception as e:
        print(f"Error renaming report: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error renaming report: {e}'}), 500

@app.route('/delete_report', methods=['POST'])
def delete_report():
    """Deletes a report folder and all its contents."""
    try:
        data = request.get_json()
        report_name = data.get('report_name')
        
        if not report_name:
            return jsonify({'success': False, 'message': 'Missing report name.'}), 400
        
        # Sanitize input to prevent directory traversal attacks
        report_name = secure_filename(report_name)
        
        report_path = os.path.join(app.config['REPORTS_FOLDER'], report_name)
        
        # Check if folder exists
        if not os.path.isdir(report_path):
            return jsonify({'success': False, 'message': f'Report folder not found: {report_name}'}), 404
        
        # Use shutil.rmtree to remove the directory and all its contents
        import shutil
        shutil.rmtree(report_path)
        
        return jsonify({'success': True, 'message': f'Report {report_name} deleted successfully'})
    
    except Exception as e:
        print(f"Error deleting report: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error deleting report: {e}'}), 500

@app.route('/list_report_contents/<report_name>', methods=['GET'])
def list_report_contents(report_name):
    """Lists the contents of a report folder."""
    try:
        # Sanitize input to prevent directory traversal attacks
        report_name = secure_filename(report_name)
        
        report_path = os.path.join(app.config['REPORTS_FOLDER'], report_name)
        
        # Check if folder exists
        if not os.path.isdir(report_path):
            return jsonify({'success': False, 'message': f'Report folder not found: {report_name}'}), 404
        
        # Function to recursively get directory structure
        def get_directory_structure(dir_path, base_path):
            items = []
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                # Get relative path for frontend use
                rel_path = os.path.relpath(item_path, base_path)
                
                if os.path.isdir(item_path):
                    # If it's a directory, recurse
                    children = get_directory_structure(item_path, base_path)
                    items.append({
                        'name': item,
                        'type': 'folder',
                        'path': rel_path.replace('\\', '/'),
                        'children': children
                    })
                else:
                    # If it's a file, add its info
                    file_size = os.path.getsize(item_path)
                    size_display = f"{file_size} bytes"
                    if file_size > 1024:
                        size_display = f"{file_size / 1024:.2f} KB"
                    if file_size > 1024 * 1024:
                        size_display = f"{file_size / (1024 * 1024):.2f} MB"
                    
                    items.append({
                        'name': item,
                        'type': 'file',
                        'path': rel_path.replace('\\', '/'),
                        'size': size_display
                    })
            
            # Sort folders first, then files
            return sorted(items, key=lambda x: (0 if x['type'] == 'folder' else 1, x['name']))
        
        # Get the directory structure
        contents = get_directory_structure(report_path, os.path.dirname(report_path))
        
        return jsonify({
            'success': True, 
            'report_name': report_name,
            'contents': contents
        })
    
    except Exception as e:
        print(f"Error listing report contents: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error listing report contents: {e}'}), 500

@app.route('/get_file_content', methods=['POST'])
def get_file_content():
    """Gets the content of a file in a report folder."""
    try:
        data = request.get_json()
        report_name = data.get('report_name')
        file_path = data.get('file_path')
        
        if not report_name or not file_path:
            return jsonify({'success': False, 'message': 'Missing report name or file path.'}), 400
        
        # Sanitize input to prevent directory traversal attacks
        report_name = secure_filename(report_name)
        
        # Construct the full path
        report_folder = os.path.join(app.config['REPORTS_FOLDER'], report_name)
        full_path = os.path.join(report_folder, file_path)
        
        # Verify the path is within the report folder
        if not os.path.abspath(full_path).startswith(os.path.abspath(report_folder)):
            return jsonify({'success': False, 'message': 'Invalid file path.'}), 403
        
        # Check if file exists
        if not os.path.isfile(full_path):
            return jsonify({'success': False, 'message': f'File not found: {file_path}'}), 404
        
        # Determine file type
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Check if it's a binary file
        binary_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.pdf']
        if file_extension in binary_extensions:
            return jsonify({
                'success': True,
                'is_binary': True,
                'file_path': file_path,
                'message': 'Binary file cannot be displayed directly.'
            })
        
        # For text-based files (JSON, CSV, TXT, etc.), read the content
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return jsonify({
                'success': True,
                'is_binary': False,
                'file_path': file_path,
                'content': content
            })
        except UnicodeDecodeError:
            # If we get a unicode error, it might be a binary file
            return jsonify({
                'success': True,
                'is_binary': True,
                'file_path': file_path,
                'message': 'Binary file cannot be displayed directly.'
            })
    
    except Exception as e:
        print(f"Error getting file content: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error getting file content: {e}'}), 500

@app.route('/delete_report_file', methods=['POST'])
def delete_report_file():
    """Deletes a file from a report folder."""
    try:
        data = request.get_json()
        report_name = data.get('report_name')
        file_path = data.get('file_path')
        
        if not report_name or not file_path:
            return jsonify({'success': False, 'message': 'Missing report name or file path.'}), 400
        
        # Sanitize input to prevent directory traversal attacks
        report_name = secure_filename(report_name)
        
        # Construct the full path
        report_folder = os.path.join(app.config['REPORTS_FOLDER'], report_name)
        full_path = os.path.join(report_folder, file_path)
        
        # Verify the path is within the report folder
        if not os.path.abspath(full_path).startswith(os.path.abspath(report_folder)):
            return jsonify({'success': False, 'message': 'Invalid file path.'}), 403
        
        # Check if file exists
        if not os.path.isfile(full_path):
            return jsonify({'success': False, 'message': f'File not found: {file_path}'}), 404
        
        # Delete the file
        os.remove(full_path)
        
        return jsonify({
            'success': True,
            'message': f'File {file_path} deleted successfully.'
        })
    
    except Exception as e:
        print(f"Error deleting file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error deleting file: {e}'}), 500

# --- New API Endpoints for Enhanced Index Page ---
@app.route('/api/detailed-reports-list')
def api_detailed_reports_list():
    """Returns detailed information about all reports including data points, date ranges, etc."""
    try:
        reports_base = app.config['REPORTS_FOLDER']
        
        # List only directories directly under the reports folder
        all_items = os.listdir(reports_base)
        report_folders = [d for d in all_items if os.path.isdir(os.path.join(reports_base, d))]
        
        # Sort by creation time, newest first
        report_folders.sort(key=lambda x: os.path.getctime(os.path.join(reports_base, x)), reverse=True)
        
        reports_data = []
        for folder in report_folders:
            # Skip non-report folders
            if folder in ['cross_comparisons']:
                continue
                
            # Get basic stats for each report
            folder_path = os.path.join(reports_base, folder)
            created_at = datetime.fromtimestamp(os.path.getctime(folder_path)).strftime('%Y-%m-%d %H:%M:%S')
            
            # Count number of stages by looking for step_* directories
            step_folders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d)) and d.startswith('step_')]
            step_folders.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)
            
            # Get GC correction factor from metadata if available
            gc_correction_factor = None
            metadata_file_path = os.path.join(folder_path, 'catalyst_metadata.json')
            if os.path.exists(metadata_file_path):
                try:
                    with open(metadata_file_path, 'r') as f:
                        metadata = json.load(f)
                        gc_correction_factor = metadata.get('gc_correction_factor')
                except Exception as e:
                    print(f"Error reading metadata for {folder}: {e}")
            
            # Calculate total data points and date range
            total_data_points = 0
            date_start = None
            date_end = None
            
            # Get data from overall CSV if it exists
            overall_csv_path = os.path.join(folder_path, 'overall_merged_data.csv')
            if os.path.exists(overall_csv_path):
                try:
                    df = pd.read_csv(overall_csv_path)
                    total_data_points = len(df)
                    
                    if 'Date' in df.columns:
                        # Convert to datetime if it's a string
                        if df['Date'].dtype == 'object':
                            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                            
                        date_values = df['Date'].dropna()
                        if not date_values.empty:
                            date_start = date_values.min().strftime('%Y-%m-%d %H:%M:%S')
                            date_end = date_values.max().strftime('%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    print(f"Error reading overall CSV for {folder}: {e}")
                    # If we can't read the CSV, try to get data from individual stages
            
            # If no data from overall CSV, collect from individual stages
            if total_data_points == 0 or date_start is None:
                for step_folder in step_folders:
                    step_num = int(step_folder.split('_')[1]) if step_folder.split('_')[1].isdigit() else 0
                    step_json = os.path.join(folder_path, step_folder, f"step_{step_num}_data.json")
                    
                    if os.path.exists(step_json):
                        try:
                            with open(step_json, 'r') as f:
                                step_data = json.load(f)
                            
                            total_data_points += len(step_data)
                            
                            # Extract date range
                            dates = []
                            for entry in step_data:
                                if 'Date' in entry:
                                    try:
                                        # Handle ISO format with Z
                                        if isinstance(entry['Date'], str) and 'Z' in entry['Date']:
                                            date_obj = datetime.fromisoformat(entry['Date'].replace('Z', '+00:00'))
                                            dates.append(date_obj)
                                        else:
                                            # Try to parse using pandas which handles multiple formats
                                            date_obj = pd.to_datetime(entry['Date'])
                                            if not pd.isna(date_obj):
                                                dates.append(date_obj.to_pydatetime())
                                    except Exception as e:
                                        # Just skip this date if it can't be parsed
                                        print(f"Error parsing date {entry.get('Date')}: {e}")
                            
                            if dates:
                                min_date = min(dates)
                                max_date = max(dates)
                                
                                # Only update if not already set or this is earlier/later
                                if date_start is None:
                                    date_start = min_date.strftime('%Y-%m-%d %H:%M:%S')
                                else:
                                    try:
                                        current_start = datetime.strptime(date_start, '%Y-%m-%d %H:%M:%S')
                                        if min_date < current_start:
                                            date_start = min_date.strftime('%Y-%m-%d %H:%M:%S')
                                    except:
                                        date_start = min_date.strftime('%Y-%m-%d %H:%M:%S')
                                
                                if date_end is None:
                                    date_end = max_date.strftime('%Y-%m-%d %H:%M:%S')
                                else:
                                    try:
                                        current_end = datetime.strptime(date_end, '%Y-%m-%d %H:%M:%S')
                                        if max_date > current_end:
                                            date_end = max_date.strftime('%Y-%m-%d %H:%M:%S')
                                    except:
                                        date_end = max_date.strftime('%Y-%m-%d %H:%M:%S')
                        except Exception as e:
                            print(f"Error processing JSON for stage {step_num} in report {folder}: {e}")
            
            reports_data.append({
                'name': folder,
                'created_at': created_at,
                'stages': len(step_folders),
                'data_points': total_data_points,
                'date_start': date_start,
                'date_end': date_end,
                'gc_correction_factor': gc_correction_factor
            })
        
        return jsonify({'success': True, 'reports': reports_data})
    except Exception as e:
        print(f"Error listing detailed reports: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error listing reports: {e}'}), 500

@app.route('/api/original-files')
def api_original_files():
    """Lists all files in the uploads folder with their types and sizes."""
    try:
        uploads_folder = app.config['UPLOAD_FOLDER']
        
        # Get all files in the uploads folder
        files = []
        for filename in os.listdir(uploads_folder):
            file_path = os.path.join(uploads_folder, filename)
            
            # Skip directories and hidden files
            if os.path.isdir(file_path) or filename.startswith('.'):
                continue
                
            # Get file size
            size_bytes = os.path.getsize(file_path)
            
            # Format file size
            if size_bytes == 0:
                size_str = '0 Bytes'
            else:
                k = 1024
                sizes = ['Bytes', 'KB', 'MB', 'GB']
                i = int(math.log(size_bytes, k))
                size_str = f"{size_bytes / (k ** i):.2f} {sizes[i]}"
            
            files.append({
                'name': filename,
                'size': size_str,
                'size_bytes': size_bytes
            })
        
        # Sort by size (largest first)
        files.sort(key=lambda x: x['size_bytes'], reverse=True)
        
        return jsonify({'success': True, 'files': files})
    except Exception as e:
        print(f"Error listing original files: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error listing files: {e}'}), 500

@app.route('/download_original_file/<filename>')
def download_original_file(filename):
    """Allows downloading the original files from uploads folder."""
    try:
        # Sanitize filename to prevent path traversal
        filename = secure_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Check if file exists
        if not os.path.isfile(file_path):
            return jsonify({'success': False, 'message': f'File not found: {filename}'}), 404
        
        # Send file for download
        return send_file(file_path, as_attachment=True, download_name=filename)
    except Exception as e:
        print(f"Error downloading file {filename}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error downloading file: {e}'}), 500

@app.route('/download_report_original_file/<report_name>/<file_type>')
def download_report_original_file(report_name, file_type):
    """Download original LV or GC files from a specific report folder."""
    try:
        # Secure the report name to prevent directory traversal
        report_name = secure_filename(report_name)
        
        # Validate file type
        if file_type not in ['LV', 'GC']:
            return jsonify({'success': False, 'message': 'Invalid file type. Must be LV or GC.'}), 400
        
        # Get report folder path
        report_folder = os.path.join(REPORTS_FOLDER, report_name)
        if not os.path.exists(report_folder):
            return jsonify({'success': False, 'message': 'Report not found'}), 404
        
        # Load metadata to get the correct filename
        metadata_file_path = os.path.join(report_folder, 'catalyst_metadata.json')
        if not os.path.exists(metadata_file_path):
            return jsonify({'success': False, 'message': 'Report metadata not found'}), 404
        
        with open(metadata_file_path, 'r') as f:
            metadata = json.load(f)
        
        experiment_number = metadata.get('experiment_number', 'Unknown')
        catalyst_batch = metadata.get('catalyst_batch', 'Unknown')
        reactor_number = metadata.get('reactor_number', 'Unknown')
        
        # Generate the expected filename
        base_filename = f"{experiment_number}_{catalyst_batch}_{reactor_number}"
        filename = f"{base_filename}_{file_type}.txt"
        
        # Check if file exists
        file_path = os.path.join(report_folder, filename)
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'message': f'Original {file_type} file not found in report'}), 404
        
        return send_file(file_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        print(f"Error downloading report original file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error downloading file: {e}'}), 500

# --- Routes for Database Management ---
@app.route('/api/database/generate', methods=['POST'])
def generate_database():
    """Generate the database from all reports."""
    recreate = request.json.get('recreate', False) if request.is_json else False
    
    try:
        # Initialize the database processor
        processor = NH3SynthDatabaseProcessor(
            reports_dir=app.config['REPORTS_FOLDER'],
            db_path=app.config['DATABASE_PATH'],
            recreate_db=recreate
        )
        
        # Process all reports
        processor.process_all_reports()
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error generating database: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/database/experiments')
def api_list_experiments():
    """Lists all experiments in the database."""
    try:
        processor = NH3SynthDatabaseProcessor(
            reports_dir=app.config['REPORTS_FOLDER'],
            db_path=app.config['DATABASE_PATH']
        )
        
        # Get all experiments
        experiments_df = processor.list_experiments()
        
        # Convert DataFrame to list of dicts for JSON response
        experiments = experiments_df.to_dict(orient='records')
        
        return jsonify({
            'success': True,
            'experiments': experiments
        })
    except Exception as e:
        print(f"Error listing experiments: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error listing experiments: {e}'}), 500

@app.route('/api/database/experiment/<experiment_id>')
def api_get_experiment(experiment_id):
    """Gets detailed information about a specific experiment."""
    try:
        processor = NH3SynthDatabaseProcessor(
            reports_dir=app.config['REPORTS_FOLDER'],
            db_path=app.config['DATABASE_PATH']
        )
        
        # Get experiment details
        experiment = processor.get_experiment_details(experiment_id)
        
        if not experiment:
            return jsonify({'success': False, 'message': f'Experiment {experiment_id} not found.'}), 404
        
        return jsonify({
            'success': True,
            'experiment': experiment
        })
    except Exception as e:
        print(f"Error getting experiment {experiment_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error getting experiment details: {e}'}), 500

@app.route('/api/database/experiment/<experiment_id>/steps')
def api_get_experiment_steps(experiment_id):
    """Gets information about the steps in a specific experiment."""
    try:
        processor = NH3SynthDatabaseProcessor(
            reports_dir=app.config['REPORTS_FOLDER'],
            db_path=app.config['DATABASE_PATH']
        )
        
        # Get steps information
        steps = processor.analyze_experiment_steps(experiment_id)
        
        if not steps:
            return jsonify({'success': False, 'message': f'Steps for experiment {experiment_id} not found.'}), 404
        
        return jsonify({
            'success': True,
            'steps': steps
        })
    except Exception as e:
        print(f"Error getting steps for experiment {experiment_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error getting experiment steps: {e}'}), 500

@app.route('/api/database/add-to-database/<report_name>', methods=['POST'])
def api_add_report_to_database(report_name):
    """Adds a specific report to the database."""
    try:
        # Sanitize report name
        report_name = secure_filename(report_name)
        report_path = os.path.join(app.config['REPORTS_FOLDER'], report_name)
        
        if not os.path.isdir(report_path):
            return jsonify({'success': False, 'message': f'Report {report_name} not found.'}), 404
        
        # Create processor instance
        processor = NH3SynthDatabaseProcessor(
            reports_dir=app.config['REPORTS_FOLDER'],
            db_path=app.config['DATABASE_PATH']
        )
        
        # Check if this experiment already exists in the database
        existing_experiment = processor.session.query(Experiment).filter_by(report_folder=report_name).first()
        
        if existing_experiment:
            return jsonify({
                'success': False, 
                'message': f'Experiment {report_name} already exists in database.',
                'experiment_id': existing_experiment.id
            }), 409
        
        # Process this specific report
        # Since the processor doesn't have a direct method to process a single report,
        # we'll implement our own logic here based on the processor's code
        
        # Read metadata
        metadata = processor.read_metadata_json(Path(report_path))
        if not metadata:
            return jsonify({'success': False, 'message': f'Failed to read metadata for {report_name}.'}), 500
        
        # Extract plot summary
        plot_summary_data = processor.extract_plot_summary(Path(report_path))
        
        # Create experiment record
        experiment = Experiment(
            report_folder=report_name,
            experiment_id=metadata.get('experiment_id', ''),
            reactor_id=metadata.get('reactor_number', ''),
            date=metadata.get('created_at', ''),
            time=metadata.get('folder_timestamp', ''),
            experiment_metadata=metadata
        )
        
        # Create plot summary record
        plot_summary = PlotSummary(
            plot_file_size=plot_summary_data.get('plot_file_size', 0),
            num_data_points=plot_summary_data.get('num_data_points', 0),
            parameters=json.dumps(plot_summary_data.get('parameters', [])),
            first_timestamp=plot_summary_data.get('first_timestamp'),
            last_timestamp=plot_summary_data.get('last_timestamp')
        )
        
        # Associate plot summary with experiment
        experiment.plot_summary = plot_summary
        
        # Analyze experiment steps
        step_analysis = processor.analyze_experiment_steps_data(Path(report_path))
        
        # Create step records and their data points
        total_data_points = 0
        for step_num, step_data in step_analysis.get('steps', {}).items():
            step = ExperimentStep(
                step_number=int(step_num),
                folder=step_data.get('folder', ''),
                has_data_json=1 if step_data.get('has_data_json') else 0,
                has_plot_json=1 if step_data.get('has_plot_json') else 0,
                data_file_size=step_data.get('data_file_size', 0),
                plot_file_size=step_data.get('plot_file_size', 0),
                temperature=step_data.get('T Reactor (sliding TC)'),
                pressure=step_data.get('Pressure reading'),
                h2_flow=step_data.get('H2 Actual Flow'),
                n2_flow=step_data.get('N2 Actual Flow'),
                stage=step_data.get('Stage'),
                data_points_count=len(step_data.get('data_points', []))
            )
            
            # Add to experiment
            experiment.steps.append(step)
            
            # Create data point records
            for point_data in step_data.get('data_points', []):
                try:
                    # Parse timestamp string to datetime object
                    timestamp = None
                    if point_data.get('timestamp'):
                        try:
                            timestamp = datetime.fromisoformat(point_data['timestamp'])
                        except (ValueError, TypeError):
                            # Try with different format if ISO format fails
                            try:
                                timestamp = datetime.strptime(point_data['timestamp'], '%Y-%m-%d %H:%M:%S')
                            except (ValueError, TypeError):
                                pass  # Leave as None if parsing fails
                    
                    # Create data point record
                    data_point = StepDataPoint(
                        relative_time=point_data.get('relative_time'),
                        timestamp=timestamp,
                        t_reactor=point_data.get('t_reactor'),
                        pressure=point_data.get('pressure'),
                        h2_flow=point_data.get('h2_flow'),
                        n2_flow=point_data.get('n2_flow'),
                        nh3_concentration=point_data.get('nh3_concentration'),
                        outlet_flow=point_data.get('outlet_flow'),
                        raw_data=point_data.get('raw_data')
                    )
                    
                    # Add to step
                    step.data_points.append(data_point)
                    total_data_points += 1
                    
                except Exception as point_error:
                    print(f"Error processing data point in step {step_num}: {str(point_error)}")
                    continue
        
        # Add to session and commit
        processor.session.add(experiment)
        processor.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Added experiment {report_name} to database',
            'experiment_id': experiment.id
        })
    except Exception as e:
        print(f"Error adding report {report_name} to database: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error adding report to database: {e}'}), 500

@app.route('/api/database/download/<format>')
def api_download_database(format):
    """Downloads the database in CSV or JSON format."""
    try:
        if format not in ['csv', 'json']:
            return jsonify({'success': False, 'message': 'Invalid format. Use csv or json.'}), 400
        
        processor = NH3SynthDatabaseProcessor(
            reports_dir=app.config['REPORTS_FOLDER'],
            db_path=app.config['DATABASE_PATH']
        )
        
        if format == 'csv':
            path = processor.export_database_to_csv()
            if not path:
                return jsonify({'success': False, 'message': 'Failed to generate CSV export.'}), 500
            return send_file(path, as_attachment=True, download_name='nh3_synth_database.csv')
        else:
            path = processor.export_database_to_json()
            if not path:
                return jsonify({'success': False, 'message': 'Failed to generate JSON export.'}), 500
            return send_file(path, as_attachment=True, download_name='nh3_synth_database.json')
    except Exception as e:
        print(f"Error downloading database in {format} format: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error downloading database: {e}'}), 500

@app.route('/database')
def database_page():
    """Serves the database management page."""
    return render_template('database.html')

@app.route('/api/database/experiment/<experiment_id>/step/<int:step_number>/data')
def api_get_step_data_points(experiment_id, step_number):
    """Gets data points for a specific step in an experiment."""
    try:
        processor = NH3SynthDatabaseProcessor(
            reports_dir=app.config['REPORTS_FOLDER'],
            db_path=app.config['DATABASE_PATH']
        )
        
        # Get data points
        data_points = processor.get_step_data_points(experiment_id, step_number)
        
        if not data_points:
            return jsonify({'success': False, 'message': f'No data points found for step {step_number} in experiment {experiment_id}.'}), 404
        
        # Limit the number of data points returned to avoid overwhelming the client
        max_points = 1000
        if len(data_points) > max_points:
            sampled_points = []
            step = len(data_points) // max_points
            for i in range(0, len(data_points), step):
                sampled_points.append(data_points[i])
            data_points = sampled_points[:max_points]
            sampled = True
        else:
            sampled = False
        
        return jsonify({
            'success': True,
            'step_number': step_number,
            'experiment_id': experiment_id,
            'data_points': data_points,
            'total_points': len(data_points),
            'sampled': sampled
        })
    except Exception as e:
        print(f"Error getting data points for step {step_number} in experiment {experiment_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error getting step data points: {e}'}), 500

@app.route('/api/database/experiment/<experiment_id>/step/<int:step_number>/export')
def api_export_step_data(experiment_id, step_number):
    """Exports data points for a specific step to a CSV file."""
    try:
        processor = NH3SynthDatabaseProcessor(
            reports_dir=app.config['REPORTS_FOLDER'],
            db_path=app.config['DATABASE_PATH']
        )
        
        # Export to CSV
        csv_path = processor.export_step_data_to_csv(experiment_id, step_number)
        
        if not csv_path:
            return jsonify({'success': False, 'message': f'Failed to export data for step {step_number} in experiment {experiment_id}.'}), 500
        
        # Return file for download
        return send_file(csv_path, as_attachment=True, download_name=f'{experiment_id}_step_{step_number}_data.csv')
    except Exception as e:
        print(f"Error exporting data for step {step_number} in experiment {experiment_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error exporting step data: {e}'}), 500

@app.route('/api/database/experiment/<experiment_id>/export')
def api_export_experiment_data(experiment_id):
    """Exports all data points for an experiment to a CSV file."""
    try:
        processor = NH3SynthDatabaseProcessor(
            reports_dir=app.config['REPORTS_FOLDER'],
            db_path=app.config['DATABASE_PATH']
        )
        
        # Export to CSV
        csv_path = processor.export_experiment_data_to_csv(experiment_id)
        
        if not csv_path:
            return jsonify({'success': False, 'message': f'Failed to export data for experiment {experiment_id}.'}), 500
        
        # Return file for download
        return send_file(csv_path, as_attachment=True, download_name=f'{experiment_id}_all_data.csv')
    except Exception as e:
        print(f"Error exporting data for experiment {experiment_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error exporting experiment data: {e}'}), 500

@app.route('/api/database/experiments/<int:experiment_id>/steps/<int:step_id>/datapoints', methods=['GET'])
def get_step_data_points(experiment_id, step_id):
    """Get data points for a specific step."""
    limit = request.args.get('limit', default=1000, type=int)
    offset = request.args.get('offset', default=0, type=int)
    
    # Initialize database
    db_path = os.path.join(app.config['DATABASE_FOLDER'], 'nh3_synth.db')
    db = NH3SynthDatabaseProcessor(db_path=db_path)
    
    # Get data points
    data_points = db.get_step_data_points(step_id, limit, offset)
    
    return jsonify(data_points)

@app.route('/api/database/experiments/<int:experiment_id>/steps/<int:step_id>/export/csv', methods=['GET'])
def export_step_data_csv(experiment_id, step_id):
    """Export step data points to CSV."""
    # Initialize database
    db_path = os.path.join(app.config['DATABASE_FOLDER'], 'nh3_synth.db')
    db = NH3SynthDatabaseProcessor(db_path=db_path)
    
    # Get experiment to use in filename
    experiment = db.get_experiment_by_id(experiment_id)
    if not experiment:
        return jsonify({'error': 'Experiment not found'}), 404
    
    # Create exports folder if it doesn't exist
    exports_folder = os.path.join(app.config['DATABASE_FOLDER'], 'exports')
    if not os.path.exists(exports_folder):
        os.makedirs(exports_folder)
    
    # Generate export filename
    filename = f"{experiment['report_folder']}_step_{step_id}_data.csv"
    export_path = os.path.join(exports_folder, filename)
    
    # Export data
    success = db.export_step_data_to_csv(step_id, export_path)
    
    if success:
        return jsonify({'success': True, 'file_path': export_path, 'filename': filename})
    else:
        return jsonify({'error': 'Failed to export data'}), 500

@app.route('/api/database/experiments/<int:experiment_id>/steps/<int:step_id>/export/json', methods=['GET'])
def export_step_data_json(experiment_id, step_id):
    """Export step data points to JSON."""
    # Initialize database
    db_path = os.path.join(app.config['DATABASE_FOLDER'], 'nh3_synth.db')
    db = NH3SynthDatabaseProcessor(db_path=db_path)
    
    # Get experiment to use in filename
    experiment = db.get_experiment_by_id(experiment_id)
    if not experiment:
        return jsonify({'error': 'Experiment not found'}), 404
    
    # Create exports folder if it doesn't exist
    exports_folder = os.path.join(app.config['DATABASE_FOLDER'], 'exports')
    if not os.path.exists(exports_folder):
        os.makedirs(exports_folder)
    
    # Generate export filename
    filename = f"{experiment['report_folder']}_step_{step_id}_data.json"
    export_path = os.path.join(exports_folder, filename)
    
    # Export data
    success = db.export_step_data_to_json(step_id, export_path)
    
    if success:
        return jsonify({'success': True, 'file_path': export_path, 'filename': filename})
    else:
        return jsonify({'error': 'Failed to export data'}), 500

@app.route('/api/database/experiments/<int:experiment_id>/steps/<int:step_id>/download/csv', methods=['GET'])
def download_step_data_csv(experiment_id, step_id):
    """Download step data points as CSV."""
    # Initialize database
    db_path = os.path.join(app.config['DATABASE_FOLDER'], 'nh3_synth.db')
    db = NH3SynthDatabaseProcessor(db_path=db_path)
    
    # Get experiment to use in filename
    experiment = db.get_experiment_by_id(experiment_id)
    if not experiment:
        return jsonify({'error': 'Experiment not found'}), 404
    
    # Create temporary file
    temp_file = os.path.join(tempfile.gettempdir(), f"{experiment['report_folder']}_step_{step_id}_data.csv")
    
    # Export data
    success = db.export_step_data_to_csv(step_id, temp_file)
    
    if success:
        return send_file(temp_file, as_attachment=True, download_name=f"{experiment['report_folder']}_step_{step_id}_data.csv")
    else:
        return jsonify({'error': 'Failed to export data'}), 500

@app.route('/api/database/experiments/<int:experiment_id>/steps/<int:step_id>/download/json', methods=['GET'])
def download_step_data_json(experiment_id, step_id):
    """Download step data points as JSON."""
    # Initialize database
    db_path = os.path.join(app.config['DATABASE_FOLDER'], 'nh3_synth.db')
    db = NH3SynthDatabaseProcessor(db_path=db_path)
    
    # Get experiment to use in filename
    experiment = db.get_experiment_by_id(experiment_id)
    if not experiment:
        return jsonify({'error': 'Experiment not found'}), 404
    
    # Create temporary file
    temp_file = os.path.join(tempfile.gettempdir(), f"{experiment['report_folder']}_step_{step_id}_data.json")
    
    # Export data
    success = db.export_step_data_to_json(step_id, temp_file)
    
    if success:
        return send_file(temp_file, as_attachment=True, download_name=f"{experiment['report_folder']}_step_{step_id}_data.json")
    else:
        return jsonify({'error': 'Failed to export data'}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # Run the Flask development server
    # Debug=True allows for automatic reloading on code changes and provides detailed error pages
    # Use host='0.0.0.0' to make it accessible on your local network
    app.run(debug=True, host='0.0.0.0', port=6007) 