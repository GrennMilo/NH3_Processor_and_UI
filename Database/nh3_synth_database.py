#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NH3 Synthesis Database Processor

This module processes NH3 synthesis experiment data stored in the reports directory.
It extracts metadata from catalyst_metadata.json files and merges it with 
data from overall_plot.json files to create a unified database of experiments.
"""

import os
import json
import pandas as pd
import re
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Any, Union, Optional, Tuple
from sqlalchemy import and_, or_, func
from Database.database_models import (
    init_db, get_session, Base, 
    Experiment, PlotSummary, ExperimentStep, StepDataPoint
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class NH3SynthDatabaseProcessor:
    """
    Class to process and organize NH3 synthesis experiment data from report folders.
    """
    
    def __init__(self, reports_dir: str = "static/reports", db_path: str = "database/nh3_synth.db", recreate_db: bool = False):
        """
        Initialize the NH3SynthDatabaseProcessor.
        
        Args:
            reports_dir: Directory containing the NH3 synthesis reports
            db_path: Path to the SQLite database file
            recreate_db: Whether to drop and recreate all database tables
        """
        self.reports_dir = Path(reports_dir)
        self.db_path = db_path
        
        # Initialize database
        engine = init_db(db_path, recreate=recreate_db)
        self.session = get_session(engine)
        
        logger.info(f"Initialized database at {db_path}")
        
        # Ensure reports directory exists
        if not self.reports_dir.exists():
            logger.warning(f"Reports directory {reports_dir} does not exist, creating it")
            self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def scan_report_folders(self) -> List[Path]:
        """
        Scan the reports directory and return a list of valid report folders.
        
        Returns:
            List of folder paths that contain both catalyst_metadata.json and overall_plot.json
        """
        valid_folders = []
        
        try:
            for item in self.reports_dir.iterdir():
                if not item.is_dir() or item.name.startswith('.'):
                    continue
                
                # Check if folder contains required files
                metadata_file = item / "catalyst_metadata.json"
                plot_file = item / "overall_plot.json"
                
                if metadata_file.exists() and plot_file.exists():
                    valid_folders.append(item)
                    logger.debug(f"Found valid report folder: {item.name}")
                else:
                    logger.warning(f"Skipping folder {item.name}: Missing required files")
        
        except Exception as e:
            logger.error(f"Error scanning report folders: {str(e)}")
        
        logger.info(f"Found {len(valid_folders)} valid report folders")
        return valid_folders
    
    def read_metadata_json(self, folder_path: Path) -> Dict:
        """
        Read and parse the catalyst_metadata.json file from a report folder.
        
        Args:
            folder_path: Path to the report folder
            
        Returns:
            Dictionary containing the metadata
        """
        metadata_file = folder_path / "catalyst_metadata.json"
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Add the folder name to the metadata
            metadata['report_folder'] = folder_path.name
            
            # Parse experiment info from folder name
            folder_parts = folder_path.name.split('_')
            if len(folder_parts) >= 4:
                # The format is expected to be: NNN_XXXX_RYYY_YYYYMMDD_HHMMSS
                # Where NNN is experiment number, XXXX is catalyst batch, RYYY is reactor number
                metadata['experiment_id'] = folder_path.name
                
                # Extract timestamp from folder name if available
                if len(folder_parts) >= 5:
                    try:
                        date_str = folder_parts[3]
                        time_str = folder_parts[4]
                        if len(date_str) == 8 and len(time_str) == 6:
                            timestamp = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                            metadata['folder_timestamp'] = timestamp
                    except Exception:
                        pass
            
            return metadata
        
        except Exception as e:
            logger.error(f"Error reading metadata from {metadata_file}: {str(e)}")
            return {}
    
    def extract_plot_summary(self, folder_path: Path) -> Dict:
        """
        Extract summary information from the overall_plot.json file.
        Instead of loading the entire file (which can be very large),
        this extracts key information about the experiment.
        
        Args:
            folder_path: Path to the report folder
            
        Returns:
            Dictionary containing summary data from the plot file
        """
        plot_file = folder_path / "overall_plot.json"
        summary = {
            'plot_file_size': 0,
            'num_data_points': 0,
            'parameters': [],
            'first_timestamp': None,
            'last_timestamp': None
        }
        
        try:
            # Get file size
            summary['plot_file_size'] = plot_file.stat().st_size
            
            # Read just enough of the file to extract key info without loading it all
            with open(plot_file, 'r') as f:
                # Read the first chunk to get the structure and parameters
                first_chunk = f.read(10000)  # Read first 10KB
                
                # Extract parameters from the first chunk
                param_match = re.search(r'"name":\s*"([^"]+)"', first_chunk)
                if param_match:
                    # Try to find more parameters
                    params = re.findall(r'"name":\s*"([^"]+)"', first_chunk)
                    summary['parameters'] = list(set(params))  # Remove duplicates
                
                # Try to find the first timestamp
                date_match = re.search(r'"Date":\s*"([^"]+)"', first_chunk)
                if date_match:
                    summary['first_timestamp'] = date_match.group(1)
                
                # Go to the end of the file to find the last timestamp
                f.seek(max(0, os.path.getsize(plot_file) - 10000))  # Last 10KB
                last_chunk = f.read()
                
                # Find the last timestamp
                date_matches = re.findall(r'"Date":\s*"([^"]+)"', last_chunk)
                if date_matches:
                    summary['last_timestamp'] = date_matches[-1]
                
                # Estimate number of data points by counting commas and dividing by expected values per entry
                # This is a rough estimate
                comma_count = first_chunk.count(',')
                avg_commas_per_entry = 20  # Estimate based on typical JSON structure
                if comma_count > 0:
                    total_size = os.path.getsize(plot_file)
                    estimated_entries = (total_size * comma_count) / (avg_commas_per_entry * len(first_chunk))
                    summary['num_data_points'] = int(estimated_entries)
            
            return summary
        
        except Exception as e:
            logger.error(f"Error extracting plot summary from {plot_file}: {str(e)}")
            return summary
    
    def process_all_reports(self) -> pd.DataFrame:
        """
        Process all valid report folders and compile their metadata into the database.
        
        Returns:
            DataFrame containing metadata from all experiments
        """
        folders = self.scan_report_folders()
        metadata_list = []
        
        for folder in folders:
            try:
                logger.info(f"Processing folder: {folder.name}")
                
                # Check if this experiment already exists in the database
                existing_experiment = self.session.query(Experiment).filter_by(report_folder=folder.name).first()
                if existing_experiment:
                    logger.info(f"Experiment {folder.name} already exists in database, skipping")
                    metadata_list.append(existing_experiment.to_dict())
                    continue
                
                # Read metadata
                metadata = self.read_metadata_json(folder)
                if not metadata:
                    continue
                
                # Extract plot summary
                plot_summary_data = self.extract_plot_summary(folder)
                
                # Create experiment record
                experiment = Experiment(
                    report_folder=folder.name,
                    experiment_id=metadata.get('experiment_id'),
                    reactor_id=metadata.get('reactor_number'),
                    date=metadata.get('created_at') if metadata.get('created_at') else None,
                    time=metadata.get('folder_timestamp') if metadata.get('folder_timestamp') else None,
                    experiment_metadata={}
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
                step_analysis = self.analyze_experiment_steps_data(folder)
                total_data_points = 0
                
                # Create step records and their data points
                for step_num, step_data in step_analysis.get('steps', {}).items():
                    # Create step record
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
                            logger.error(f"Error processing data point in step {step_num}: {str(point_error)}")
                            continue
                
                # Add to session and commit
                self.session.add(experiment)
                self.session.commit()
                
                logger.info(f"Processed {total_data_points} data points across {len(experiment.steps)} steps")
                
                # Add to metadata list for DataFrame
                combined_data = experiment.to_dict()
                metadata_list.append(combined_data)
                
                logger.info(f"Successfully processed {folder.name}")
            
            except Exception as e:
                logger.error(f"Error processing folder {folder.name}: {str(e)}")
                self.session.rollback()
        
        # Convert to DataFrame for compatibility with existing code
        if metadata_list:
            metadata_db = pd.DataFrame(metadata_list)
            logger.info(f"Created metadata database with {len(metadata_list)} experiments")
        else:
            metadata_db = pd.DataFrame()
            logger.warning("No valid experiment data found")
        
        return metadata_db
    
    def get_experiment_details(self, experiment_id: str) -> Dict:
        """
        Get detailed information about a specific experiment.
        
        Args:
            experiment_id: ID or folder name of the experiment
            
        Returns:
            Dictionary containing experiment details
        """
        # Query the database for the experiment
        experiment = self.session.query(Experiment).filter(
            or_(
                Experiment.report_folder == experiment_id,
                Experiment.report_folder.like(f"{experiment_id}%")
            )
        ).first()
        
        if experiment:
            return experiment.to_dict()
        
        # If not found in database, try to process it directly
        for folder in self.reports_dir.iterdir():
            if folder.is_dir() and (folder.name == experiment_id or folder.name.startswith(experiment_id)):
                metadata = self.read_metadata_json(folder)
                plot_summary = self.extract_plot_summary(folder)
                return {**metadata, **plot_summary}
        
        logger.warning(f"Experiment {experiment_id} not found")
        return {}
    
    def list_experiments(self) -> pd.DataFrame:
        """
        Return a list of all experiments in the database.
        
        Returns:
            DataFrame containing basic information about all experiments
        """
        experiments = self.session.query(Experiment).all()
        
        if not experiments:
            logger.warning("No experiments found in database. Processing reports...")
            self.process_all_reports()
            experiments = self.session.query(Experiment).all()
        
        data = []
        for exp in experiments:
            data.append({
                'id': exp.id,
                'report_folder': exp.report_folder,
                'experiment_id': exp.experiment_id,
                'reactor_id': exp.reactor_id,
                'date': exp.date,
                'time': exp.time
            })
        
        return pd.DataFrame(data)
    
    def analyze_experiment_steps_data(self, experiment_folder: Path) -> Dict:
        """
        Analyze the steps in a specific experiment folder.
        
        Args:
            experiment_folder: Path to the experiment folder
            
        Returns:
            Dictionary containing information about the experiment steps
        """
        # Find all step folders
        step_info = {}
        step_folders = []
        
        for item in experiment_folder.iterdir():
            if item.is_dir() and item.name.startswith("step_"):
                step_num = item.name.replace("step_", "")
                if step_num.isdigit():
                    step_folders.append((int(step_num), item))
        
        # Sort steps by number
        step_folders.sort()
        
        # Analyze each step
        for step_num, step_folder in step_folders:
            step_data_file = step_folder / f"{step_folder.name}_data.json"
            step_plot_file = step_folder / f"{step_folder.name}_plot.json"
            
            step_info[step_num] = {
                'folder': step_folder.name,
                'has_data_json': step_data_file.exists(),
                'has_plot_json': step_plot_file.exists(),
                'data_file_size': step_data_file.stat().st_size if step_data_file.exists() else 0,
                'plot_file_size': step_plot_file.stat().st_size if step_plot_file.exists() else 0,
                'data_points': []  # Will store actual data points from the JSON file
            }
            
            # Try to extract step summary info (temperature, etc.)
            if step_data_file.exists():
                try:
                    with open(step_data_file, 'r') as f:
                        # Read the entire JSON file for full data extraction
                        step_data = json.load(f)
                        
                        # Store the number of data points
                        step_info[step_num]['data_points_count'] = len(step_data)
                        
                        # Extract key parameters from first data point for summary
                        if step_data and len(step_data) > 0:
                            first_data_point = step_data[0]
                            # Extract key parameters
                            for key in ['Stage', 'T Reactor (sliding TC)', 'Pressure reading', 
                                        'H2 Actual Flow', 'N2 Actual Flow']:
                                if key in first_data_point:
                                    step_info[step_num][key] = first_data_point[key]
                        
                        # Process all data points for detailed storage
                        for data_point in step_data:
                            # Create a standardized data point dictionary
                            point_data = {
                                'relative_time': data_point.get('Relative Time', None),
                                'timestamp': None,
                                't_reactor': None,
                                'pressure': None,
                                'h2_flow': None,
                                'n2_flow': None,
                                'nh3_concentration': None,
                                'outlet_flow': None,
                                'raw_data': json.dumps(data_point)
                            }
                            
                            # Extract date/timestamp
                            if 'Date' in data_point:
                                try:
                                    # Handle different date formats
                                    if isinstance(data_point['Date'], str):
                                        # ISO format with Z
                                        if 'Z' in data_point['Date']:
                                            point_data['timestamp'] = data_point['Date'].replace('Z', '+00:00')
                                        else:
                                            point_data['timestamp'] = data_point['Date']
                                except Exception as e:
                                    logger.debug(f"Could not parse date: {data_point.get('Date')}, error: {e}")
                            
                            # Extract common parameters with different possible field names
                            # Temperature
                            for temp_key in ['T Reactor (sliding TC)', 'T Heater UP', 'T Heater 1', 'T Heater 1_LV', 'T_Heater_1', 'Temperature']:
                                if temp_key in data_point and isinstance(data_point[temp_key], (int, float)) and not pd.isna(data_point[temp_key]):
                                    point_data['t_reactor'] = data_point[temp_key]
                                    break
                            
                            # Pressure
                            for pressure_key in ['Pressure reading', 'Pressure', 'Pressure_LV', 'Pressure (bar)', 'Pressure setpoint']:
                                if pressure_key in data_point and isinstance(data_point[pressure_key], (int, float)) and not pd.isna(data_point[pressure_key]):
                                    point_data['pressure'] = data_point[pressure_key]
                                    break
                            
                            # H2 Flow
                            for h2_key in ['H2 Actual Flow', 'H2_Flow', 'H2 Flow', 'H2 Flow (ml/min)']:
                                if h2_key in data_point and isinstance(data_point[h2_key], (int, float)) and not pd.isna(data_point[h2_key]):
                                    point_data['h2_flow'] = data_point[h2_key]
                                    break
                            
                            # N2 Flow
                            for n2_key in ['N2 Actual Flow', 'N2_Flow', 'N2 Flow', 'N2 Flow (ml/min)']:
                                if n2_key in data_point and isinstance(data_point[n2_key], (int, float)) and not pd.isna(data_point[n2_key]):
                                    point_data['n2_flow'] = data_point[n2_key]
                                    break
                            
                            # NH3 Concentration
                            for nh3_key in ['NH3_GC', 'NH3_clean_GC', 'NH3 (%)', 'NH3']:
                                if nh3_key in data_point and isinstance(data_point[nh3_key], (int, float)) and not pd.isna(data_point[nh3_key]):
                                    point_data['nh3_concentration'] = data_point[nh3_key]
                                    break
                            
                            # Outlet Flow
                            for outlet_key in ['Outlet meas.flowrate', 'Outlet g/h', 'Outlet mass flowrate', 'Outlet_mass', 'Mass flowrate (g/h)']:
                                if outlet_key in data_point and isinstance(data_point[outlet_key], (int, float)) and not pd.isna(data_point[outlet_key]):
                                    point_data['outlet_flow'] = data_point[outlet_key]
                                    break
                            
                            # Add the processed data point to the step info
                            step_info[step_num]['data_points'].append(point_data)
                            
                except Exception as e:
                    logger.error(f"Error extracting step data from {step_data_file}: {str(e)}")
        
        return {
            'experiment_id': experiment_folder.name,
            'report_folder': experiment_folder.name,
            'total_steps': len(step_folders),
            'steps': step_info
        }
    
    def analyze_experiment_steps(self, experiment_id: str) -> Dict:
        """
        Analyze the steps in a specific experiment.
        
        Args:
            experiment_id: ID or folder name of the experiment
            
        Returns:
            Dictionary containing information about the experiment steps
        """
        # Try to get from database first
        experiment = self.session.query(Experiment).filter(
            or_(
                Experiment.report_folder == experiment_id,
                Experiment.report_folder.like(f"{experiment_id}%")
            )
        ).first()
        
        if experiment and experiment.steps:
            steps_dict = {}
            for step in experiment.steps:
                steps_dict[step.step_number] = step.to_dict()
            
            return {
                'experiment_id': experiment_id,
                'report_folder': experiment.report_folder,
                'total_steps': len(experiment.steps),
                'steps': steps_dict
            }
        
        # If not in database or no steps, try to analyze directly
        for folder in self.reports_dir.iterdir():
            if folder.is_dir() and (folder.name == experiment_id or folder.name.startswith(experiment_id)):
                return self.analyze_experiment_steps_data(folder)
        
        logger.warning(f"Experiment {experiment_id} not found")
        return {}
    
    def export_database_to_csv(self, filename: str = "nh3_synth_database.csv") -> str:
        """
        Export the database to a CSV file for compatibility with external tools.
        
        Args:
            filename: Name of the output file
            
        Returns:
            Path to the saved file
        """
        experiments = self.session.query(Experiment).all()
        
        if not experiments:
            logger.warning("No data to export. Run process_all_reports() first.")
            return ""
        
        # Convert to DataFrame
        data = []
        for exp in experiments:
            exp_dict = exp.to_dict()
            
            # Flatten the plot summary data
            if 'plot_summary' in exp_dict and exp_dict['plot_summary']:
                for key, value in exp_dict['plot_summary'].items():
                    if key != 'parameters':  # Skip parameters as they are a list
                        exp_dict[f'plot_{key}'] = value
            
            # Remove nested objects
            if 'plot_summary' in exp_dict:
                del exp_dict['plot_summary']
            if 'steps' in exp_dict:
                del exp_dict['steps']
            
            data.append(exp_dict)
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        output_path = Path("database") / filename
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported database to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error exporting database: {str(e)}")
            return ""
    
    def export_database_to_json(self, filename: str = "nh3_synth_database.json") -> str:
        """
        Export the database to a JSON file for compatibility with external tools.
        
        Args:
            filename: Name of the output file
            
        Returns:
            Path to the saved file
        """
        experiments = self.session.query(Experiment).all()
        
        if not experiments:
            logger.warning("No data to export. Run process_all_reports() first.")
            return ""
        
        # Convert to dictionary
        data = {}
        for exp in experiments:
            data[exp.report_folder] = exp.to_dict()
        
        # Save to JSON
        output_path = Path("database") / filename
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Exported database to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error exporting database: {str(e)}")
            return ""
    
    def get_step_data_points(self, experiment_id: str, step_number: int) -> List[Dict]:
        """
        Get all data points for a specific step in an experiment.
        
        Args:
            experiment_id: ID or folder name of the experiment
            step_number: Number of the step
            
        Returns:
            List of dictionaries containing data points
        """
        # Find the experiment
        experiment = self.session.query(Experiment).filter(
            or_(
                Experiment.report_folder == experiment_id,
                Experiment.report_folder.like(f"{experiment_id}%")
            )
        ).first()
        
        if not experiment:
            logger.warning(f"Experiment {experiment_id} not found")
            return []
        
        # Find the step
        step = self.session.query(ExperimentStep).filter_by(
            experiment_id=experiment.id,
            step_number=step_number
        ).first()
        
        if not step:
            logger.warning(f"Step {step_number} not found for experiment {experiment_id}")
            return []
        
        # Get all data points for this step
        data_points = self.session.query(StepDataPoint).filter_by(step_id=step.id).all()
        
        # Convert to dictionaries
        return [dp.to_dict() for dp in data_points]
    
    def export_step_data_to_csv(self, experiment_id: str, step_number: int, filename: str = None) -> str:
        """
        Export data points for a specific step to a CSV file.
        
        Args:
            experiment_id: ID or folder name of the experiment
            step_number: Number of the step
            filename: Name of the output file (optional)
            
        Returns:
            Path to the saved file
        """
        # Get the data points
        data_points = self.get_step_data_points(experiment_id, step_number)
        
        if not data_points:
            logger.warning(f"No data points found for step {step_number} in experiment {experiment_id}")
            return ""
        
        # Convert to DataFrame
        df = pd.DataFrame(data_points)
        
        # Generate filename if not provided
        if not filename:
            filename = f"{experiment_id}_step_{step_number}_data_points.csv"
        
        # Save to CSV
        output_path = Path("database") / filename
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported step data to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error exporting step data: {str(e)}")
            return ""
    
    def get_all_experiment_data_points(self, experiment_id: str) -> Dict[int, List[Dict]]:
        """
        Get all data points for all steps in an experiment.
        
        Args:
            experiment_id: ID or folder name of the experiment
            
        Returns:
            Dictionary mapping step numbers to lists of data points
        """
        # Find the experiment
        experiment = self.session.query(Experiment).filter(
            or_(
                Experiment.report_folder == experiment_id,
                Experiment.report_folder.like(f"{experiment_id}%")
            )
        ).first()
        
        if not experiment:
            logger.warning(f"Experiment {experiment_id} not found")
            return {}
        
        # Get all steps
        steps = self.session.query(ExperimentStep).filter_by(experiment_id=experiment.id).all()
        
        # Get data points for each step
        step_data = {}
        for step in steps:
            data_points = self.session.query(StepDataPoint).filter_by(step_id=step.id).all()
            step_data[step.step_number] = [dp.to_dict() for dp in data_points]
        
        return step_data
    
    def export_experiment_data_to_csv(self, experiment_id: str, filename: str = None) -> str:
        """
        Export all data points for an experiment to a CSV file.
        
        Args:
            experiment_id: ID or folder name of the experiment
            filename: Name of the output file (optional)
            
        Returns:
            Path to the saved file
        """
        # Get all data points
        step_data = self.get_all_experiment_data_points(experiment_id)
        
        if not step_data:
            logger.warning(f"No data points found for experiment {experiment_id}")
            return ""
        
        # Combine all data points into a single DataFrame
        all_data = []
        for step_num, data_points in step_data.items():
            for dp in data_points:
                dp['step_number'] = step_num
                all_data.append(dp)
        
        df = pd.DataFrame(all_data)
        
        # Generate filename if not provided
        if not filename:
            filename = f"{experiment_id}_all_data_points.csv"
        
        # Save to CSV
        output_path = Path("database") / filename
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported experiment data to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error exporting experiment data: {str(e)}")
            return ""

    def get_experiment_by_id(self, experiment_id):
        """Get an experiment by its ID."""
        session = self.session
        try:
            exp = session.query(Experiment).filter_by(id=experiment_id).first()
            if not exp:
                return None
            
            result = {
                'id': exp.id,
                'report_folder': exp.report_folder,
                'experiment_id': exp.experiment_id,
                'reactor_id': exp.reactor_id,
                'date': exp.date,
                'time': exp.time,
                'metadata': exp.experiment_metadata,
                'steps': []
            }
            
            for step in exp.steps:
                result['steps'].append({
                    'id': step.id,
                    'step_number': step.step_number,
                    'description': step.description,
                    'data_point_count': step.data_point_count
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting experiment by ID: {str(e)}")
            return None
        
        finally:
            session.close()
    
    def get_step_data_points(self, step_id, limit=1000, offset=0):
        """Get data points for a specific step."""
        session = self.session
        try:
            # Get the total count
            total_count = session.query(func.count(StepDataPoint.id)).filter_by(step_id=step_id).scalar()
            
            # Get the data points with pagination
            data_points = session.query(StepDataPoint).filter_by(step_id=step_id).order_by(StepDataPoint.relative_time).offset(offset).limit(limit).all()
            
            result = {
                'total_count': total_count,
                'offset': offset,
                'limit': limit,
                'data_points': []
            }
            
            for point in data_points:
                result['data_points'].append({
                    'id': point.id,
                    'date': point.date,
                    'stage': point.stage,
                    'relative_time': point.relative_time,
                    'h2_node': point.h2_node,
                    'h2_set_point': point.h2_set_point,
                    'h2_actual_flow': point.h2_actual_flow,
                    'n2_node': point.n2_node,
                    'n2_set_point': point.n2_set_point,
                    'n2_actual_flow': point.n2_actual_flow,
                    't_reactor': point.t_reactor,
                    't_heater_1': point.t_heater_1,
                    'sp_heater_1': point.sp_heater_1,
                    'pressure_setpoint': point.pressure_setpoint,
                    'n2_poisoning_set_point': point.n2_poisoning_set_point,
                    'n2_poisoning_actual_flow': point.n2_poisoning_actual_flow,
                    'h2_uncontrolled_flow': point.h2_uncontrolled_flow,
                    'n2_uncontrolled_flow': point.n2_uncontrolled_flow,
                    'inlet_calc_flowrate': point.inlet_calc_flowrate,
                    'outlet_meas_flowrate': point.outlet_meas_flowrate,
                    'pressure_reading': point.pressure_reading,
                    't1': point.t1,
                    't2': point.t2,
                    't3': point.t3,
                    't4': point.t4,
                    't5': point.t5,
                    't6': point.t6,
                    't7': point.t7,
                    't8': point.t8,
                    't9': point.t9,
                    't10': point.t10,
                    't11': point.t11,
                    't12': point.t12,
                    't13': point.t13,
                    't14': point.t14,
                    'h2_clean_gc': point.h2_clean_gc,
                    'n2_clean_gc': point.n2_clean_gc,
                    'nh3_clean_gc': point.nh3_clean_gc
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting step data points: {str(e)}")
            return {'total_count': 0, 'offset': offset, 'limit': limit, 'data_points': []}
        
        finally:
            session.close()
    
    def export_step_data_to_csv(self, step_id, output_path):
        """Export data points for a specific step to a CSV file."""
        session = self.session
        try:
            data_points = session.query(StepDataPoint).filter_by(step_id=step_id).order_by(StepDataPoint.relative_time).all()
            
            if not data_points:
                logger.warning(f"No data points found for step ID {step_id}")
                return False
            
            data = []
            for point in data_points:
                # Convert each data point to a dictionary
                data_dict = {
                    'date': point.date,
                    'stage': point.stage,
                    'relative_time': point.relative_time,
                    'h2_node': point.h2_node,
                    'h2_set_point': point.h2_set_point,
                    'h2_actual_flow': point.h2_actual_flow,
                    'n2_node': point.n2_node,
                    'n2_set_point': point.n2_set_point,
                    'n2_actual_flow': point.n2_actual_flow,
                    't_reactor': point.t_reactor,
                    't_heater_1': point.t_heater_1,
                    'sp_heater_1': point.sp_heater_1,
                    'pressure_setpoint': point.pressure_setpoint,
                    'n2_poisoning_set_point': point.n2_poisoning_set_point,
                    'n2_poisoning_actual_flow': point.n2_poisoning_actual_flow,
                    'h2_uncontrolled_flow': point.h2_uncontrolled_flow,
                    'n2_uncontrolled_flow': point.n2_uncontrolled_flow,
                    'inlet_calc_flowrate': point.inlet_calc_flowrate,
                    'outlet_meas_flowrate': point.outlet_meas_flowrate,
                    'pressure_reading': point.pressure_reading,
                    't1': point.t1,
                    't2': point.t2,
                    't3': point.t3,
                    't4': point.t4,
                    't5': point.t5,
                    't6': point.t6,
                    't7': point.t7,
                    't8': point.t8,
                    't9': point.t9,
                    't10': point.t10,
                    't11': point.t11,
                    't12': point.t12,
                    't13': point.t13,
                    't14': point.t14,
                    'h2_clean_gc': point.h2_clean_gc,
                    'n2_clean_gc': point.n2_clean_gc,
                    'nh3_clean_gc': point.nh3_clean_gc
                }
                data.append(data_dict)
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(data)} data points to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting step data to CSV: {str(e)}")
            return False
        
        finally:
            session.close()
    
    def export_step_data_to_json(self, step_id, output_path):
        """Export data points for a specific step to a JSON file."""
        session = self.session
        try:
            data_points = session.query(StepDataPoint).filter_by(step_id=step_id).order_by(StepDataPoint.relative_time).all()
            
            if not data_points:
                logger.warning(f"No data points found for step ID {step_id}")
                return False
            
            data = []
            for point in data_points:
                # Convert each data point to a dictionary
                data_dict = {
                    'date': point.date,
                    'stage': point.stage,
                    'relative_time': point.relative_time,
                    'h2_node': point.h2_node,
                    'h2_set_point': point.h2_set_point,
                    'h2_actual_flow': point.h2_actual_flow,
                    'n2_node': point.n2_node,
                    'n2_set_point': point.n2_set_point,
                    'n2_actual_flow': point.n2_actual_flow,
                    't_reactor': point.t_reactor,
                    't_heater_1': point.t_heater_1,
                    'sp_heater_1': point.sp_heater_1,
                    'pressure_setpoint': point.pressure_setpoint,
                    'n2_poisoning_set_point': point.n2_poisoning_set_point,
                    'n2_poisoning_actual_flow': point.n2_poisoning_actual_flow,
                    'h2_uncontrolled_flow': point.h2_uncontrolled_flow,
                    'n2_uncontrolled_flow': point.n2_uncontrolled_flow,
                    'inlet_calc_flowrate': point.inlet_calc_flowrate,
                    'outlet_meas_flowrate': point.outlet_meas_flowrate,
                    'pressure_reading': point.pressure_reading,
                    't1': point.t1,
                    't2': point.t2,
                    't3': point.t3,
                    't4': point.t4,
                    't5': point.t5,
                    't6': point.t6,
                    't7': point.t7,
                    't8': point.t8,
                    't9': point.t9,
                    't10': point.t10,
                    't11': point.t11,
                    't12': point.t12,
                    't13': point.t13,
                    't14': point.t14,
                    'h2_clean_gc': point.h2_clean_gc,
                    'n2_clean_gc': point.n2_clean_gc,
                    'nh3_clean_gc': point.nh3_clean_gc
                }
                data.append(data_dict)
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported {len(data)} data points to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting step data to JSON: {str(e)}")
            return False
        
        finally:
            session.close()

    def _process_step_data_file(self, data_file_path, step_id):
        """Process a step data file and return a list of data point objects."""
        try:
            with open(data_file_path, 'r') as f:
                data = json.load(f)
            
            data_points = []
            
            for point in data:
                data_point = StepDataPoint(
                    step_id=step_id,
                    date=point.get("Date"),
                    stage=point.get("Stage"),
                    relative_time=point.get("RelativeTime"),
                    h2_node=point.get("H2 Node"),
                    h2_set_point=point.get("H2 Set-Point"),
                    h2_actual_flow=point.get("H2 Actual Flow"),
                    n2_node=point.get("N2 Node"),
                    n2_set_point=point.get("N2 Set-Point"),
                    n2_actual_flow=point.get("N2 Actual Flow"),
                    t_reactor=point.get("T Reactor (sliding TC)"),
                    t_heater_1=point.get("T Heater 1"),
                    sp_heater_1=point.get("SP Heater 1"),
                    pressure_setpoint=point.get("Pressure setpoint"),
                    n2_poisoning_set_point=point.get("N2 poisoning set-point"),
                    n2_poisoning_actual_flow=point.get("N2 poisoning Actual flow"),
                    h2_uncontrolled_flow=point.get("H2 Uncontrolled flow"),
                    n2_uncontrolled_flow=point.get("N2 Uncontrolled flow"),
                    inlet_calc_flowrate=point.get("Inlet calc. flowrate"),
                    outlet_meas_flowrate=point.get("Outlet meas.flowrate"),
                    pressure_reading=point.get("Pressure reading"),
                    t1=point.get("T1"),
                    t2=point.get("T2"),
                    t3=point.get("T3"),
                    t4=point.get("T4"),
                    t5=point.get("T5"),
                    t6=point.get("T6"),
                    t7=point.get("T7"),
                    t8=point.get("T8"),
                    t9=point.get("T9"),
                    t10=point.get("T10"),
                    t11=point.get("T11"),
                    t12=point.get("T12"),
                    t13=point.get("T13"),
                    t14=point.get("T14"),
                    h2_clean_gc=point.get("H2_clean_GC"),
                    n2_clean_gc=point.get("N2_clean_GC"),
                    nh3_clean_gc=point.get("NH3_clean_GC"),
                    raw_data=point
                )
                data_points.append(data_point)
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error processing data file {data_file_path}: {str(e)}")
            return []

def main():
    """Main function to demonstrate the usage of the NH3SynthDatabaseProcessor."""
    processor = NH3SynthDatabaseProcessor()
    
    # Process all reports
    print("\n" + "="*80)
    print("NH3 SYNTHESIS DATABASE PROCESSOR".center(80))
    print("="*80)
    
    print("\n[!] Starting to process NH3 synthesis experiment reports...")
    metadata_db = processor.process_all_reports()
    
    if not metadata_db.empty:
        # Export database files for compatibility
        csv_path = processor.export_database_to_csv()
        json_path = processor.export_database_to_json()
        
        # Print summary
        print("\n" + "-"*80)
        print("PROCESSING COMPLETE".center(80))
        print("-"*80)
        print(f"\n[!] Found {len(metadata_db)} experiment reports")
        print(f"[!] Database created at: {processor.db_path}")
        print(f"[!] Exported database files:")
        if csv_path:
            print(f"    - CSV: {csv_path}")
        if json_path:
            print(f"    - JSON: {json_path}")
        
        # Display experiment list
        print("\n" + "-"*80)
        print("EXPERIMENT LIST".center(80))
        print("-"*80)
        experiments = processor.list_experiments()
        print(experiments.to_string(index=False))
        
        # Provide example usage
        print("\n" + "-"*80)
        print("EXAMPLE USAGE".center(80))
        print("-"*80)
        print("""
To use this module in your Python code:

    from nh3_synth_database import NH3SynthDatabaseProcessor
    
    # Create processor instance
    processor = NH3SynthDatabaseProcessor()
    
    # Process all reports
    metadata_db = processor.process_all_reports()
    
    # Get details for a specific experiment
    experiment_details = processor.get_experiment_details("001_2194_R101_20250609_155129")
    
    # Analyze steps in an experiment
    step_analysis = processor.analyze_experiment_steps("001_2194_R101_20250609_155129")
""")
    else:
        print("\n[!] No valid experiment data found. Please check the reports directory.")

if __name__ == "__main__":
    main() 