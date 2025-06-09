#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NH3 Synthesis Database Query Examples

This script demonstrates various ways to query and analyze the NH3 synthesis database.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from Database.nh3_synth_database import NH3SynthDatabaseProcessor
from Database.database_models import init_db, get_session, Experiment, PlotSummary, ExperimentStep, StepDataPoint
from sqlalchemy import func, desc

def print_section(title):
    """Print a section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")

def main():
    """Run example queries against the NH3 synthesis database."""
    print_section("NH3 SYNTHESIS DATABASE QUERY EXAMPLES")
    
    # Initialize the processor and database connection
    processor = NH3SynthDatabaseProcessor()
    session = processor.session
    
    # Check if database has data, process reports if needed
    experiment_count = session.query(func.count(Experiment.id)).scalar()
    if experiment_count == 0:
        print("[!] Database is empty. Processing experiment reports...")
        processor.process_all_reports()
    else:
        print(f"[!] Found {experiment_count} experiments in database.")
    
    # Example 1: List all experiments
    print_section("EXAMPLE 1: List all experiments")
    experiments = processor.list_experiments()
    print(experiments.to_string(index=False))
    
    # Example 2: Get details for a specific experiment
    print_section("EXAMPLE 2: Get details for a specific experiment")
    # Get the first experiment from the database
    first_experiment = session.query(Experiment).first()
    if first_experiment:
        experiment_id = first_experiment.folder_name
        print(f"[!] Getting details for experiment: {experiment_id}")
        experiment_details = processor.get_experiment_details(experiment_id)
        
        # Print formatted details
        print("\nEXPERIMENT DETAILS:")
        print(f"  Experiment Number: {experiment_details.get('experiment_number', 'N/A')}")
        print(f"  Catalyst Batch: {experiment_details.get('catalyst_batch', 'N/A')}")
        print(f"  Reactor Number: {experiment_details.get('reactor_number', 'N/A')}")
        print(f"  Catalyst Weight: {experiment_details.get('catalyst_weight', 'N/A')} g")
        print(f"  Catalyst Volume: {experiment_details.get('catalyst_volume', 'N/A')} ml")
        print(f"  Catalyst State: {experiment_details.get('catalyst_state', 'N/A')}")
        print(f"  Catalyst Notes: {experiment_details.get('catalyst_notes', 'N/A')}")
        print(f"  Created At: {experiment_details.get('created_at', 'N/A')}")
        
        # Plot summary information
        if 'plot_summary' in experiment_details and experiment_details['plot_summary']:
            print("\nDATA SUMMARY:")
            plot_summary = experiment_details['plot_summary']
            print(f"  Plot File Size: {plot_summary.get('plot_file_size', 0) / (1024*1024):.2f} MB")
            print(f"  Estimated Data Points: {plot_summary.get('num_data_points', 'N/A')}")
            print(f"  First Timestamp: {plot_summary.get('first_timestamp', 'N/A')}")
            print(f"  Last Timestamp: {plot_summary.get('last_timestamp', 'N/A')}")
            
            if 'parameters' in plot_summary and plot_summary['parameters']:
                print("\nMEASURED PARAMETERS:")
                for param in sorted(plot_summary['parameters']):
                    print(f"  - {param}")
    else:
        print("[!] No experiments found in the database.")
    
    # Example 3: Analyze steps for a specific experiment
    print_section("EXAMPLE 3: Analyze experiment steps")
    if first_experiment:
        experiment_id = first_experiment.folder_name
        print(f"[!] Analyzing steps for experiment: {experiment_id}")
        step_analysis = processor.analyze_experiment_steps(experiment_id)
        
        if step_analysis and 'steps' in step_analysis:
            print(f"\nTotal Steps: {step_analysis.get('total_steps', 0)}")
            
            # Create a summary table of steps
            step_data = []
            for step_num, step_info in step_analysis['steps'].items():
                step_data.append({
                    'Step': step_num,
                    'Temperature (°C)': step_info.get('T Reactor (sliding TC)', 'N/A'),
                    'Pressure (bar)': step_info.get('Pressure reading', 'N/A'),
                    'H2 Flow': step_info.get('H2 Actual Flow', 'N/A'),
                    'N2 Flow': step_info.get('N2 Actual Flow', 'N/A'),
                    'Data Size (KB)': step_info.get('data_file_size', 0) / 1024,
                    'Data Points': step_info.get('data_points_count', 'N/A')
                })
            
            step_df = pd.DataFrame(step_data)
            print("\nSTEP SUMMARY:")
            print(step_df.to_string(index=False))
    else:
        print("[!] No experiments found in the database.")
    
    # Example 4: Compare catalyst batches using SQLAlchemy queries
    print_section("EXAMPLE 4: Compare catalyst batches")
    
    # Get list of distinct catalyst batches
    catalyst_batches = session.query(Experiment.catalyst_batch,
                                     func.count(Experiment.id).label('count')) \
                             .group_by(Experiment.catalyst_batch) \
                             .all()
    
    if catalyst_batches and len(catalyst_batches) > 0:
        print(f"[!] Comparing {len(catalyst_batches)} different catalyst batches:")
        
        for batch, count in catalyst_batches:
            print(f"  - Batch {batch}: {count} experiments")
        
        # For each batch, calculate statistics
        batch_comparison = []
        for batch, _ in catalyst_batches:
            batch_stats = session.query(
                func.avg(Experiment.catalyst_weight).label('avg_weight'),
                func.min(Experiment.catalyst_weight).label('min_weight'),
                func.max(Experiment.catalyst_weight).label('max_weight'),
                func.avg(Experiment.catalyst_volume).label('avg_volume'),
                func.min(Experiment.catalyst_volume).label('min_volume'),
                func.max(Experiment.catalyst_volume).label('max_volume'),
                func.avg(Experiment.catalyst_bed_length).label('avg_bed_length'),
                func.min(Experiment.catalyst_bed_length).label('min_bed_length'),
                func.max(Experiment.catalyst_bed_length).label('max_bed_length')
            ).filter(Experiment.catalyst_batch == batch).first()
            
            batch_comparison.append({
                'Batch': batch,
                'Weight (avg)': batch_stats.avg_weight,
                'Weight (min)': batch_stats.min_weight,
                'Weight (max)': batch_stats.max_weight,
                'Volume (avg)': batch_stats.avg_volume,
                'Volume (min)': batch_stats.min_volume,
                'Volume (max)': batch_stats.max_volume,
                'Bed Length (avg)': batch_stats.avg_bed_length,
                'Bed Length (min)': batch_stats.min_bed_length,
                'Bed Length (max)': batch_stats.max_bed_length
            })
        
        batch_df = pd.DataFrame(batch_comparison)
        print("\nCATALYST BATCH COMPARISON:")
        print(batch_df.to_string(index=False))
        
        # Example 5: SQL-specific query - newest experiments first
        print_section("EXAMPLE 5: Latest experiments (SQL-specific query)")
        latest_experiments = session.query(Experiment) \
                                   .order_by(desc(Experiment.created_at)) \
                                   .limit(5) \
                                   .all()
        
        print("[!] Most recent experiments:")
        for exp in latest_experiments:
            created_at = exp.created_at.strftime('%Y-%m-%d %H:%M:%S') if exp.created_at else 'N/A'
            print(f"  - {exp.folder_name} (Batch {exp.catalyst_batch}, Reactor {exp.reactor_number}), created: {created_at}")
    else:
        print("[!] No catalyst batches found to compare.")
    
    # Example 6: Analyze data points for a specific step
    print_section("EXAMPLE 6: Analyze data points for a specific step")
    if first_experiment:
        experiment_id = first_experiment.folder_name
        
        # Find a step with data points
        steps_with_data = session.query(ExperimentStep) \
                               .filter_by(experiment_id=first_experiment.id) \
                               .filter(ExperimentStep.data_points_count > 0) \
                               .order_by(ExperimentStep.step_number) \
                               .first()
        
        if steps_with_data:
            step_number = steps_with_data.step_number
            print(f"[!] Analyzing data points for experiment: {experiment_id}, step: {step_number}")
            
            # Get data points
            data_points = processor.get_step_data_points(experiment_id, step_number)
            
            if data_points:
                print(f"\nFound {len(data_points)} data points for step {step_number}")
                
                # Print first few data points
                num_to_show = min(5, len(data_points))
                print(f"\nFirst {num_to_show} data points:")
                for i, dp in enumerate(data_points[:num_to_show]):
                    print(f"\nData Point {i+1}:")
                    for key, value in dp.items():
                        if key != 'raw_data':  # Skip raw data as it's too verbose
                            print(f"  {key}: {value}")
                
                # Export data to CSV
                csv_path = processor.export_step_data_to_csv(experiment_id, step_number)
                if csv_path:
                    print(f"\nExported data points to CSV: {csv_path}")
                
                # Plot data if we have enough points
                if len(data_points) > 1:
                    try:
                        # Create a simple plot of temperature vs. time
                        plt.figure(figsize=(10, 6))
                        
                        # Extract data for plotting
                        times = [dp.get('relative_time', i) for i, dp in enumerate(data_points) if dp.get('t_reactor') is not None]
                        temps = [dp.get('t_reactor') for dp in data_points if dp.get('t_reactor') is not None]
                        
                        if times and temps:
                            plt.plot(times, temps, 'r-', linewidth=2)
                            plt.title(f'Temperature vs. Time for Step {step_number}')
                            plt.xlabel('Relative Time')
                            plt.ylabel('Temperature (°C)')
                            plt.grid(True)
                            
                            # Save the plot
                            output_dir = Path("database/plots")
                            output_dir.mkdir(parents=True, exist_ok=True)
                            output_path = output_dir / f"{experiment_id}_step_{step_number}_temp_plot.png"
                            plt.savefig(output_path)
                            plt.close()
                            
                            print(f"\nCreated temperature plot: {output_path}")
                    except Exception as e:
                        print(f"Error creating plot: {str(e)}")
            else:
                print(f"No data points found for step {step_number}")
        else:
            print("No steps with data points found for this experiment")
    else:
        print("[!] No experiments found in the database.")
    
    # Example 7: Export all data points for an experiment
    print_section("EXAMPLE 7: Export all experiment data")
    if first_experiment:
        experiment_id = first_experiment.folder_name
        print(f"[!] Exporting all data for experiment: {experiment_id}")
        
        # Get total number of data points
        total_data_points = session.query(func.count(StepDataPoint.id)) \
                                 .join(ExperimentStep) \
                                 .filter(ExperimentStep.experiment_id == first_experiment.id) \
                                 .scalar()
        
        if total_data_points > 0:
            print(f"\nFound {total_data_points} data points across all steps")
            
            # Export to CSV
            csv_path = processor.export_experiment_data_to_csv(experiment_id)
            if csv_path:
                print(f"\nExported all data points to CSV: {csv_path}")
        else:
            print("No data points found for this experiment")
    else:
        print("[!] No experiments found in the database.")
    
    print_section("QUERIES COMPLETED")

if __name__ == "__main__":
    main() 