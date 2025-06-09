#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NH3 Synthesis Database Models

This module defines the SQLAlchemy ORM models for the NH3 synthesis database.
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, Text, ForeignKey, create_engine, JSON, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import json
import os
from pathlib import Path
import time
import logging

Base = declarative_base()

class Experiment(Base):
    """Model representing an NH3 synthesis experiment."""
    __tablename__ = 'experiments'
    
    id = Column(Integer, primary_key=True)
    report_folder = Column(String, nullable=False, unique=True)
    experiment_id = Column(String, nullable=False)
    reactor_id = Column(String, nullable=False)
    date = Column(String, nullable=False)
    time = Column(String, nullable=False)
    experiment_metadata = Column(JSON, nullable=True)
    
    # Relationships
    steps = relationship("ExperimentStep", back_populates="experiment", cascade="all, delete-orphan")
    plot_summary = relationship("PlotSummary", uselist=False, back_populates="experiment", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Experiment(report_folder='{self.report_folder}', experiment_id='{self.experiment_id}', reactor_id='{self.reactor_id}')>"
        
    def to_dict(self):
        """Convert the experiment model to a dictionary."""
        result = {
            'id': self.id,
            'report_folder': self.report_folder,
            'experiment_id': self.experiment_id,
            'reactor_id': self.reactor_id,
            'date': self.date,
            'time': self.time,
            'experiment_metadata': self.experiment_metadata,
            'steps': []
        }
        
        # Add steps data
        if self.steps:
            result['steps'] = [step.to_dict() for step in self.steps]
            
        # Add plot summary data
        if self.plot_summary:
            result['plot_summary'] = {
                'id': self.plot_summary.id,
                'plot_file_size': self.plot_summary.plot_file_size,
                'num_data_points': self.plot_summary.num_data_points,
                'parameters': self.plot_summary.parameters,
                'first_timestamp': self.plot_summary.first_timestamp,
                'last_timestamp': self.plot_summary.last_timestamp
            }
            
        return result


class ExperimentStep(Base):
    """Model representing a step in an NH3 synthesis experiment."""
    __tablename__ = 'experiment_steps'
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.id'), nullable=False)
    step_number = Column(Integer, nullable=False)
    folder = Column(String, nullable=True)
    has_data_json = Column(Integer, default=0)
    has_plot_json = Column(Integer, default=0)
    data_file_size = Column(Integer, default=0)
    plot_file_size = Column(Integer, default=0)
    temperature = Column(Float, nullable=True)
    pressure = Column(Float, nullable=True)
    h2_flow = Column(Float, nullable=True)
    n2_flow = Column(Float, nullable=True)
    stage = Column(String, nullable=True)
    data_points_count = Column(Integer, default=0)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="steps")
    data_points = relationship("StepDataPoint", back_populates="step", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ExperimentStep(experiment_id={self.experiment_id}, step_number={self.step_number})>"
        
    def to_dict(self):
        """Convert the step model to a dictionary."""
        return {
            'id': self.id,
            'experiment_id': self.experiment_id,
            'step_number': self.step_number,
            'folder': self.folder,
            'has_data_json': self.has_data_json,
            'has_plot_json': self.has_plot_json,
            'data_file_size': self.data_file_size,
            'plot_file_size': self.plot_file_size,
            'T Reactor (sliding TC)': self.temperature,
            'Pressure reading': self.pressure,
            'H2 Actual Flow': self.h2_flow,
            'N2 Actual Flow': self.n2_flow,
            'Stage': self.stage,
            'data_points_count': self.data_points_count
        }


class StepDataPoint(Base):
    """Model representing a single data point in a step."""
    __tablename__ = 'step_data_points'
    
    id = Column(Integer, primary_key=True)
    step_id = Column(Integer, ForeignKey('experiment_steps.id'), nullable=False)
    relative_time = Column(Float, nullable=True)
    timestamp = Column(DateTime, nullable=True)
    t_reactor = Column(Float, nullable=True)
    pressure = Column(Float, nullable=True)
    h2_flow = Column(Float, nullable=True)
    n2_flow = Column(Float, nullable=True)
    nh3_concentration = Column(Float, nullable=True)
    outlet_flow = Column(Float, nullable=True)
    raw_data = Column(JSON, nullable=True)
    
    # Relationships
    step = relationship("ExperimentStep", back_populates="data_points")
    
    def __repr__(self):
        return f"<StepDataPoint(step_id={self.step_id}, relative_time={self.relative_time})>"


class PlotSummary(Base):
    """Model representing summary information about the experiment's plot data."""
    __tablename__ = 'plot_summaries'
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.id'), nullable=False)
    plot_file_size = Column(Integer, default=0)
    num_data_points = Column(Integer, default=0)
    parameters = Column(String, nullable=True)  # JSON string of parameters
    first_timestamp = Column(String, nullable=True)
    last_timestamp = Column(String, nullable=True)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="plot_summary")
    
    def __repr__(self):
        return f"<PlotSummary(experiment_id={self.experiment_id}, num_data_points={self.num_data_points})>"


def init_db(db_path='database/nh3_synth.db', recreate=False):
    """Initialize the database and create all tables."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Create engine and tables
    engine = create_engine(f'sqlite:///{db_path}')
    
    if recreate:
        # Try to drop all tables with retry for locked database
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Drop all tables and recreate them
                Base.metadata.drop_all(engine)
                break  # If successful, exit the loop
            except Exception as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    logging.warning(f"Database is locked. Retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logging.error(f"Failed to drop tables: {str(e)}")
                    # Continue without dropping - we'll try to work with existing schema
    
    try:
        # Create tables if they don't exist
        Base.metadata.create_all(engine)
    except Exception as e:
        logging.error(f"Error creating database tables: {str(e)}")
    
    return engine


def get_session(engine):
    """Create a session for database operations."""
    Session = sessionmaker(bind=engine)
    return Session() 