import json
import logging
from enum import Enum

import duckdb
import pandas as pd


class TrainingStatus(Enum):
    """Enum for training request statuses."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingRequestDataLoader:
    """
    Data loader for managing training requests in DuckDB.
    Handles persistence of training requests, their configurations, and status tracking.
    """

    def __init__(self, db_path="md:b3"):
        """
        Initialize the DuckDB connection for training request data.
        
        Args:
            db_path (str): Path to DuckDB database file or ':memory:' for in-memory DB.
                          Defaults to MotherDuck connection.
        """
        self.db_path = db_path
        self.conn = duckdb.connect(database=db_path)
        self.table_name = "b3_training_requests"
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self):
        """
        Create the training requests table if it doesn't exist.
        Stores training request metadata, configurations, and status information.
        """
        # Create a sequence for log_id if it does not exist
        create_sequence_sql = f"""
        CREATE SEQUENCE IF NOT EXISTS {self.table_name}_log_id_seq;
        """
        self.conn.execute(create_sequence_sql)

        # Create the table using the sequence for log_id
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            log_id INTEGER PRIMARY KEY DEFAULT nextval('{self.table_name}_log_id_seq'),
            endpoint VARCHAR NOT NULL,
            request_payload JSON,
            response_payload JSON,
            status VARCHAR NOT NULL,
            error_message TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        logging.info(f"Creating table {self.table_name} if not exists...")
        self.conn.execute(create_table_sql)

    def _base_select(self) -> str:
        """
        Base SELECT clause used across fetch methods to avoid duplication.
        """
        return f"SELECT * FROM {self.table_name}"

    def log_api_activity(self, endpoint: str, request_data: dict, response_data: dict, status: str,
                         error_message: str = None) -> bool:
        """
        Log an API request and its response.
        Args:
            endpoint (str): The API endpoint name or path
            request_data (dict): The request payload
            response_data (dict): The response payload
            status (str): 'success' or 'error'
            error_message (str, optional): Error message if any
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            insert_sql = f"""
            INSERT INTO {self.table_name} (endpoint, request_payload, response_payload, status, error_message)
            VALUES (?, ?, ?, ?, ?)
            """
            self.conn.execute(insert_sql, [
                endpoint,
                json.dumps(request_data) if request_data else None,
                json.dumps(response_data) if response_data else None,
                status,
                error_message
            ])
            logging.info(f"Logged API activity for endpoint {endpoint}")
            return True
        except Exception as e:
            logging.error(f"Error logging API activity for endpoint {endpoint}: {str(e)}")
            return False

    def fetch_all(self) -> pd.DataFrame:
        """
        Fetch all training request records.
        
        Returns:
            pd.DataFrame: All training request data
        """
        logging.info(f"Fetching all data from {self.table_name}...")
        return self.conn.execute(self._base_select()).fetchdf()

    def fetch_by_request_id(self, request_id: str) -> pd.DataFrame:
        """
        Fetch specific training request by ID.
        
        Args:
            request_id (str): Unique training request identifier
            
        Returns:
            pd.DataFrame: Training request data
        """
        logging.info(f"Fetching training request {request_id}...")
        query = f"{self._base_select()} WHERE request_id = ?"
        return self.conn.execute(query, (request_id,)).fetchdf()

    def fetch_by_model(self, model_name: str) -> pd.DataFrame:
        """
        Fetch all training requests for a specific pipeline.
        
        Args:
            model_name (str): Name of the pipeline
            
        Returns:
            pd.DataFrame: Training request data for the pipeline
        """
        logging.info(f"Fetching training requests for pipeline {model_name}...")
        query = f"{self._base_select()} WHERE model_name = ?"
        return self.conn.execute(query, (model_name,)).fetchdf()

    def fetch_by_status(self, status: TrainingStatus) -> pd.DataFrame:
        """
        Fetch training requests by status.
        
        Args:
            status (TrainingStatus): Status to filter by
            
        Returns:
            pd.DataFrame: Training request data with specified status
        """
        logging.info(f"Fetching training requests with status {status.value}...")
        query = f"{self._base_select()} WHERE status = ?"
        return self.conn.execute(query, (status.value,)).fetchdf()

    def fetch_pending_requests(self) -> pd.DataFrame:
        """
        Fetch all pending training requests.
        
        Returns:
            pd.DataFrame: Pending training request data
        """
        return self.fetch_by_status(TrainingStatus.PENDING)

    def fetch_in_progress_requests(self) -> pd.DataFrame:
        """
        Fetch all in-progress training requests.
        
        Returns:
            pd.DataFrame: In-progress training request data
        """
        return self.fetch_by_status(TrainingStatus.IN_PROGRESS)

    def fetch_by_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch training requests within a date range.
        
        Args:
            start_date (str): Start date (inclusive), format 'YYYY-MM-DD'
            end_date (str): End date (inclusive), format 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: Training request data within date range
        """
        logging.info(f"Fetching training requests from {start_date} to {end_date}...")
        query = f"{self._base_select()} WHERE DATE(request_timestamp) >= ? AND DATE(request_timestamp) <= ?"
        return self.conn.execute(query, (start_date, end_date)).fetchdf()

    def fetch_latest_by_model(self, model_name: str) -> pd.DataFrame:
        """
        Fetch the latest training request for a specific pipeline.
        
        Args:
            model_name (str): Name of the pipeline
            
        Returns:
            pd.DataFrame: Latest training request data for the pipeline
        """
        logging.info(f"Fetching latest training request for pipeline {model_name}...")
        query = (
            f"{self._base_select()} WHERE model_name = ? "
            f"ORDER BY request_timestamp DESC LIMIT 1"
        )
        return self.conn.execute(query, (model_name,)).fetchdf()

    def get_training_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of all training requests.
        
        Returns:
            pd.DataFrame: Summary statistics
        """
        logging.info("Generating training request summary...")
        query = f"""
        SELECT 
            model_name,
            COUNT(*) as total_requests,
            COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_count,
            COUNT(CASE WHEN status = 'in_progress' THEN 1 END) as in_progress_count,
            COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_count,
            COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_count,
            COUNT(CASE WHEN status = 'cancelled' THEN 1 END) as cancelled_count,
            MAX(request_timestamp) as latest_request,
            AVG(CASE WHEN completed_at IS NOT NULL AND started_at IS NOT NULL 
                THEN EXTRACT(EPOCH FROM (completed_at - started_at)) END) as avg_duration_seconds
        FROM {self.table_name}
        GROUP BY model_name
        ORDER BY latest_request DESC
        """
        return self.conn.execute(query).fetchdf()

    def delete_request(self, request_id: str) -> bool:
        """
        Delete a specific training request record.
        
        Args:
            request_id (str): Unique training request identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logging.info(f"Deleting training request {request_id}...")
            query = f"DELETE FROM {self.table_name} WHERE request_id = ?"
            self.conn.execute(query, (request_id,))
            logging.info(f"Successfully deleted training request {request_id}")
            return True
        except Exception as e:
            logging.error(f"Error deleting training request {request_id}: {str(e)}")
            return False

    def close(self):
        """Close the database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            logging.info("Database connection closed")
