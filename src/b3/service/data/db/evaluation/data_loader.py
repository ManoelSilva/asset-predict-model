import json
import logging
from datetime import datetime
from typing import Dict, Any

import duckdb
import pandas as pd


class EvaluationDataLoader:
    """
    Data loader for managing model evaluation results in DuckDB.
    Handles persistence of evaluation metrics, metadata, and visualization paths.
    """

    def __init__(self, db_path="md:b3"):
        """
        Initialize the DuckDB connection for evaluation data.
        
        Args:
            db_path (str): Path to DuckDB database file or ':memory:' for in-memory DB.
                          Defaults to MotherDuck connection.
        """
        self.db_path = db_path
        self.conn = duckdb.connect(database=db_path)
        self.table_name = "b3_model_evaluation"
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self):
        """
        Create the evaluation table if it doesn't exist.
        Stores evaluation metadata, metrics, and visualization paths.
        """
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            evaluation_id VARCHAR PRIMARY KEY,
            model_name VARCHAR NOT NULL,
            evaluation_timestamp TIMESTAMP NOT NULL,
            dataset_type VARCHAR NOT NULL,  -- 'validation' or 'test'
            visualization_path VARCHAR,
            metrics_json TEXT,  -- JSON string containing all metrics
            precision_buy DOUBLE,
            precision_sell DOUBLE,
            precision_hold DOUBLE,
            recall_buy DOUBLE,
            recall_sell DOUBLE,
            recall_hold DOUBLE,
            f1_score_buy DOUBLE,
            f1_score_sell DOUBLE,
            f1_score_hold DOUBLE,
            accuracy DOUBLE,
            macro_avg_precision DOUBLE,
            macro_avg_recall DOUBLE,
            macro_avg_f1_score DOUBLE,
            weighted_avg_precision DOUBLE,
            weighted_avg_recall DOUBLE,
            weighted_avg_f1_score DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        logging.info(f"Creating table {self.table_name} if not exists...")
        self.conn.execute(create_table_sql)

    def save_evaluation(self, evaluation_id: str, model_name: str, dataset_type: str,
                        evaluation_results: Dict[str, Any], visualization_path: str = None) -> bool:
        """
        Save model evaluation results to the database.
        
        Args:
            evaluation_id (str): Unique identifier for this evaluation
            model_name (str): Name of the model being evaluated
            dataset_type (str): Type of dataset ('validation' or 'test')
            evaluation_results (dict): Classification report results from sklearn
            visualization_path (str, optional): Path to saved visualization
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logging.info(f"Saving evaluation {evaluation_id} for model {model_name} on {dataset_type} set...")

            # Extract metrics from classification report
            metrics = self._extract_metrics(evaluation_results)

            # Convert metrics to JSON for storage
            metrics_json = json.dumps(evaluation_results, default=str)

            # Prepare data for insertion
            data = {
                'evaluation_id': evaluation_id,
                'model_name': model_name,
                'evaluation_timestamp': datetime.now(),
                'dataset_type': dataset_type,
                'visualization_path': visualization_path,
                'metrics_json': metrics_json,
                **metrics
            }

            # Insert data
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data.keys()])
            values = list(data.values())

            insert_sql = f"""
            INSERT OR REPLACE INTO {self.table_name} ({columns})
            VALUES ({placeholders})
            """

            self.conn.execute(insert_sql, values)
            logging.info(f"Successfully saved evaluation {evaluation_id}")
            return True

        except Exception as e:
            logging.error(f"Error saving evaluation {evaluation_id}: {str(e)}")
            return False

    def _extract_metrics(self, evaluation_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract key metrics from sklearn classification report.
        
        Args:
            evaluation_results (dict): Classification report from sklearn
            
        Returns:
            dict: Extracted metrics as key-value pairs
        """
        metrics = {}

        # Extract per-class metrics
        for class_name in ['buy', 'sell', 'hold']:
            if class_name in evaluation_results:
                class_metrics = evaluation_results[class_name]
                metrics[f'precision_{class_name}'] = class_metrics.get('precision', 0.0)
                metrics[f'recall_{class_name}'] = class_metrics.get('recall', 0.0)
                metrics[f'f1_score_{class_name}'] = class_metrics.get('f1-score', 0.0)

        # Extract overall metrics
        metrics['accuracy'] = evaluation_results.get('accuracy', 0.0)

        # Extract macro averages
        if 'macro avg' in evaluation_results:
            macro_avg = evaluation_results['macro avg']
            metrics['macro_avg_precision'] = macro_avg.get('precision', 0.0)
            metrics['macro_avg_recall'] = macro_avg.get('recall', 0.0)
            metrics['macro_avg_f1_score'] = macro_avg.get('f1-score', 0.0)

        # Extract weighted averages
        if 'weighted avg' in evaluation_results:
            weighted_avg = evaluation_results['weighted avg']
            metrics['weighted_avg_precision'] = weighted_avg.get('precision', 0.0)
            metrics['weighted_avg_recall'] = weighted_avg.get('recall', 0.0)
            metrics['weighted_avg_f1_score'] = weighted_avg.get('f1-score', 0.0)

        return metrics

    def fetch_all(self) -> pd.DataFrame:
        """
        Fetch all evaluation records.
        
        Returns:
            pd.DataFrame: All evaluation data
        """
        logging.info(f"Fetching all data from {self.table_name}...")
        return self.conn.execute(f"SELECT * FROM {self.table_name}").fetchdf()

    def fetch_by_model(self, model_name: str) -> pd.DataFrame:
        """
        Fetch all evaluations for a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            pd.DataFrame: Evaluation data for the model
        """
        logging.info(f"Fetching evaluations for model {model_name}...")
        query = f"SELECT * FROM {self.table_name} WHERE model_name = ?"
        return self.conn.execute(query, (model_name,)).fetchdf()

    def fetch_by_evaluation_id(self, evaluation_id: str) -> pd.DataFrame:
        """
        Fetch specific evaluation by ID.
        
        Args:
            evaluation_id (str): Unique evaluation identifier
            
        Returns:
            pd.DataFrame: Evaluation data
        """
        logging.info(f"Fetching evaluation {evaluation_id}...")
        query = f"SELECT * FROM {self.table_name} WHERE evaluation_id = ?"
        return self.conn.execute(query, (evaluation_id,)).fetchdf()

    def fetch_by_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch evaluations within a date range.
        
        Args:
            start_date (str): Start date (inclusive), format 'YYYY-MM-DD'
            end_date (str): End date (inclusive), format 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: Evaluation data within date range
        """
        logging.info(f"Fetching evaluations from {start_date} to {end_date}...")
        query = f"""
        SELECT * FROM {self.table_name} 
        WHERE DATE(evaluation_timestamp) >= ? AND DATE(evaluation_timestamp) <= ?
        """
        return self.conn.execute(query, (start_date, end_date)).fetchdf()

    def fetch_latest_by_model(self, model_name: str) -> pd.DataFrame:
        """
        Fetch the latest evaluation for a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            pd.DataFrame: Latest evaluation data for the model
        """
        logging.info(f"Fetching latest evaluation for model {model_name}...")
        query = f"""
        SELECT * FROM {self.table_name} 
        WHERE model_name = ? 
        ORDER BY evaluation_timestamp DESC 
        LIMIT 1
        """
        return self.conn.execute(query, (model_name,)).fetchdf()

    def get_evaluation_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of all evaluations.
        
        Returns:
            pd.DataFrame: Summary statistics
        """
        logging.info("Generating evaluation summary...")
        query = f"""
        SELECT 
            model_name,
            COUNT(*) as total_evaluations,
            COUNT(CASE WHEN dataset_type = 'validation' THEN 1 END) as validation_count,
            COUNT(CASE WHEN dataset_type = 'test' THEN 1 END) as test_count,
            MAX(evaluation_timestamp) as latest_evaluation,
            AVG(accuracy) as avg_accuracy,
            AVG(macro_avg_f1_score) as avg_macro_f1,
            AVG(weighted_avg_f1_score) as avg_weighted_f1
        FROM {self.table_name}
        GROUP BY model_name
        ORDER BY latest_evaluation DESC
        """
        return self.conn.execute(query).fetchdf()

    def delete_evaluation(self, evaluation_id: str) -> bool:
        """
        Delete a specific evaluation record.
        
        Args:
            evaluation_id (str): Unique evaluation identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logging.info(f"Deleting evaluation {evaluation_id}...")
            query = f"DELETE FROM {self.table_name} WHERE evaluation_id = ?"
            result = self.conn.execute(query, (evaluation_id,))
            logging.info(f"Successfully deleted evaluation {evaluation_id}")
            return True
        except Exception as e:
            logging.error(f"Error deleting evaluation {evaluation_id}: {str(e)}")
            return False

    def close(self):
        """Close the database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            logging.info("Database connection closed")
