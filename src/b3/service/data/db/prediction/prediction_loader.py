import datetime
import json
import logging
import duckdb
import pandas as pd


class PredictionDataLoader:
    def __init__(self, db_path="md:b3"):
        """
        Initialize the DuckDB connection for predictions.
        Args:
            db_path (str): Path to DuckDB database file or ':memory:'.
        """
        self.db_path = db_path
        self.conn = duckdb.connect(database=db_path)
        self.table_name = "b3_predictions"
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Create the predictions table if it doesn't exist."""
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            ticker VARCHAR,
            date DATE,
            features JSON,
            prediction JSON,
            model_type VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            predicted_price FLOAT
        )
        """
        self.conn.execute(query)
        
        # Migration: Check if predicted_price column exists
        try:
            self.conn.execute(f"SELECT predicted_price FROM {self.table_name} LIMIT 0")
        except Exception:
            logging.info(f"Adding predicted_price column to {self.table_name}")
            try:
                self.conn.execute(f"ALTER TABLE {self.table_name} ADD COLUMN predicted_price FLOAT")
            except Exception as e:
                logging.error(f"Failed to add predicted_price column: {e}")

    def save_prediction(self, ticker, features, prediction, model_type, date=None, predicted_price=None):
        """
        Save a prediction to the database.
        
        Args:
            ticker (str): Asset ticker
            features (dict/list/df): Input features used for prediction
            prediction (dict/list/array): Prediction result
            model_type (str): Type of model used
            date (str, optional): Date of the data point. Defaults to today.
            predicted_price (float, optional): Predicted price value.
        """
        try:
            if date is None:
                date = datetime.date.today().strftime('%Y-%m-%d')
            
            # Convert complex types to JSON strings
            if hasattr(features, 'to_json'):
                features_json = features.to_json()
            elif hasattr(features, 'tolist'):
                features_json = json.dumps(features.tolist())
            else:
                features_json = json.dumps(features)
                
            if hasattr(prediction, 'tolist'):
                prediction_json = json.dumps(prediction.tolist())
            else:
                prediction_json = json.dumps(prediction)

            # Ensure predicted_price is a float or None (handle numpy types)
            if predicted_price is not None:
                try:
                    predicted_price = float(predicted_price)
                except:
                    predicted_price = None

            query = f"""
            INSERT INTO {self.table_name} (ticker, date, features, prediction, model_type, predicted_price)
            VALUES (?, ?, ?, ?, ?, ?)
            """
            self.conn.execute(query, (ticker, date, features_json, prediction_json, model_type, predicted_price))
            logging.info(f"Saved prediction for {ticker} (date: {date}, price: {predicted_price})")
            
        except Exception as e:
            logging.error(f"Error saving prediction: {e}")

    def fetch_predictions(self, ticker=None, limit=1000):
        """Fetch saved predictions."""
        query = f"SELECT * FROM {self.table_name}"
        params = []
        
        if ticker:
            query += " WHERE ticker = ?"
            params.append(ticker)
            
        query += " ORDER BY date DESC, created_at DESC LIMIT ?"
        params.append(limit)
        
        return self.conn.execute(query, params).fetchdf()

    def fetch_labeled_data_for_retraining(self, ticker_filter=None):
        """
        Fetch data for transfer learning/fine-tuning.
        This joins the saved predictions with the 'b3_featured' table to get ground truth targets.
        
        Args:
            ticker_filter (list, optional): List of tickers to include.
            
        Returns:
            pd.DataFrame: Dataframe with features and targets, ready for pipeline processing.
        """
        # We assume b3_featured has the ground truth (targets) and up-to-date features.
        # We join on ticker and date.
        # We only want rows where we have both a prediction record (meaning we cared about it)
        # and a ground truth record (meaning we can learn from it).
        
        # Note: b3_featured usually contains columns like 'target', 'date', 'ticker', and feature columns.
        
        base_query = """
        SELECT f.* 
        FROM b3_predictions p
        JOIN b3_featured f ON p.ticker = f.ticker AND p.date = f.date
        WHERE 1=1
        """
        
        # Filter for rows that actually have targets defined (not null)
        # Assuming target column names contain 'target'
        # But specifically we look for the main target column usually used.
        # We'll rely on the fact that b3_featured generally has targets.
        
        params = []
        if ticker_filter:
            placeholders = ','.join(['?'] * len(ticker_filter))
            base_query += f" AND p.ticker IN ({placeholders})"
            params.extend(ticker_filter)
            
        # Deduplicate: if we predicted multiple times for the same day, we only need the training data once.
        base_query = f"SELECT DISTINCT * FROM ({base_query}) t"
        
        logging.info("Fetching labeled data for retraining...")
        df = self.conn.execute(base_query, params).fetchdf()
        
        if df.empty:
            logging.warning("No labeled data found for retraining (intersection of predictions and featured data).")
            
        return df

