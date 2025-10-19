import datetime
import logging

import duckdb
import pandas as pd


class B3DataLoader(object):
    def __init__(self, db_path="md:b3"):
        """
        Initialize the DuckDB connection.
        Args:
            db_path (str): Path to DuckDB database file or ':memory:' for in-memory DB.
        """
        self.db_path = db_path
        self.conn = duckdb.connect(database=db_path)
        self.table_name = "b3_featured"

    def fetch_all(self):
        """
        Fetch all data from the table.
        Returns:
            pd.DataFrame: All data.
        """
        logging.info("Fetching all data from b3_featured..")
        # Join with historical table to bring in raw close price
        query = f"""
        SELECT f.*, h.close AS close
        FROM {self.table_name} AS f
        LEFT JOIN b3_hist AS h
          ON f.date = h.date AND f.ticker = h.ticker
        """
        return self.conn.execute(query).fetchdf()

    def fetch(self, start_date=None, end_date=None, ticker=None):
        """
        Fetch data for a specific date or date range, optionally filtered by ticker.
        Args:
            start_date (str, optional): Start date (inclusive), format 'YYYY-MM-DD'. Defaults to today if not provided.
            end_date (str, optional): End date (inclusive), format 'YYYY-MM-DD'.
            ticker (str, optional): Ticker symbol to filter by.
        Returns:
            pd.DataFrame: Filtered data.
        """
        if start_date is None:
            start_date = self._get_default_start_date(ticker)
        base_select = f"SELECT f.*, h.close AS close FROM {self.table_name} AS f LEFT JOIN b3_hist AS h ON f.date = h.date AND f.ticker = h.ticker"
        if end_date and ticker:
            query = base_select + " WHERE f.date >= ? AND f.date <= ? AND f.ticker = ?"
            params = (start_date, end_date, ticker)
        elif end_date:
            query = base_select + " WHERE f.date >= ? AND f.date <= ?"
            params = (start_date, end_date)
        elif ticker:
            query = base_select + " WHERE f.date = ? AND f.ticker = ?"
            params = (start_date, ticker)
        else:
            query = base_select + " WHERE f.date = ?"
            params = (start_date,)
        return self.conn.execute(query, params).fetchdf()

    def _get_default_start_date(self, ticker=None):
        today_str = datetime.date.today().strftime('%Y-%m-%d')
        if ticker:
            query = f"SELECT MAX(date) as last_date FROM {self.table_name} WHERE ticker = ?"
            result = self.conn.execute(query, (ticker,)).fetchone()
        else:
            query = f"SELECT MAX(date) as last_date FROM {self.table_name}"
            result = self.conn.execute(query).fetchone()
        last_date = result[0] if result and result[0] is not None else None
        if last_date:
            return last_date
        return today_str
