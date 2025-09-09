import duckdb
import pandas as pd
import datetime


class B3DataLoader(object):
    _URL = 'https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_D05092025.ZIP'

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
        return self.conn.execute(f"SELECT * FROM {self.table_name}").fetchdf()

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
        if end_date and ticker:
            query = f"SELECT * FROM {self.table_name} WHERE date >= ? AND date <= ? AND ticker = ?"
            params = (start_date, end_date, ticker)
        elif end_date:
            query = f"SELECT * FROM {self.table_name} WHERE date >= ? AND date <= ?"
            params = (start_date, end_date)
        elif ticker:
            query = f"SELECT * FROM {self.table_name} WHERE date = ? AND ticker = ?"
            params = (start_date, ticker)
        else:
            query = f"SELECT * FROM {self.table_name} WHERE date = ?"
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