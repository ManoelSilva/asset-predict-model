import os
from typing import Optional, Tuple, List, Dict

import requests


class AssetApiClient:
    BASE_URL = os.getenv("ASSET_API_BASE_URL", "http://localhost:5002/asset/")

    @staticmethod
    def fetch_ticker_info(ticker):
        """Fetch current ticker information."""
        url = f"{AssetApiClient.BASE_URL}{ticker}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json(), None
        except requests.RequestException as e:
            return None, str(e)

    @staticmethod
    def fetch_historical_data(ticker: str, days: int = 52, end_date: Optional[str] = None) -> Tuple[
        Optional[List[Dict]], Optional[str]]:
        """
        Fetch historical data for a ticker from the asset data lake API.
        
        Args:
            ticker: Ticker symbol (e.g., "PETR4")
            days: Number of days of historical data to fetch (default: 52 to give 32 for LSTM lookback)
            end_date: End date in YYYY-MM-DD format (default: today)
            
        Returns:
            Tuple of (historical_data_list, error_message)
        """
        url = f"{AssetApiClient.BASE_URL}{ticker}/history?days={days}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Handle response format
            if isinstance(data, list):
                return data, None
            elif isinstance(data, dict):
                # Check for data key
                if 'data' in data and isinstance(data['data'], list):
                    return data['data'], None
                # If no data key, return the whole dict as single record
                return [data], None
            else:
                return None, f"Unexpected response format from Asset API: {type(data)}"

        except requests.RequestException as e:
            return None, f"Error fetching historical data from Asset API: {str(e)}"
