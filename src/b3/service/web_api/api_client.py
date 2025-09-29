import requests

class AssetApiClient:
    BASE_URL = "http://localhost:5002/asset/"

    @staticmethod
    def fetch_ticker_info(ticker):
        url = f"{AssetApiClient.BASE_URL}{ticker}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json(), None
        except requests.RequestException as e:
            return None, str(e)
