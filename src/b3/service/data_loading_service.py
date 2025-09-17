import logging
from pandas import DataFrame

from b3.service.data_loader import B3DataLoader


class B3DataLoadingService:
    """
    Service responsible for loading and fetching B3 market data.
    """
    
    def __init__(self):
        self._loader = B3DataLoader()
    
    def load_data(self) -> DataFrame:
        """
        Loads all B3 market data.
        
        Returns:
            DataFrame: Raw market data
        """
        logging.info("Loading B3 market data...")
        df = self._loader.fetch_all()
        logging.info(f"Loaded {len(df)} records")
        return df
    
    def get_loader(self):
        """
        Returns the underlying data loader instance.
        
        Returns:
            B3DataLoader: The data loader instance
        """
        return self._loader
