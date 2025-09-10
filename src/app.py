import logging

logging.basicConfig(level=logging.INFO)

from b3.model import B3Model
from dotenv import load_dotenv

load_dotenv()


class AssetPredictApp(object):
    def __init__(self, b3_model: B3Model) -> None:
        self._b3_model = b3_model

    def create_b3_model(self):
        # Initialize the model generator
        # Train the model and save it (this will use all numeric features by default)
        self._b3_model.train_and_save(model_dir="models")

    def predict(self, ticker: str):
        self._b3_predict(ticker)

    def _b3_predict(self, ticker: str):
        data_loader = self._b3_model.get_loader()
        new_data = data_loader.fetch(ticker=ticker)
        features = [col for col in new_data.select_dtypes(include=['number']).columns if col != 'date']
        X_new = new_data[features]

        # Predict cluster (buy/sell) for new data
        predictions = self._b3_model.predict(X_new, model_dir="models")
        logging.info("Predicted actions: {}".format(predictions))

    def get_model(self):
        return self._b3_model


if __name__ == '__main__':
    app = AssetPredictApp(B3Model())
    # app.create_b3_model()
    app.predict(ticker='BTCI11')
