import argparse
import logging

from b3.service.pipeline.model.manager import ModelManagerService

logging.basicConfig(level=logging.INFO)

from dotenv import load_dotenv

load_dotenv()


class AssetPredictApp(object):
    def __init__(self, model_manager: ModelManagerService) -> None:
        self._model_manager = model_manager

    def train(self, n_jobs: int = 5):
        self._model_manager.train(model_dir="models", n_jobs=n_jobs)

    def predict(self, ticker: str):
        return self._b3_predict(ticker)

    def _b3_predict(self, ticker: str):
        data_loader = self._model_manager.get_loader()
        new_data = data_loader.fetch(ticker=ticker)
        features = [col for col in new_data.select_dtypes(include=['number']).columns if col != 'date']
        X_new = new_data[features]

        # Predict cluster (buy/sell) for new data
        predictions = self._model_manager.predict(X_new, model_dir="models")
        logging.info("Predicted actions: {}".format(predictions))
        return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Asset Predict Model CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the B3 model')
    train_parser.add_argument('--n_jobs', type=int, default=5, help='Number of jobs for training (parallelism)')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict using the B'
                                                           ''
                                                           '3 model')
    predict_parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol to predict')

    args = parser.parse_args()
    app = AssetPredictApp(ModelManagerService())

    if args.command == 'train':
        # Pass n_jobs to the model's train_and_save method
        app.train(n_jobs=args.n_jobs)
    elif args.command == 'predict':
        app.predict(ticker=args.ticker)
