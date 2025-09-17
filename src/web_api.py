import logging
from flask import Flask, request, jsonify
import threading
from waitress import serve
from flask_cors import CORS

from app import AssetPredictApp
from b3.service.model import B3Model


def create_flask_app():
    flask_app = Flask(__name__)
    CORS(flask_app, origins=["http://localhost:4200"])
    asset_predict_app = AssetPredictApp(B3Model())

    @flask_app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json()
        if not data or 'ticker' not in data:
            return jsonify({'error': 'Missing ticker in request'}), 400
        ticker = data['ticker']
        try:
            predictions = asset_predict_app.predict(ticker)
            return jsonify({'predictions': list(predictions)})
        except Exception as e:
            logging.exception("Prediction error")
            return jsonify({'error': str(e)}), 500

    @flask_app.route('/train', methods=['POST'])
    def train():
        def run_training(n_jobs):
            try:
                asset_predict_app.train(n_jobs=n_jobs)
                logging.info('Model training completed successfully.')
            except Exception as e:
                logging.exception('Training error in background thread', e)

        n_jobs = request.json.get('n_jobs', 5) if request.is_json else 5
        thread = threading.Thread(target=run_training, args=(n_jobs,))
        thread.start()
        return jsonify({'message': 'Model training started in background.'}), 202

    return flask_app


if __name__ == '__main__':
    serve(create_flask_app(), host='0.0.0.0', port=5001)
