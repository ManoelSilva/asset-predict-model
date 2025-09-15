import logging
from flask import Flask, request, jsonify
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

    return flask_app


if __name__ == '__main__':
    create_flask_app().run(host='0.0.0.0', port=5001)
