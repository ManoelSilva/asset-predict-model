from typing import Optional

import numpy as np
from pandas import Series
from tensorflow import keras
from tensorflow.keras import layers

from b3.service.pipeline.model.lstm.lstm_config import LSTMConfig

LABELS = ["buy", "sell", "hold"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


class B3KerasMTLModel:
    """
    Wrapper exposing predict methods similar to scikit-learn for compatibility
    with existing evaluation flows. It returns action labels for predict and
    provides predict_return for the regression head.
    """

    def __init__(self, input_timesteps: int, input_features: int, config: Optional[LSTMConfig] = None):
        self.config = config or LSTMConfig()
        self.input_timesteps = input_timesteps
        self.input_features = input_features
        self.model = self._build_model()

    def _build_model(self):
        inputs = layers.Input(shape=(self.input_timesteps, self.input_features))
        x = layers.LSTM(self.config.units, dropout=self.config.dropout)(inputs)
        action_out = layers.Dense(len(LABELS), activation="softmax", name="action")(x)
        ret_out = layers.Dense(1, activation="linear", name="ret")(x)
        model = keras.Model(inputs=inputs, outputs={"action": action_out, "ret": ret_out})
        model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss={"action": "categorical_crossentropy", "ret": "mse"},
            loss_weights={"action": self.config.loss_weight_action, "ret": self.config.loss_weight_return},
            metrics={"action": ["accuracy"], "ret": ["mae"]}
        )
        return model

    def fit(self, X: np.ndarray, y_action: Series, y_return: np.ndarray):
        y_action_clean = y_action.astype(str).str.replace("_target", "", regex=False)
        y_ids = np.array([LABEL_TO_ID[v] for v in y_action_clean], dtype=np.int32)
        y_cat = keras.utils.to_categorical(y_ids, num_classes=len(LABELS))
        self.model.fit(X, {"action": y_cat, "ret": y_return},
                       epochs=self.config.epochs, batch_size=self.config.batch_size,
                       validation_split=0.2, verbose=1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        outs = self.model.predict(X, verbose=0)
        action_probs = outs["action"] if isinstance(outs, dict) else outs[0]
        ids = action_probs.argmax(axis=1)
        return np.array([ID_TO_LABEL[i] for i in ids])

    def predict_return(self, X: np.ndarray) -> np.ndarray:
        outs = self.model.predict(X, verbose=0)
        ret = outs["ret"] if isinstance(outs, dict) else outs[1]
        return ret.reshape(-1)
