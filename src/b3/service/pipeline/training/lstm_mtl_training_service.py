import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

LABELS = ["buy", "sell", "hold"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


@dataclass
class LSTMMultiTaskConfig:
    lookback: int = 32
    horizon: int = 1
    units: int = 96
    dropout: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 25
    batch_size: int = 128
    loss_weight_action: float = 1.0
    loss_weight_return: float = 0.5


class B3KerasMTLModel:
    """
    Wrapper exposing predict methods similar to scikit-learn for compatibility
    with existing evaluation flows. It returns action labels for predict and
    provides predict_return for the regression head.
    """

    def __init__(self, input_timesteps: int, input_features: int, config: Optional[LSTMMultiTaskConfig] = None):
        self.config = config or LSTMMultiTaskConfig()
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


class B3LSTMMultiTaskTrainingService:
    _RANDOM_STATE = 42

    @staticmethod
    def _group_keys(df: DataFrame):
        if "ticker" in df.columns:
            return list(df["ticker"].astype(str).unique())
        return [None]

    @staticmethod
    def _df_for_group(df: DataFrame, group: Optional[str]) -> DataFrame:
        gdf = df if group is None else df[df["ticker"].astype(str) == group]
        for col in ["date", "datetime", "timestamp"]:
            if col in gdf.columns:
                return gdf.sort_values(col)
        return gdf

    def build_sequences(self, X_df: DataFrame, y_action: Series, df_full: DataFrame,
                        price_col: str = "close", lookback: int = 32, horizon: int = 1
                        ) -> Tuple[np.ndarray, Series, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build sequences for multi-task learning.

        Returns:
            X_seq: (N, lookback, n_features)
            y_action_seq: Series of action labels aligned to sequences
            y_return_seq: (N,) predicted target returns over horizon
            last_price_seq: (N,) price at prediction time (for price mapping)
            future_price_seq: (N,) actual future price at t+h (for regression eval)
        """
        features = X_df.columns.tolist()
        X_list, act_list, ret_list, last_price_list, future_price_list = [], [], [], [], []

        for key in self._group_keys(df_full):
            gdf = self._df_for_group(df_full, key)
            Xg = X_df.loc[gdf.index, features].values.astype("float32")
            yg = y_action.loc[gdf.index].astype(str).str.replace("_target", "", regex=False)
            if price_col not in gdf.columns:
                raise ValueError(f"Price column '{price_col}' not found in dataframe")
            pg = gdf[price_col].astype(float).values

            if len(Xg) <= lookback + horizon:
                continue

            for i in range(lookback, len(Xg) - horizon):
                X_list.append(Xg[i - lookback:i])
                act_list.append(yg.iloc[i])
                p0 = pg[i - 1]
                p1 = pg[i - 1 + horizon]
                r = (p1 / p0) - 1.0
                ret_list.append(r)
                last_price_list.append(p0)
                future_price_list.append(p1)

        if not X_list:
            return (np.zeros((0, lookback, len(features)), dtype="float32"),
                    pd.Series([], dtype="object"),
                    np.zeros((0,), dtype="float32"),
                    np.zeros((0,), dtype="float32"),
                    np.zeros((0,), dtype="float32"))

        X_seq = np.stack(X_list, axis=0)
        y_action_seq = pd.Series(act_list, name="target")
        y_return_seq = np.array(ret_list, dtype="float32")
        last_price_seq = np.array(last_price_list, dtype="float32")
        future_price_seq = np.array(future_price_list, dtype="float32")
        return X_seq, y_action_seq, y_return_seq, last_price_seq, future_price_seq

    def split_sequences(self, X: np.ndarray, y_action: Series, y_return: np.ndarray,
                        last_price: np.ndarray, future_price: np.ndarray,
                        test_size: float, val_size: float):
        y_norm = y_action.astype(str)
        X_train, X_temp, yA_train, yA_temp, yR_train, yR_temp, p0_train, p0_temp, pf_train, pf_temp = train_test_split(
            X, y_norm, y_return, last_price, future_price,
            test_size=test_size, random_state=self._RANDOM_STATE, stratify=y_norm
        )
        X_val, X_test, yA_val, yA_test, yR_val, yR_test, p0_val, p0_test, pf_val, pf_test = train_test_split(
            X_temp, yA_temp, yR_temp, p0_temp, pf_temp,
            test_size=val_size, random_state=self._RANDOM_STATE, stratify=yA_temp
        )
        return (X_train, X_val, X_test,
                yA_train, yA_val, yA_test,
                yR_train, yR_val, yR_test,
                p0_train, p0_val, p0_test,
                pf_train, pf_val, pf_test)

    def train_model(self, X_train: np.ndarray, yA_train: Series, yR_train: np.ndarray,
                    lookback: int, n_features: int, config: Optional[LSTMMultiTaskConfig] = None) -> B3KerasMTLModel:
        cfg = config or LSTMMultiTaskConfig(lookback=lookback)
        clf = B3KerasMTLModel(input_timesteps=lookback, input_features=n_features, config=cfg)
        clf.fit(X_train, yA_train, yR_train)
        logging.info("LSTM multi-task training completed")
        return clf


