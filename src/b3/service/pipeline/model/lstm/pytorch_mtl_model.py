import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from pandas import Series
from typing import Optional

from b3.service.pipeline.model.lstm.lstm_config import LSTMConfig

LABELS = ["buy", "sell", "hold"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


class LSTMNet(nn.Module):
    def __init__(self, input_features: int, hidden_size: int, dropout: float):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.action_head = nn.Linear(hidden_size, len(LABELS))
        self.return_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, timesteps, features)
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        # lstm_out shape: (batch_size, timesteps, hidden_size)
        last_out = lstm_out[:, -1, :]

        x = self.dropout(last_out)

        action_logits = self.action_head(x)
        return_pred = self.return_head(x)

        return action_logits, return_pred


class B3PytorchMTLModel:
    """
    Wrapper exposing predict methods similar to scikit-learn for compatibility
    with existing evaluation flows. It returns action labels for predict and
    provides predict_return for the regression head.
    """

    def __init__(self, input_timesteps: int, input_features: int, config: Optional[LSTMConfig] = None, device=None):
        self.config = config or LSTMConfig()
        self.input_timesteps = input_timesteps
        self.input_features = input_features
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMNet(input_features, self.config.units, self.config.dropout).to(self.device)
        self.history = {
            'loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_mae': []
        }

    def fit(self, X, y_action, y_return):
        # Reset history
        self.history = {
            'loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_mae': []
        }

        # Prepare targets
        if hasattr(y_action, 'astype'):
            y_action_clean = y_action.astype(str).str.replace("_target", "", regex=False)
        else:
            y_action_clean = Series(y_action).astype(str).str.replace("_target", "", regex=False)
            
        y_ids = np.array([LABEL_TO_ID[v] for v in y_action_clean], dtype=np.int64)
        
        # Convert to tensors
        if isinstance(X, torch.Tensor):
            X_tensor = X.float()
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32)
            
        y_action_tensor = torch.tensor(y_ids, dtype=torch.long)
        
        if isinstance(y_return, torch.Tensor):
            y_return_tensor = y_return.float()
        else:
            y_return_tensor = torch.tensor(y_return, dtype=torch.float32)
            
        if y_return_tensor.dim() == 1:
            y_return_tensor = y_return_tensor.unsqueeze(1) # shape (N, 1)

        dataset = TensorDataset(X_tensor, y_action_tensor, y_return_tensor)

        # Validation split (20%)
        val_size = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        # Loss and Optimizer
        criterion_action = nn.CrossEntropyLoss()
        criterion_return = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        logging.info(f"Training on {self.device}")

        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0.0

            for batch_X, batch_yA, batch_yR in train_loader:
                batch_X, batch_yA, batch_yR = batch_X.to(self.device), batch_yA.to(self.device), batch_yR.to(
                    self.device)

                optimizer.zero_grad()
                pred_logits, pred_return = self.model(batch_X)

                loss_action = criterion_action(pred_logits, batch_yA)
                loss_return = criterion_return(pred_return, batch_yR)

                loss = (self.config.loss_weight_action * loss_action) + \
                       (self.config.loss_weight_return * loss_return)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_X.size(0)

            avg_loss = total_loss / len(train_dataset)

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_mae = 0.0

            with torch.no_grad():
                for batch_X, batch_yA, batch_yR in val_loader:
                    batch_X, batch_yA, batch_yR = batch_X.to(self.device), batch_yA.to(self.device), batch_yR.to(
                        self.device)

                    pred_logits, pred_return = self.model(batch_X)

                    loss_action = criterion_action(pred_logits, batch_yA)
                    loss_return = criterion_return(pred_return, batch_yR)

                    loss = (self.config.loss_weight_action * loss_action) + \
                           (self.config.loss_weight_return * loss_return)

                    val_loss += loss.item() * batch_X.size(0)

                    # Metrics
                    preds_cls = torch.argmax(pred_logits, dim=1)
                    val_acc += (preds_cls == batch_yA).sum().item()
                    val_mae += torch.abs(pred_return - batch_yR).sum().item()

            avg_val_loss = val_loss / len(val_dataset)
            avg_val_acc = val_acc / len(val_dataset)
            avg_val_mae = val_mae / len(val_dataset)

            # Store history
            self.history['loss'].append(avg_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_acc'].append(avg_val_acc)
            self.history['val_mae'].append(avg_val_mae)

            logging.info(f"Epoch {epoch + 1}/{self.config.epochs} - "
                         f"loss: {avg_loss:.4f} - "
                         f"val_loss: {avg_val_loss:.4f} - "
                         f"val_acc: {avg_val_acc:.4f} - "
                         f"val_mae: {avg_val_mae:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        if isinstance(X, torch.Tensor):
            X_tensor = X.float().to(self.device)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Process in batches if X is large to avoid OOM
        batch_size = 256
        predictions = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X_tensor[i:i + batch_size]
                logits, _ = self.model(batch)
                probs = torch.softmax(logits, dim=1)
                ids = torch.argmax(probs, dim=1).cpu().numpy()
                predictions.extend([ID_TO_LABEL[i] for i in ids])

        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        if isinstance(X, torch.Tensor):
            X_tensor = X.float().to(self.device)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        batch_size = 256
        probs_list = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X_tensor[i:i + batch_size]
                logits, _ = self.model(batch)
                probs = torch.softmax(logits, dim=1)
                probs_list.extend(probs.cpu().numpy())

        return np.array(probs_list)

    def predict_return(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        if isinstance(X, torch.Tensor):
            X_tensor = X.float().to(self.device)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        batch_size = 256
        returns = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X_tensor[i:i + batch_size]
                _, pred_return = self.model(batch)
                returns.extend(pred_return.cpu().numpy().flatten())

        return np.array(returns)
