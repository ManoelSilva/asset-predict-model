import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
import numpy as np
from pandas import Series
from typing import Optional
from sklearn.metrics import f1_score

from b3.service.pipeline.model.lstm.lstm_config import LSTMConfig

LABELS = ["buy", "sell", "hold"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits
        # targets: labels
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


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
            'val_f1': [],
            'val_mae': []
        }

    @staticmethod
    def _prepare_tensors(X, y_action, y_return):
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
            y_return_tensor = y_return_tensor.unsqueeze(1)  # shape (N, 1)

        return X_tensor, y_action_tensor, y_return_tensor

    def fit(self, X, y_action, y_return, X_val=None, y_val=None, yR_val=None):
        # Reset history
        self.history = {
            'loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_mae': []
        }

        X_train_t, y_train_t, yR_train_t = self._prepare_tensors(X, y_action, y_return)

        if X_val is not None and y_val is not None and yR_val is not None:
            train_dataset = TensorDataset(X_train_t, y_train_t, yR_train_t)
            X_val_t, y_val_t, yR_val_t = self._prepare_tensors(X_val, y_val, yR_val)
            val_dataset = TensorDataset(X_val_t, y_val_t, yR_val_t)
            logging.info("Using provided validation set")
        else:
            dataset = TensorDataset(X_train_t, y_train_t, yR_train_t)
            # Validation split (20%)
            val_size = int(0.2 * len(dataset))
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            logging.info(f"Splitting training data: Train={train_size}, Val={val_size}")

        # -----------------------------------------------------
        # OVERSAMPLING & WEIGHTING SETUP
        # -----------------------------------------------------
        # Get labels for training data subset
        if isinstance(train_dataset, torch.utils.data.Subset):
            indices = train_dataset.indices
            y_train_tensor = train_dataset.dataset.tensors[1][indices]
        else:
            y_train_tensor = train_dataset.tensors[1]

        y_train_np = y_train_tensor.cpu().numpy()

        # Calculate class counts in training set
        class_counts = np.bincount(y_train_np, minlength=len(LABELS))
        # Handle zero counts safely
        class_counts = np.maximum(class_counts, 1)

        # Calculate weights: inverse frequency (1/N)
        class_weights = 1.0 / class_counts

        # Assign weight to each sample
        sample_weights = class_weights[y_train_np]
        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float)

        # Create WeightedRandomSampler
        # This will oversample minority classes by picking them more often
        sampler = WeightedRandomSampler(
            weights=sample_weights_tensor,
            num_samples=len(sample_weights_tensor),
            replacement=True
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        # Loss and Optimizer
        # Using FocalLoss for class imbalance handling on top of sampling
        criterion_action = FocalLoss(gamma=2.0)
        criterion_return = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        logging.info(f"Training on {self.device} with Oversampling & FocalLoss")

        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0.0
            total_loss_action = 0.0
            total_loss_return = 0.0

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
                total_loss_action += loss_action.item() * batch_X.size(0)
                total_loss_return += loss_return.item() * batch_X.size(0)

            avg_loss = total_loss / len(train_dataset)
            avg_loss_action = total_loss_action / len(train_dataset)
            avg_loss_return = total_loss_return / len(train_dataset)

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_mae = 0.0

            all_preds = []
            all_targets = []

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
                    all_preds.extend(preds_cls.cpu().numpy())
                    all_targets.extend(batch_yA.cpu().numpy())

                    val_mae += torch.abs(pred_return - batch_yR).sum().item()

            avg_val_loss = val_loss / len(val_dataset)
            val_f1 = f1_score(all_targets, all_preds, average='macro')
            avg_val_mae = val_mae / len(val_dataset)

            # Store history
            self.history['loss'].append(avg_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_f1'].append(val_f1)
            self.history['val_mae'].append(avg_val_mae)

            logging.info(f"Epoch {epoch + 1}/{self.config.epochs} - "
                         f"loss: {avg_loss:.4f} (Act: {avg_loss_action:.4f}, Ret: {avg_loss_return:.4f}) - "
                         f"val_loss: {avg_val_loss:.4f} - "
                         f"val_f1: {val_f1:.4f} - "
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
