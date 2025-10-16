import logging
from typing import cast

from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split


class B3ModelTrainingService:
    """
    Service responsible for training B3 prediction models including data splitting
    and hyperparameter tuning.
    """

    _RANDOM_STATE = 42

    def split_data(self, X: DataFrame, y: Series, test_size: float = 0.2, val_size: float = 0.2):
        """
        Split data into train/validation/test sets with stratification.

        Args:
            X: Feature data
            y: Target labels
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation

        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, random_state=self._RANDOM_STATE, stratify=y
        )

        # Second split: separate validation set from remaining data
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=self._RANDOM_STATE, stratify=y_temp
        )

        logging.info(f"Data split - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def tune_hyperparameters(self, X_train: DataFrame, y_train: Series, n_jobs: int = 5) -> BaseEstimator:
        """
        Tune RandomForestClassifier hyperparameters using RandomizedSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training targets
            n_jobs: Number of parallel jobs
            
        Returns:
            RandomForestClassifier: Best tuned pipeline
        """
        logging.info("Starting hyperparameter tuning...")

        param_dist = {
            'min_samples_leaf': [1, 5],  # Reduced grid size
            'max_features': ['sqrt', 0.5]  # Reduced grid size
        }

        rf = RandomForestClassifier(random_state=self._RANDOM_STATE)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self._RANDOM_STATE)  # Fewer folds

        random_search = RandomizedSearchCV(
            rf, param_distributions=param_dist, n_iter=4, cv=cv, scoring='f1_weighted',
            n_jobs=n_jobs, verbose=1, random_state=self._RANDOM_STATE
        )

        random_search.fit(X_train, y_train)
        print("Best parameters found:", random_search.best_params_)
        logging.info(f"Hyperparameter tuning completed. Best params: {random_search.best_params_}")

        return random_search.best_estimator_

    def train_model(self, X_train: DataFrame, y_train: Series, n_jobs: int = 5) -> RandomForestClassifier:
        """
        Complete pipeline training pipeline including hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            n_jobs: Number of parallel jobs
            
        Returns:
            RandomForestClassifier: Trained pipeline
        """
        logging.info("Starting pipeline training...")
        model = self.tune_hyperparameters(X_train, y_train, n_jobs)
        logging.info("Model training completed")
        return cast(RandomForestClassifier, model)
