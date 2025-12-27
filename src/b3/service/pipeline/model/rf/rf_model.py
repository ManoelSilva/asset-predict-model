import logging
from typing import Dict, Any, Tuple, Optional, cast

import numpy as np
import pandas as pd
from asset_model_data_storage.data_storage_service import DataStorageService
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split

from b3.service.pipeline.model.model import BaseModel
from b3.service.pipeline.model.params import SplitDataParams, TrainModelParams, PrepareDataParams
from b3.service.pipeline.model.rf.rf_config import RandomForestConfig
from b3.service.pipeline.model.utils import prepare_enrichment_dataframe, get_feature_columns
from b3.service.pipeline.model_evaluation_service import B3ModelEvaluationService
from b3.service.pipeline.persist.rf_persist_service import RandomForestPersistService
from constants import RANDOM_STATE


class RandomForestModel(BaseModel):
    """
    Random Forest implementation for B3 asset prediction.
    """

    def __init__(self, config: Optional[RandomForestConfig] = None,
                 storage_service: Optional[DataStorageService] = None):
        """
        Initialize Random Forest pipeline.
        
        Args:
            config: Random Forest configuration
            storage_service: Data storage service for saving/loading models
        """
        super().__init__(config or RandomForestConfig())
        self._storage_service = storage_service or DataStorageService()
        self._saving_service = RandomForestPersistService(self._storage_service)
        self._evaluation_service = B3ModelEvaluationService(self._storage_service)

    def run_pipeline(self, X: pd.DataFrame, y: pd.Series, df_processed: pd.DataFrame, config: Any) -> Tuple[str, Dict]:
        """
        Runs the full Random Forest training pipeline.
        """
        logging.info("Starting Random Forest training pipeline")

        # 1. Prepare
        prepare_params = PrepareDataParams(X=X, y=y)
        X_prepared, y_prepared = self.prepare_data(prepare_params)

        # 2. Split
        split_params = SplitDataParams(
            X=X_prepared,
            y=y_prepared,
            test_size=config.test_size,
            val_size=config.val_size
        )
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(split_params)

        # 3. Train
        train_params = TrainModelParams(
            X_train=X_train,
            y_train=y_train
        )
        trained_model = self.train_model(train_params, n_jobs=config.n_jobs)

        # 4. Evaluate
        evaluation_results = self.evaluate_model(
            trained_model, X_val, y_val, X_test, y_test, df_processed
        )

        # 5. Save
        model_path = self.save_model(trained_model, config.model_dir)
        logging.info(f"Random Forest training completed successfully. Model saved at: {model_path}")

        return model_path, evaluation_results

    def prepare_data(self, params: PrepareDataParams, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for Random Forest training.
        For RF, data preparation is minimal - just ensure proper format.
        
        Args:
            params: PrepareDataParams object containing all parameters for data preparation
            **kwargs: Additional parameters (not used for RF)
            
        Returns:
            Tuple of prepared X and y data
        """
        if not self.validate_data(params.X, params.y):
            raise ValueError("Invalid input data for Random Forest")

        # Ensure proper data types
        X_prepared = params.X.copy()
        y_prepared = params.y.copy()

        # Convert any non-numeric columns to numeric if possible
        for col in X_prepared.columns:
            if X_prepared[col].dtype == 'object':
                try:
                    X_prepared[col] = pd.to_numeric(X_prepared[col], errors='coerce')
                except:
                    logging.warning(f"Could not convert column {col} to numeric")

        # Remove any rows with NaN values
        mask = ~(X_prepared.isna().any(axis=1) | y_prepared.isna())
        X_prepared = X_prepared[mask]
        y_prepared = y_prepared[mask]

        logging.info(f"Prepared data for Random Forest: {X_prepared.shape[0]} samples, {X_prepared.shape[1]} features")
        return X_prepared, y_prepared

    def split_data(self, params: SplitDataParams) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train/validation/test sets for Random Forest.
        
        Args:
            params: SplitDataParams object containing all parameters for data splitting
            
        Returns:
            Tuple containing (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_train, X_temp, y_train, y_temp = train_test_split(
            params.X, params.y, test_size=params.test_size, random_state=RANDOM_STATE, stratify=params.y
        )

        # Second split: separate validation set from remaining data
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=params.val_size, random_state=RANDOM_STATE, stratify=y_temp
        )

        logging.info(f"Data split - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_model(self, params: TrainModelParams, **kwargs) -> RandomForestClassifier:
        """
        Train Random Forest pipeline.
        
        Args:
            params: TrainModelParams object containing all parameters for model training
            **kwargs: Additional training parameters
            
        Returns:
            Trained RandomForestClassifier
        """
        n_jobs = kwargs.get('n_jobs', self.config.n_jobs)

        logging.info(f"Training Random Forest pipeline with {n_jobs} jobs")
        logging.info("Starting hyperparameter tuning...")

        param_dist = {
            'min_samples_leaf': [1, 5],  # Reduced grid size
            'max_features': ['sqrt', 0.5]  # Reduced grid size
        }

        rf = RandomForestClassifier(random_state=RANDOM_STATE)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)  # Fewer folds

        random_search = RandomizedSearchCV(
            rf, param_distributions=param_dist, n_iter=4, cv=cv, scoring='f1_weighted',
            n_jobs=n_jobs, verbose=1, random_state=RANDOM_STATE
        )

        random_search.fit(params.X_train, params.y_train)
        logging.info(f"Hyperparameter tuning completed. Best params: {random_search.best_params_}")

        model = cast(RandomForestClassifier, random_search.best_estimator_)

        self.model = model
        self.is_trained = True

        logging.info("Random Forest training completed successfully")
        return model

    def _enrich_df_with_predictions(self, df: pd.DataFrame, max_samples: int = 5):
        """
        Add predictions to the dataframe for visualization.
        Operates in-place.
        Optimized to batch predictions.
        """
        result = prepare_enrichment_dataframe(df, max_samples=max_samples)
        df, sample_tickers, sub_df = result

        if df is None or sub_df is None:
            return

        try:
            feature_cols = get_feature_columns(sub_df)

            if not feature_cols:
                return

            X_sub = sub_df[feature_cols]
            y_sub = sub_df['target'] if 'target' in sub_df.columns else pd.Series(index=sub_df.index)

            # Prepare data
            prepare_params = PrepareDataParams(X=X_sub, y=y_sub)
            X_prep, _ = self.prepare_data(prepare_params)

            if X_prep.empty:
                return

            # Predict
            preds = self.predict(X_prep)

            # Assign back using index
            df.loc[X_prep.index, 'predicted_action'] = preds

        except Exception as e:
            logging.warning(f"Failed to generate visualization predictions: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using trained Random Forest pipeline.
        
        Args:
            X: Feature data for prediction
            
        Returns:
            Array of predictions
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        if not self.validate_data(X):
            raise ValueError("Invalid input data for prediction")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities from Random Forest pipeline.
        
        Args:
            X: Feature data for prediction
            
        Returns:
            Array of prediction probabilities
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        if not self.validate_data(X):
            raise ValueError("Invalid input data for prediction")

        return self.model.predict_proba(X)

    def save_model(self, model: RandomForestClassifier, model_dir: str) -> str:
        """
        Save Random Forest pipeline to storage.
        
        Args:
            model: Trained RandomForestClassifier
            model_dir: Directory to save the pipeline
            
        Returns:
            Path to saved pipeline
        """
        return self._saving_service.save_model(model, model_dir)

    def load_model(self, model_path: str) -> RandomForestClassifier:
        """
        Load Random Forest pipeline from storage.
        
        Args:
            model_path: Path to saved pipeline
            
        Returns:
            Loaded RandomForestClassifier
        """
        model = self._saving_service.load_model(model_path)
        self.model = model
        self.is_trained = True
        return model

    def evaluate_model(self, model: RandomForestClassifier, X_val: pd.DataFrame, y_val: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series, df_processed: pd.DataFrame,
                       **kwargs) -> Dict[str, Any]:
        """
        Evaluate Random Forest pipeline.
        
        Args:
            model: Trained RandomForestClassifier
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            y_test: Test targets
            df_processed: Processed dataframe for visualization
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation results
        """
        model_name = kwargs.get('model_name', 'b3_random_forest')
        persist_results = kwargs.get('persist_results', True)

        # Enrich dataframe with predictions for visualization
        df_viz = df_processed.copy()
        # Increase max_samples to give plotter more options for diverse examples
        self._enrich_df_with_predictions(df_viz, max_samples=20)

        return self._evaluation_service.evaluate_model_comprehensive(
            model, X_val, y_val, X_test, y_test, df_viz,
            model_name=model_name, persist_results=persist_results
        )

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance from trained Random Forest pipeline.
        
        Returns:
            Array of feature importances or None if pipeline not trained
        """
        if not self.is_trained or self.model is None:
            return None
        return self.model.feature_importances_
