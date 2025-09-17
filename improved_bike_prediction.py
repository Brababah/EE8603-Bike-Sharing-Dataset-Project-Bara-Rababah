#!/usr/bin/env python3
"""
Improved Bike Sharing Demand Prediction Pipeline

This module provides an improved implementation of the bike sharing demand prediction
system with better code organization, proper ML practices, and comprehensive evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, List
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import randint, uniform
import joblib


class BikeDataProcessor:
    """Handles data loading and preprocessing for bike sharing dataset."""
    
    def __init__(self):
        self.scaler = None
        self.encoder = None
        self.feature_names = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and perform initial data cleaning."""
        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded successfully: {data.shape}")
            return data
        except FileNotFoundError:
            print(f"Error: File {filepath} not found")
            return None
            
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create additional features and clean data."""
        data = data.copy()
        
        # Convert date column to datetime
        if 'dteday' in data.columns:
            data['dteday'] = pd.to_datetime(data['dteday'])
            
        # Create season names for better interpretability
        season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
        if 'season' in data.columns:
            data['season_name'] = data['season'].map(season_map)
            
        # Create day names
        day_names = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
                    4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
        if 'weekday' in data.columns:
            data['day_name'] = data['weekday'].map(day_names)
            
        return data
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable."""
        # Remove redundant and non-predictive features
        features_to_drop = [
            'cnt',  # target variable
            'instant',  # just an index
            'dteday',  # date already encoded in other features
            'atemp',  # highly correlated with temp
            'casual',  # part of target variable
            'registered',  # part of target variable
            'season_name',  # categorical version exists
            'day_name'  # categorical version exists
        ]
        
        # Only drop columns that exist in the dataset
        existing_features_to_drop = [col for col in features_to_drop if col in data.columns]
        
        X = data.drop(columns=existing_features_to_drop)
        y = data['cnt'] if 'cnt' in data.columns else None
        
        return X, y
        
    def get_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create preprocessing pipeline."""
        # Identify numeric and categorical features
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target-related features from numeric features if they exist
        target_related = ['casual', 'registered', 'cnt']
        numeric_features = [f for f in numeric_features if f not in target_related]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ],
            remainder='passthrough'
        )
        
        return preprocessor


class ModelEvaluator:
    """Comprehensive model evaluation and comparison."""
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results = {}
        
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name: str) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        # Cross-validation on training set
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=self.cv_folds, 
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        # Fit model and predict on test set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'cv_rmse_mean': -cv_scores.mean(),
            'cv_rmse_std': cv_scores.std(),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'test_r2': r2_score(y_test, y_pred),
            'test_mae': mean_absolute_error(y_test, y_pred),
        }
        
        # Calculate adjusted R²
        n = len(y_test)
        p = X_test.shape[1]
        metrics['test_adj_r2'] = 1 - (1 - metrics['test_r2']) * (n - 1) / (n - p - 1)
        
        self.results[model_name] = metrics
        return metrics
        
    def print_results(self, model_name: str):
        """Print formatted results for a model."""
        if model_name not in self.results:
            print(f"No results found for {model_name}")
            return
            
        metrics = self.results[model_name]
        print(f"\n{model_name} Results:")
        print(f"Cross-Validation RMSE: {metrics['cv_rmse_mean']:.2f} ± {metrics['cv_rmse_std']:.2f}")
        print(f"Test RMSE: {metrics['test_rmse']:.2f}")
        print(f"Test R²: {metrics['test_r2']:.4f}")
        print(f"Test Adjusted R²: {metrics['test_adj_r2']:.4f}")
        print(f"Test MAE: {metrics['test_mae']:.2f}")
        
    def compare_models(self) -> pd.DataFrame:
        """Create comparison dataframe of all evaluated models."""
        if not self.results:
            print("No models have been evaluated yet.")
            return pd.DataFrame()
            
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.sort_values('test_rmse')
        
        print("\n=== Model Comparison Summary ===")
        print(comparison_df.round(4).to_string())
        
        best_model = comparison_df.index[0]
        print(f"\nBest performing model: {best_model}")
        print(f"Test RMSE: {comparison_df.loc[best_model, 'test_rmse']:.2f}")
        print(f"Test R²: {comparison_df.loc[best_model, 'test_r2']:.4f}")
        
        return comparison_df
        
    def plot_comparison(self):
        """Create visualization comparing model performance."""
        if not self.results:
            print("No models have been evaluated yet.")
            return
            
        df = pd.DataFrame(self.results).T
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = ['test_rmse', 'test_r2', 'test_mae']
        titles = ['Test RMSE (Lower is Better)', 'Test R² (Higher is Better)', 'Test MAE (Lower is Better)']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            bars = axes[i].bar(df.index, df[metric])
            axes[i].set_title(title)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')
                           
        plt.tight_layout()
        plt.show()


class BikeDemandPredictor:
    """Main class for bike demand prediction pipeline."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.processor = BikeDataProcessor()
        self.evaluator = ModelEvaluator(random_state=random_state)
        self.models = {}
        self.best_model = None
        
    def load_and_prepare_data(self, filepath: str, test_size: float = 0.2) -> Tuple:
        """Load and prepare data for modeling."""
        # Load data
        data = self.processor.load_data(filepath)
        if data is None:
            return None, None, None, None
            
        # Create features
        data = self.processor.create_features(data)
        
        # Prepare features and target
        X, y = self.processor.prepare_features(data)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
        
    def create_models(self) -> Dict[str, Pipeline]:
        """Create model pipelines with preprocessing."""
        # Get sample data to create preprocessor
        sample_data = pd.read_csv('day.csv') if 'day.csv' else None
        if sample_data is None:
            raise ValueError("Cannot create models without sample data")
            
        data_processed = self.processor.create_features(sample_data)
        X_sample, _ = self.processor.prepare_features(data_processed)
        preprocessor = self.processor.get_preprocessor(X_sample)
        
        models = {
            'Linear Regression': Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ]),
            
            'Ridge Regression': Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', Ridge(random_state=self.random_state))
            ]),
            
            'Decision Tree': Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', DecisionTreeRegressor(random_state=self.random_state))
            ]),
            
            'Random Forest': Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(random_state=self.random_state))
            ])
        }
        
        self.models = models
        return models
        
    def optimize_hyperparameters(self, X_train, y_train) -> Dict[str, Any]:
        """Optimize hyperparameters for tree-based models."""
        optimized_models = {}
        
        # Decision Tree optimization
        dt_params = {
            'regressor__max_depth': range(3, 21),
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4]
        }
        
        dt_grid = GridSearchCV(
            self.models['Decision Tree'],
            dt_params,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        dt_grid.fit(X_train, y_train)
        optimized_models['Decision Tree (Optimized)'] = dt_grid.best_estimator_
        
        # Random Forest optimization with RandomizedSearchCV for efficiency
        rf_params = {
            'regressor__n_estimators': randint(50, 300),
            'regressor__max_depth': randint(5, 20),
            'regressor__min_samples_split': randint(2, 20),
            'regressor__min_samples_leaf': randint(1, 10)
        }
        
        rf_random = RandomizedSearchCV(
            self.models['Random Forest'],
            rf_params,
            n_iter=50,
            cv=5,
            scoring='neg_root_mean_squared_error',
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_random.fit(X_train, y_train)
        optimized_models['Random Forest (Optimized)'] = rf_random.best_estimator_
        
        # Ridge regression optimization
        ridge_params = {
            'regressor__alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
        }
        
        ridge_grid = GridSearchCV(
            self.models['Ridge Regression'],
            ridge_params,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        ridge_grid.fit(X_train, y_train)
        optimized_models['Ridge Regression (Optimized)'] = ridge_grid.best_estimator_
        
        return optimized_models
        
    def run_comparison(self, filepath: str) -> pd.DataFrame:
        """Run complete model comparison pipeline."""
        print("=== Bike Demand Prediction Model Comparison ===\n")
        
        # Load and prepare data
        X_train, X_test, y_train, y_test = self.load_and_prepare_data(filepath)
        if X_train is None:
            return pd.DataFrame()
            
        # Create base models
        self.create_models()
        
        # Evaluate base models
        print("\n--- Evaluating Base Models ---")
        for name, model in self.models.items():
            metrics = self.evaluator.evaluate_model(
                model, X_train, X_test, y_train, y_test, name
            )
            self.evaluator.print_results(name)
            
        # Optimize hyperparameters
        print("\n--- Optimizing Hyperparameters ---")
        optimized_models = self.optimize_hyperparameters(X_train, y_train)
        
        # Evaluate optimized models
        print("\n--- Evaluating Optimized Models ---")
        for name, model in optimized_models.items():
            metrics = self.evaluator.evaluate_model(
                model, X_train, X_test, y_train, y_test, name
            )
            self.evaluator.print_results(name)
            
        # Compare all models
        comparison_df = self.evaluator.compare_models()
        
        # Create visualizations
        self.evaluator.plot_comparison()
        
        # Save best model
        best_model_name = comparison_df.index[0]
        if 'Optimized' in best_model_name:
            self.best_model = optimized_models[best_model_name]
        else:
            self.best_model = self.models[best_model_name]
            
        return comparison_df
        
    def save_best_model(self, filepath: str = 'best_bike_demand_model.pkl'):
        """Save the best performing model."""
        if self.best_model is None:
            print("No best model found. Run comparison first.")
            return
            
        joblib.dump(self.best_model, filepath)
        print(f"Best model saved to {filepath}")


def main():
    """Main execution function."""
    # Initialize predictor
    predictor = BikeDemandPredictor(random_state=42)
    
    # Run comparison
    results = predictor.run_comparison('day.csv')
    
    # Save best model
    predictor.save_best_model()
    
    return results


if __name__ == "__main__":
    results = main()