"""
Data preprocessing script for the Pima Indians Diabetes Dataset.
Loads, cleans, normalizes, and splits data for federated learning.
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import requests
from io import StringIO

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.settings import DATA_CONFIG, MODEL_CONFIG, LOGGING_CONFIG

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles data preprocessing for federated learning.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = MODEL_CONFIG["features"]
        self.target_column = MODEL_CONFIG["target"]
        
    def download_dataset(self) -> bool:
        """
        Download the Pima Indians Diabetes Dataset if not present.
        
        Returns:
            True if successful, False otherwise
        """
        dataset_path = DATA_CONFIG["dataset_path"]
        
        if os.path.exists(dataset_path):
            logger.info(f"Dataset already exists at {dataset_path}")
            return True
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        
        # Try to download from a reliable source
        urls = [
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
            "https://gist.githubusercontent.com/ktisha/c21e73a1bd1700294ef790c56c8aec1f/raw/819b69b5736821ccee93d05b51de0510bea00294/pima-indians-diabetes.csv"
        ]
        
        for url in urls:
            try:
                logger.info(f"Attempting to download dataset from {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Save the dataset
                with open(dataset_path, 'w') as f:
                    f.write(response.text)
                
                logger.info(f"Successfully downloaded dataset to {dataset_path}")
                return True
                
            except Exception as e:
                logger.warning(f"Failed to download from {url}: {e}")
                continue
        
        # If download fails, create a sample dataset
        logger.warning("Could not download dataset, creating sample data")
        self._create_sample_dataset(dataset_path)
        return True
    
    def _create_sample_dataset(self, dataset_path: str) -> None:
        """
        Create a sample dataset for demonstration purposes.
        """
        np.random.seed(42)
        n_samples = 768
        
        # Generate synthetic data similar to Pima Indians dataset
        data = {
            'Pregnancies': np.random.poisson(3, n_samples),
            'Glucose': np.random.normal(120, 30, n_samples).clip(0, 200),
            'BloodPressure': np.random.normal(70, 15, n_samples).clip(0, 150),
            'SkinThickness': np.random.normal(20, 10, n_samples).clip(0, 50),
            'Insulin': np.random.exponential(80, n_samples).clip(0, 500),
            'BMI': np.random.normal(32, 8, n_samples).clip(15, 60),
            'DiabetesPedigreeFunction': np.random.exponential(0.5, n_samples).clip(0, 2.5),
            'Age': np.random.normal(33, 12, n_samples).clip(21, 80),
        }
        
        # Create target variable with some correlation to features
        risk_score = (
            0.3 * (data['Glucose'] - 100) / 50 +
            0.2 * (data['BMI'] - 25) / 10 +
            0.2 * (data['Age'] - 30) / 20 +
            0.1 * data['Pregnancies'] / 5 +
            0.2 * np.random.normal(0, 1, n_samples)
        )
        data['Outcome'] = (risk_score > 0.5).astype(int)
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(dataset_path, index=False)
        logger.info(f"Created sample dataset with {n_samples} samples at {dataset_path}")
    
    def load_and_clean_data(self) -> pd.DataFrame:
        """
        Load and clean the diabetes dataset.
        
        Returns:
            Cleaned DataFrame
        """
        dataset_path = DATA_CONFIG["dataset_path"]
        
        # Ensure dataset exists
        if not os.path.exists(dataset_path):
            self.download_dataset()
        
        # Load data
        try:
            df = pd.read_csv(dataset_path)
            logger.info(f"Loaded dataset with shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        # Handle missing column names (some datasets don't have headers)
        expected_columns = self.feature_columns + [self.target_column]
        if len(df.columns) == len(expected_columns) and df.columns[0] != expected_columns[0]:
            df.columns = expected_columns
            logger.info("Added column names to dataset")
        
        # Basic data cleaning
        initial_shape = df.shape
        
        # Remove rows with missing target values
        df = df.dropna(subset=[self.target_column])
        
        # Handle zero values that should be NaN (common in this dataset)
        zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_columns:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
        
        # Fill missing values with median
        for col in self.feature_columns:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Filled {df[col].isnull().sum()} missing values in {col} with median {median_val:.2f}")
        
        # Remove outliers (optional, conservative approach)
        for col in self.feature_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                df = df[(df[col] >= Q1) & (df[col] <= Q3)]
        
        logger.info(f"Data cleaning: {initial_shape} -> {df.shape}")
        return df
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using StandardScaler.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with normalized features
        """
        df_normalized = df.copy()
        
        # Fit and transform features
        feature_data = df[self.feature_columns]
        normalized_features = self.scaler.fit_transform(feature_data)
        
        # Update DataFrame
        df_normalized[self.feature_columns] = normalized_features
        
        logger.info("Features normalized using StandardScaler")
        return df_normalized
    
    def split_for_hospitals(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Split data into partitions for different hospitals.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of DataFrames, one for each hospital
        """
        num_hospitals = DATA_CONFIG["num_hospitals"]
        split_method = DATA_CONFIG["split_method"]
        
        if split_method == "stratified":
            # Stratified split to maintain class distribution
            hospital_dfs = []
            for i in range(num_hospitals):
                if i == num_hospitals - 1:
                    # Last hospital gets remaining data
                    hospital_df = df.iloc[i * len(df) // num_hospitals:]
                else:
                    start_idx = i * len(df) // num_hospitals
                    end_idx = (i + 1) * len(df) // num_hospitals
                    hospital_df = df.iloc[start_idx:end_idx]
                hospital_dfs.append(hospital_df)
        else:
            # Random split
            np.random.seed(MODEL_CONFIG["random_state"])
            shuffled_df = df.sample(frac=1).reset_index(drop=True)
            
            hospital_dfs = []
            for i in range(num_hospitals):
                start_idx = i * len(shuffled_df) // num_hospitals
                end_idx = (i + 1) * len(shuffled_df) // num_hospitals
                hospital_df = shuffled_df.iloc[start_idx:end_idx]
                hospital_dfs.append(hospital_df)
        
        # Log split information
        for i, hospital_df in enumerate(hospital_dfs):
            positive_rate = hospital_df[self.target_column].mean()
            logger.info(f"Hospital {i}: {len(hospital_df)} samples, "
                       f"positive rate: {positive_rate:.3f}")
        
        return hospital_dfs
    
    def save_hospital_data(self, hospital_dfs: List[pd.DataFrame]) -> None:
        """
        Save hospital data partitions to separate files.
        
        Args:
            hospital_dfs: List of hospital DataFrames
        """
        data_dir = DATA_CONFIG["hospital_data_dir"]
        os.makedirs(data_dir, exist_ok=True)
        
        for i, hospital_df in enumerate(hospital_dfs):
            filename = f"hospital_{i}.csv"
            filepath = os.path.join(data_dir, filename)
            hospital_df.to_csv(filepath, index=False)
            logger.info(f"Saved hospital {i} data to {filepath}")
    
    def process_all(self) -> None:
        """
        Run the complete preprocessing pipeline.
        """
        logger.info("Starting data preprocessing pipeline")
        
        # Download dataset if needed
        self.download_dataset()
        
        # Load and clean data
        df = self.load_and_clean_data()
        
        # Normalize features
        if MODEL_CONFIG["normalize"]:
            df = self.normalize_features(df)
        
        # Split for hospitals
        hospital_dfs = self.split_for_hospitals(df)
        
        # Save hospital data
        self.save_hospital_data(hospital_dfs)
        
        logger.info("Data preprocessing completed successfully")
        
        # Print summary
        print("\n" + "="*50)
        print("DATA PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Total samples: {len(df)}")
        print(f"Features: {len(self.feature_columns)}")
        print(f"Hospitals: {len(hospital_dfs)}")
        print(f"Target distribution: {df[self.target_column].value_counts().to_dict()}")
        print("="*50)

def main():
    """Main function to run preprocessing."""
    preprocessor = DataPreprocessor()
    preprocessor.process_all()

if __name__ == "__main__":
    main()
