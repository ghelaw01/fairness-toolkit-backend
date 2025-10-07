"""
Data Processing Module for COMPAS Dataset

This module provides data processing utilities specifically designed for
the COMPAS recidivism dataset and similar criminal justice datasets.

Author: Manus AI
License: MIT
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings

class COMPASDataProcessor:
    """
    Data processor specifically designed for the COMPAS recidivism dataset.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the COMPAS data processor.
        
        Args:
            data_path: Path to the COMPAS dataset CSV file
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.feature_encoders = {}
        self.scaler = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the COMPAS dataset.
        
        Returns:
            Raw COMPAS dataset
        """
        try:
            self.raw_data = pd.read_csv(self.data_path)
            print(f"Loaded COMPAS dataset with {len(self.raw_data)} records and {len(self.raw_data.columns)} columns")
            return self.raw_data
        except Exception as e:
            raise ValueError(f"Error loading data from {self.data_path}: {e}")
    
    def explore_data(self) -> Dict[str, Any]:
        """
        Perform initial data exploration.
        
        Returns:
            Dict containing data exploration results
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        exploration = {
            'shape': self.raw_data.shape,
            'columns': list(self.raw_data.columns),
            'missing_values': self.raw_data.isnull().sum().to_dict(),
            'data_types': self.raw_data.dtypes.to_dict(),
            'unique_values': {},
            'summary_stats': {}
        }
        
        # Get unique values for categorical columns
        categorical_columns = self.raw_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            unique_vals = self.raw_data[col].unique()
            exploration['unique_values'][col] = list(unique_vals[:10])  # First 10 unique values
        
        # Summary statistics for numerical columns
        numerical_columns = self.raw_data.select_dtypes(include=[np.number]).columns
        exploration['summary_stats'] = self.raw_data[numerical_columns].describe().to_dict()
        
        return exploration
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the COMPAS dataset by handling missing values and inconsistencies.
        
        Returns:
            Cleaned dataset
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        data = self.raw_data.copy()
        
        # Remove rows with missing critical information
        critical_columns = ['age', 'sex', 'race', 'two_year_recid']
        data = data.dropna(subset=critical_columns)
        
        # Clean age data (remove unrealistic ages)
        data = data[(data['age'] >= 18) & (data['age'] <= 100)]
        
        # Standardize race categories
        race_mapping = {
            'African-American': 'African-American',
            'Caucasian': 'Caucasian',
            'Hispanic': 'Hispanic',
            'Other': 'Other',
            'Asian': 'Other',
            'Native American': 'Other'
        }
        data['race'] = data['race'].map(race_mapping).fillna('Other')
        
        # Clean sex categories
        data['sex'] = data['sex'].map({'Male': 'Male', 'Female': 'Female'})
        data = data.dropna(subset=['sex'])
        
        # Handle missing values in other columns
        # Fill missing priors_count with 0
        data['priors_count'] = data['priors_count'].fillna(0)
        
        # Fill missing decile_score with median
        data['decile_score'] = data['decile_score'].fillna(data['decile_score'].median())
        
        # Create age categories
        age_bins = [0, 25, 45, 100]
        age_labels = ['Less than 25', '25 - 45', 'Greater than 45']
        data['age_cat_cleaned'] = pd.cut(data['age'], bins=age_bins, labels=age_labels, include_lowest=True)
        data['age_cat_cleaned'] = data['age_cat_cleaned'].astype(str)
        
        # Remove duplicates
        data = data.drop_duplicates()
        
        print(f"Data cleaned: {len(data)} records remaining from {len(self.raw_data)} original records")
        
        self.processed_data = data
        return data
    
    def create_features(self) -> pd.DataFrame:
        """
        Create additional features for analysis.
        
        Returns:
            Dataset with additional features
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Call clean_data() first.")
        
        data = self.processed_data.copy()
        
        # Create binary features
        data['is_male'] = (data['sex'] == 'Male').astype(int)
        data['is_african_american'] = (data['race'] == 'African-American').astype(int)
        data['is_caucasian'] = (data['race'] == 'Caucasian').astype(int)
        
        # Create risk level categories
        risk_bins = [0, 4, 7, 10]
        risk_labels = ['Low', 'Medium', 'High']
        data['risk_level'] = pd.cut(data['decile_score'], bins=risk_bins, labels=risk_labels, include_lowest=True)
        data['risk_level'] = data['risk_level'].astype(str)
        
        # Create prior offense categories
        priors_bins = [-1, 0, 2, 5, float('inf')]
        priors_labels = ['No_Priors', 'Few_Priors', 'Some_Priors', 'Many_Priors']
        data['priors_category'] = pd.cut(data['priors_count'], bins=priors_bins, labels=priors_labels, include_lowest=True)
        data['priors_category'] = data['priors_category'].astype(str)
        
        # Create juvenile offense indicator
        juv_columns = ['juv_fel_count', 'juv_misd_count', 'juv_other_count']
        for col in juv_columns:
            if col in data.columns:
                data[col] = data[col].fillna(0)
        
        data['has_juvenile_record'] = ((data.get('juv_fel_count', 0) + 
                                      data.get('juv_misd_count', 0) + 
                                      data.get('juv_other_count', 0)) > 0).astype(int)
        
        # Create charge degree features
        if 'c_charge_degree' in data.columns:
            data['is_felony'] = (data['c_charge_degree'] == 'F').astype(int)
            data['is_misdemeanor'] = (data['c_charge_degree'] == 'M').astype(int)
        
        self.processed_data = data
        return data
    
    def prepare_for_modeling(self, target_column: str = 'two_year_recid',
                           sensitive_attributes: List[str] = ['race', 'sex'],
                           test_size: float = 0.2, 
                           random_state: int = 42) -> Dict[str, Any]:
        """
        Prepare data for machine learning modeling.
        
        Args:
            target_column: Name of the target variable
            sensitive_attributes: List of sensitive attributes to preserve
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            
        Returns:
            Dict containing train/test splits and metadata
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Call create_features() first.")
        
        data = self.processed_data.copy()
        
        # Select features for modeling
        feature_columns = [
            'age', 'priors_count', 'decile_score',
            'is_male', 'is_african_american', 'is_caucasian',
            'has_juvenile_record'
        ]
        
        # Add charge degree features if available
        if 'is_felony' in data.columns:
            feature_columns.extend(['is_felony', 'is_misdemeanor'])
        
        # Filter to only include rows with all required columns
        required_columns = feature_columns + [target_column] + sensitive_attributes
        available_columns = [col for col in required_columns if col in data.columns]
        
        if len(available_columns) < len(required_columns):
            missing_cols = set(required_columns) - set(available_columns)
            warnings.warn(f"Missing columns: {missing_cols}")
        
        # Use only available feature columns
        available_features = [col for col in feature_columns if col in data.columns]
        
        # Create final dataset
        final_data = data[available_features + [target_column] + sensitive_attributes].dropna()
        
        # Separate features and target
        X = final_data[available_features]
        y = final_data[target_column]
        
        # Preserve sensitive attributes
        sensitive_data = final_data[sensitive_attributes]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Split sensitive attributes accordingly
        sensitive_train = sensitive_data.iloc[X_train.index.intersection(sensitive_data.index)]
        sensitive_test = sensitive_data.iloc[X_test.index.intersection(sensitive_data.index)]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train.values,
            'y_test': y_test.values,
            'X_train_raw': X_train,
            'X_test_raw': X_test,
            'sensitive_train': sensitive_train,
            'sensitive_test': sensitive_test,
            'feature_names': available_features,
            'target_name': target_column,
            'sensitive_attributes': sensitive_attributes,
            'scaler': self.scaler
        }
    
    def get_data_summary(self) -> pd.DataFrame:
        """
        Get a summary of the processed data.
        
        Returns:
            DataFrame with data summary
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Call create_features() first.")
        
        summary_data = []
        
        # Overall statistics
        summary_data.append({
            'Metric': 'Total Records',
            'Value': len(self.processed_data),
            'Description': 'Total number of records in processed dataset'
        })
        
        # Recidivism rate
        if 'two_year_recid' in self.processed_data.columns:
            recid_rate = self.processed_data['two_year_recid'].mean()
            summary_data.append({
                'Metric': 'Overall Recidivism Rate',
                'Value': f"{recid_rate:.3f}",
                'Description': 'Proportion of individuals who reoffended within two years'
            })
        
        # Demographics
        if 'race' in self.processed_data.columns:
            race_dist = self.processed_data['race'].value_counts(normalize=True)
            for race, prop in race_dist.items():
                summary_data.append({
                    'Metric': f'Proportion {race}',
                    'Value': f"{prop:.3f}",
                    'Description': f'Proportion of {race} individuals in dataset'
                })
        
        if 'sex' in self.processed_data.columns:
            sex_dist = self.processed_data['sex'].value_counts(normalize=True)
            for sex, prop in sex_dist.items():
                summary_data.append({
                    'Metric': f'Proportion {sex}',
                    'Value': f"{prop:.3f}",
                    'Description': f'Proportion of {sex} individuals in dataset'
                })
        
        # Age statistics
        if 'age' in self.processed_data.columns:
            summary_data.extend([
                {
                    'Metric': 'Mean Age',
                    'Value': f"{self.processed_data['age'].mean():.1f}",
                    'Description': 'Average age of individuals in dataset'
                },
                {
                    'Metric': 'Age Range',
                    'Value': f"{self.processed_data['age'].min()}-{self.processed_data['age'].max()}",
                    'Description': 'Age range in dataset'
                }
            ])
        
        # Risk score statistics
        if 'decile_score' in self.processed_data.columns:
            summary_data.extend([
                {
                    'Metric': 'Mean Risk Score',
                    'Value': f"{self.processed_data['decile_score'].mean():.1f}",
                    'Description': 'Average COMPAS risk score (1-10 scale)'
                },
                {
                    'Metric': 'High Risk Rate',
                    'Value': f"{(self.processed_data['decile_score'] >= 7).mean():.3f}",
                    'Description': 'Proportion classified as high risk (score ≥ 7)'
                }
            ])
        
        return pd.DataFrame(summary_data)
    
    def export_processed_data(self, output_path: str) -> None:
        """
        Export processed data to CSV.
        
        Args:
            output_path: Path to save the processed data
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Call create_features() first.")
        
        self.processed_data.to_csv(output_path, index=False)
        print(f"Processed data exported to {output_path}")


def load_and_process_compas(data_path: str) -> Tuple[COMPASDataProcessor, Dict[str, Any]]:
    """
    Convenience function to load and process COMPAS data.
    
    Args:
        data_path: Path to the COMPAS dataset
        
    Returns:
        Tuple of (processor, modeling_data)
    """
    processor = COMPASDataProcessor(data_path)
    
    # Load and process data
    processor.load_data()
    processor.clean_data()
    processor.create_features()
    
    # Prepare for modeling
    modeling_data = processor.prepare_for_modeling()
    
    return processor, modeling_data

