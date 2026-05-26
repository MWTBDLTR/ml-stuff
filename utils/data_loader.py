"""
Data loading and preprocessing utilities.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_digits, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_iris_data(test_size=0.2, random_state=42):
    """
    Load and split the Iris dataset.
    
    Parameters:
    -----------
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test)
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def load_digits_data(test_size=0.2, random_state=42):
    """
    Load and split the handwritten digits dataset.
    
    Parameters:
    -----------
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test)
    """
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def load_housing_data(test_size=0.2, random_state=42):
    """
    Load and split the California housing dataset.
    
    Parameters:
    -----------
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test)
    """
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def explore_dataset(X, y, feature_names=None):
    """
    Print basic information about a dataset.
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Labels/targets
    feature_names : list, optional
        Names of features
    """
    print(f"Dataset shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Classes: {np.unique(y)}")
    
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    
    print(f"\nFeature statistics:")
    df = pd.DataFrame(X, columns=feature_names)
    print(df.describe())


def scale_features(X_train, X_test, method='standard'):
    """
    Scale features using standardization or normalization.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    method : str
        'standard' for StandardScaler, 'minmax' for MinMaxScaler
    
    Returns:
    --------
    tuple : (X_train_scaled, X_test_scaled)
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled


def create_synthetic_data(n_samples=100, n_features=2, n_classes=2, random_state=42):
    """
    Create synthetic classification dataset.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    n_classes : int
        Number of classes
    random_state : int
        Random seed
    
    Returns:
    --------
    tuple : (X, y)
    """
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_classes=n_classes,
        random_state=random_state
    )
    
    return X, y
