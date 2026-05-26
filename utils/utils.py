"""
Utility functions for machine learning workflows.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plot a confusion matrix for classification results.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    title : str
        Title for the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def evaluate_classification(y_true, y_pred):
    """
    Evaluate classification model with multiple metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    
    Returns:
    --------
    dict : Dictionary with evaluation metrics
    """
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)
    
    return {'accuracy': acc}


def evaluate_regression(y_true, y_pred):
    """
    Evaluate regression model with multiple metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    dict : Dictionary with evaluation metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def plot_training_history(history, title="Training History"):
    """
    Plot training and validation loss/accuracy for neural networks.
    
    Parameters:
    -----------
    history : keras history object
        History from model.fit()
    title : str
        Title for the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if 'accuracy' in history.history:
        axes[1].plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def normalize_features(X_train, X_test):
    """
    Normalize features to [0, 1] range.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    
    Returns:
    --------
    tuple : (normalized_X_train, normalized_X_test)
    """
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    
    X_train_norm = (X_train - X_min) / (X_max - X_min + 1e-8)
    X_test_norm = (X_test - X_min) / (X_max - X_min + 1e-8)
    
    return X_train_norm, X_test_norm


def standardize_features(X_train, X_test):
    """
    Standardize features to mean=0, std=1.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    
    Returns:
    --------
    tuple : (standardized_X_train, standardized_X_test)
    """
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    
    X_train_std = (X_train - X_mean) / (X_std + 1e-8)
    X_test_std = (X_test - X_mean) / (X_std + 1e-8)
    
    return X_train_std, X_test_std


def plot_feature_importance(importance, feature_names, title="Feature Importance"):
    """
    Plot feature importance from tree-based models.
    
    Parameters:
    -----------
    importance : array-like
        Feature importance values
    feature_names : list
        Names of features
    title : str
        Title for the plot
    """
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=14, fontweight='bold')
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()
