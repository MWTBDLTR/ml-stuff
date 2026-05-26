"""
Training script for machine learning models using scikit-learn and TensorFlow.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def train_scikit_model(X_train, y_train, X_test, y_test, model_type='random_forest'):
    """
    Train a scikit-learn model.
    
    Parameters:
    -----------
    X_train, y_train : Training data and labels
    X_test, y_test : Test data and labels
    model_type : str
        Type of model ('random_forest', 'logistic_regression', 'svm')
    
    Returns:
    --------
    model : Trained model
    """
    print(f"Training {model_type}...")
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == 'svm':
        model = SVC(kernel='rbf', random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model


def build_simple_neural_network(input_dim, num_classes, hidden_units=64):
    """
    Build a simple neural network with Keras.
    
    Parameters:
    -----------
    input_dim : int
        Input dimension
    num_classes : int
        Number of output classes
    hidden_units : int
        Number of hidden units
    
    Returns:
    --------
    model : Keras sequential model
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden_units, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(hidden_units // 2, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_neural_network(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """
    Train a Keras neural network.
    
    Parameters:
    -----------
    model : Keras model
    X_train, y_train : Training data
    X_test, y_test : Test data
    epochs : int
    batch_size : int
    
    Returns:
    --------
    history : Training history
    """
    print("Training neural network...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    return history


def build_cnn_for_images(num_classes, input_shape=(28, 28, 1)):
    """
    Build a Convolutional Neural Network for image classification.
    
    Parameters:
    -----------
    num_classes : int
        Number of output classes
    input_shape : tuple
        Input shape (height, width, channels)
    
    Returns:
    --------
    model : Keras CNN model
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == "__main__":
    # Example: Load and train on Iris dataset
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train scikit-learn model
    rf_model = train_scikit_model(X_train, y_train, X_test, y_test, 'random_forest')
    
    # Train neural network
    nn_model = build_simple_neural_network(input_dim=4, num_classes=3)
    history = train_neural_network(nn_model, X_train, y_train, X_test, y_test, epochs=50)
