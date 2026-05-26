"""
Simple neural network implementation from scratch (educational purpose).
This shows the basics of how neural networks work without using high-level frameworks.
"""

import numpy as np


class SimpleNeuralNetwork:
    """
    A simple feedforward neural network with one hidden layer.
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize the neural network.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_size : int
            Number of neurons in hidden layer
        output_size : int
            Number of output neurons
        learning_rate : float
            Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU."""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation for output layer."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        X : array-like of shape (batch_size, input_size)
            Input data
        
        Returns:
        --------
        tuple : (output, hidden_layer_output)
        """
        # Hidden layer with ReLU
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Output layer with softmax
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        
        return self.a2, self.a1
    
    def compute_loss(self, predictions, targets):
        """
        Compute cross-entropy loss.
        
        Parameters:
        -----------
        predictions : array-like
            Model predictions (probabilities)
        targets : array-like
            True labels (one-hot encoded)
        
        Returns:
        --------
        float : Average loss
        """
        m = predictions.shape[0]
        log_likelihood = -np.log(predictions[range(m), np.argmax(targets, axis=1)])
        return np.sum(log_likelihood) / m
    
    def backward(self, X, targets, learning_rate):
        """
        Backward pass (backpropagation).
        
        Parameters:
        -----------
        X : array-like
            Input data
        targets : array-like
            True labels (one-hot encoded)
        learning_rate : float
            Learning rate
        """
        m = X.shape[0]
        
        # Output layer gradient
        dz2 = self.a2 - targets
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer gradient
        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs=100, batch_size=32, verbose=True):
        """
        Train the neural network.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Training labels
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for mini-batch gradient descent
        verbose : bool
            Print loss every 10 epochs
        """
        # Convert labels to one-hot encoding
        n_classes = len(np.unique(y))
        y_one_hot = np.eye(n_classes)[y]
        
        n_samples = X.shape[0]
        losses = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_one_hot[indices]
            
            # Mini-batch gradient descent
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward and backward pass
                predictions, _ = self.forward(X_batch)
                self.backward(X_batch, y_batch, self.learning_rate)
            
            # Compute loss on entire training set
            predictions, _ = self.forward(X)
            loss = self.compute_loss(predictions, y_one_hot)
            losses.append(loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data
        
        Returns:
        --------
        array : Predicted class labels
        """
        predictions, _ = self.forward(X)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Parameters:
        -----------
        X : array-like
            Input data
        
        Returns:
        --------
        array : Probability predictions
        """
        predictions, _ = self.forward(X)
        return predictions


# Example usage
if __name__ == "__main__":
    # Create synthetic data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    
    # Generate data
    X, y = make_classification(n_samples=200, n_features=10, n_classes=3, 
                               n_informative=10, n_redundant=0, random_state=42)
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train network
    nn = SimpleNeuralNetwork(input_size=10, hidden_size=16, output_size=3, learning_rate=0.01)
    losses = nn.train(X_train, y_train, epochs=100, batch_size=16, verbose=True)
    
    # Make predictions
    y_pred = nn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
