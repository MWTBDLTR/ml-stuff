# Machine Learning Learning Guide

A comprehensive guide for getting started with machine learning for complete beginners.

## 📖 What's Included

### 1. **Notebooks** (`notebooks/` folder)
Interactive Jupyter notebooks that teach ML concepts with executable code:

- **01_data_exploration.ipynb** - Start here! Learn data exploration basics
  - Loading datasets
  - Understanding data structure
  - Visualizing distributions
  - Correlation analysis

- **02_supervised_learning.ipynb** - Classification with scikit-learn
  - Data preparation and splitting
  - Logistic Regression
  - Decision Trees
  - Random Forests
  - Model evaluation and comparison

- **03_neural_networks.ipynb** - Introduction to deep learning
  - Neural network concepts
  - Building networks with TensorFlow/Keras
  - Training and evaluation
  - Activation functions explained
  - Architecture exploration

### 2. **Python Scripts** (`scripts/` folder)
Ready-to-run Python scripts implementing ML algorithms:

- **neural_network_simple.py** - Neural network from scratch
- **train_model.py** - Training various models with scikit-learn and TensorFlow

### 3. **Utilities** (`utils/` folder)
Helper functions and utilities:

- **data_loader.py** - Load and preprocess datasets
- **utils.py** - Visualization and evaluation tools

## 🚀 Getting Started

### Step 1: Set Up Your Environment
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Start Learning
```bash
# Launch Jupyter
jupyter notebook

# Open notebooks/ folder and start with 01_data_exploration.ipynb
```

### Step 3: Follow the Learning Path
1. **Data Exploration** - Understand your data first
2. **Supervised Learning** - Learn classical ML algorithms
3. **Neural Networks** - Dive into deep learning

## 💡 Learning Tips

1. **Read the code comments** - Every section has explanatory comments
2. **Run cells incrementally** - Execute one cell at a time to understand flow
3. **Experiment** - Modify parameters and see what happens
4. **Visualize** - Always plot your data and results
5. **Reference the utils** - Use helper functions for common tasks

## 📚 Key Concepts Covered

### Foundations
- [ ] Dataset structure and exploration
- [ ] Data preprocessing and normalization
- [ ] Train-test splitting
- [ ] Feature scaling

### Supervised Learning
- [ ] Linear Regression
- [ ] Logistic Regression
- [ ] Decision Trees
- [ ] Random Forests
- [ ] Model evaluation metrics

### Neural Networks & Deep Learning
- [ ] Perceptron basics
- [ ] Activation functions (ReLU, Sigmoid, Softmax)
- [ ] Forward propagation
- [ ] Backpropagation
- [ ] Loss functions
- [ ] Dropout and regularization

## 🔧 Common Workflows

### Load and Explore Data
```python
from utils.data_loader import load_iris_data, explore_dataset

X_train, X_test, y_train, y_test = load_iris_data()
explore_dataset(X_train, y_train)
```

### Train a Model
```python
from utils.data_loader import load_iris_data, scale_features
from scripts.train_model import train_scikit_model

X_train, X_test, y_train, y_test = load_iris_data()
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
model = train_scikit_model(X_train_scaled, y_train, 
                          X_test_scaled, y_test, 
                          'random_forest')
```

### Evaluate Model
```python
from utils.utils import plot_confusion_matrix, evaluate_classification

y_pred = model.predict(X_test_scaled)
plot_confusion_matrix(y_test, y_pred)
evaluate_classification(y_test, y_pred)
```

## 📊 Datasets Used

1. **Iris Dataset** - 150 samples, 4 features, 3 classes (flower classification)
2. **California Housing** - Real estate prediction (in scripts)
3. **MNIST Digits** - Handwritten digit recognition (optional)

## 🎓 Recommended Learning Order

1. Start with **01_data_exploration.ipynb**
   - Understand what data looks like
   - Learn visualization techniques

2. Move to **02_supervised_learning.ipynb**
   - See how different algorithms work
   - Compare their performance

3. Explore **03_neural_networks.ipynb**
   - Understand deep learning fundamentals
   - Build your first neural network

## 🌐 External Resources

- **Scikit-learn Docs**: https://scikit-learn.org/stable/
- **TensorFlow/Keras**: https://www.tensorflow.org/
- **3Blue1Brown Neural Networks**: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- **Andrew Ng's ML Course**: https://www.coursera.org/learn/machine-learning
- **Fast.ai**: https://www.fast.ai/

## ❓ Troubleshooting

### Import Errors
- Make sure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

### Jupyter Not Found
- Install Jupyter: `pip install jupyter`

### TensorFlow Issues
- On M1/M2 Mac: `pip install tensorflow-macos`

## 📝 Exercises

Try these exercises to solidify your learning:

1. **Data Exploration**
   - Load the California Housing dataset
   - Find the feature with highest correlation with price
   - Create visualizations for each feature

2. **Supervised Learning**
   - Train a model to predict iris species
   - Improve accuracy by tuning hyperparameters
   - Compare at least 3 different algorithms

3. **Neural Networks**
   - Build a network with different architectures
   - Experiment with different activation functions
   - Try different optimizers (Adam, SGD, RMSprop)

## 🎯 Next Steps After This Course

- Convolutional Neural Networks (CNNs) for images
- Recurrent Neural Networks (RNNs) for sequences
- Natural Language Processing (NLP)
- Unsupervised Learning (Clustering, Dimensionality Reduction)
- Reinforcement Learning
- Advanced architectures (Transformers, GANs)

## 📄 License

See LICENSE file for details.

---

**Happy Learning!** 🚀

Remember: The best way to learn ML is by doing. Experiment, break things, and learn from the results!
