# Machine Learning Basics: An Introduction Guide

A beginner-friendly project for learning machine learning fundamentals, including supervised learning, neural networks, and deep learning with practical examples.

## 📚 Project Structure

```
ml-stuff/
├── notebooks/          # Interactive Jupyter notebooks for learning
├── scripts/            # Python scripts with implementations
├── data/               # Datasets and data handling
├── utils/              # Utility functions and helpers
├── models/             # Trained model storage
├── requirements.txt    # Python dependencies
└── README.md
```

## 🎯 Learning Path

This project covers ML concepts in a structured progression:

1. **Foundations** - Understanding data and basic concepts
2. **Supervised Learning** - Regression and classification with scikit-learn
3. **Introduction to Neural Networks** - Basic concepts and architecture
4. **Deep Learning with TensorFlow** - Building and training neural networks
5. **Practical Projects** - Real-world applications and hands-on practice

## 📖 Topics Covered

### Beginner Level
- Loading and exploring datasets
- Data preprocessing and normalization
- Train-test splitting
- Feature scaling

### Supervised Learning
- Linear Regression
- Logistic Regression
- Decision Trees and Random Forests
- Model evaluation metrics (accuracy, precision, recall)

### Neural Networks & Deep Learning
- Perceptron and neural network basics
- Activation functions (ReLU, Sigmoid, Softmax)
- Loss functions and optimization
- Building networks with TensorFlow/Keras
- Training and validation
- Regularization techniques (Dropout, L1/L2)

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-stuff
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Jupyter Notebooks

```bash
jupyter notebook
```

Navigate to the `notebooks/` folder and open the notebooks in order.

## 📚 Notebooks

- `01_data_exploration.ipynb` - Loading and visualizing data
- `02_supervised_learning_scikit.ipynb` - ML with scikit-learn
- `03_neural_network_basics.ipynb` - Introduction to neural networks
- `04_deep_learning_tensorflow.ipynb` - Building networks with TensorFlow
- `05_practical_project.ipynb` - End-to-end ML project

## 🐍 Scripts

- `data_loader.py` - Functions for loading and preprocessing data
- `train_model.py` - Training models and evaluation
- `neural_network_example.py` - Simple neural network implementation
- `utils.py` - Helper functions

## 📊 Datasets

Example datasets are included or can be downloaded:
- **Iris Dataset** - Simple classification example
- **California Housing** - Regression example
- **MNIST** - Handwritten digits (deep learning)

## 🔧 Tools & Libraries

- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Scikit-learn** - Classical ML algorithms
- **TensorFlow/Keras** - Deep learning framework
- **Matplotlib/Seaborn** - Data visualization
- **Jupyter** - Interactive notebooks

## 💡 Tips for Learning

1. **Follow the progression** - Start with foundations before jumping to deep learning
2. **Experiment** - Modify code and see what happens
3. **Visualize** - Always plot your data and results
4. **Document** - Write notes about what you learn
5. **Practice** - Try problems on your own before looking at solutions

## 📝 Common Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Run a Python script
python scripts/train_model.py

# Start Jupyter notebook server
jupyter notebook

# Install additional packages
pip install <package-name>

# Deactivate virtual environment
deactivate
```

## 🎓 Resources for Further Learning

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow/Keras Guides](https://www.tensorflow.org/guide)
- [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Fast.ai Courses](https://www.fast.ai/)

## ⚠️ Notes

- GPU acceleration recommended for TensorFlow on large datasets
- All notebooks assume basic Python knowledge
- Run notebooks sequentially for best learning experience

## 📄 License

See LICENSE file for details.

---

Happy Learning! 🎉
