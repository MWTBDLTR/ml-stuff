# Project Structure

```
ml-stuff/
│
├── README.md                          # Project overview and setup instructions
├── LEARNING_GUIDE.md                  # Detailed learning guide and tips
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore patterns
│
├── notebooks/                         # Interactive Jupyter Notebooks
│   ├── 01_data_exploration.ipynb      # Data exploration and visualization
│   ├── 02_supervised_learning.ipynb   # Classification with scikit-learn
│   └── 03_neural_networks.ipynb       # Deep learning with TensorFlow
│
├── scripts/                           # Python scripts
│   ├── neural_network_simple.py       # Simple neural network from scratch
│   ├── train_model.py                 # Model training utilities
│   └── __init__.py                    # Python package init
│
├── utils/                             # Utility modules
│   ├── data_loader.py                 # Data loading and preparation
│   ├── utils.py                       # Visualization and evaluation
│   └── __init__.py                    # Python package init
│
├── data/                              # Data storage directory
│   ├── raw/                           # Original datasets
│   └── processed/                     # Processed datasets
│
└── models/                            # Trained model storage
    └── trained/                       # Saved trained models
```

## File Descriptions

### Core Files

- **README.md** - Project overview, installation, and quick start guide
- **LEARNING_GUIDE.md** - Detailed learning resources and workflows
- **requirements.txt** - All Python packages needed (numpy, pandas, scikit-learn, TensorFlow, etc.)

### Notebooks (Start Here!)

1. **01_data_exploration.ipynb**
   - Load Iris dataset
   - Explore structure and statistics
   - Visualize distributions
   - Correlation analysis
   - Best for: Understanding data

2. **02_supervised_learning.ipynb**
   - Data preparation (split, scale)
   - Train multiple algorithms
   - Compare performance
   - Confusion matrices
   - Best for: Learning classical ML

3. **03_neural_networks.ipynb**
   - Neural network concepts
   - Build with TensorFlow/Keras
   - Training visualization
   - Activation functions
   - Best for: Getting started with deep learning

### Scripts (Ready-to-Run Code)

- **neural_network_simple.py** - NN implementation from scratch for educational purposes
- **train_model.py** - Functions for training scikit-learn and TensorFlow models

### Utilities (Helper Functions)

- **data_loader.py**
  - `load_iris_data()` - Load iris dataset
  - `load_digits_data()` - Load handwritten digits
  - `load_housing_data()` - Load California housing
  - `explore_dataset()` - Print dataset info
  - `scale_features()` - Normalize/standardize features

- **utils.py**
  - `plot_confusion_matrix()` - Visualize predictions
  - `evaluate_classification()` - Classification metrics
  - `evaluate_regression()` - Regression metrics
  - `plot_training_history()` - Neural network training curves
  - `plot_feature_importance()` - Feature rankings

## How to Use Each Component

### Running Notebooks
```bash
jupyter notebook
# Then open notebooks/01_data_exploration.ipynb in browser
```

### Using Scripts
```bash
python scripts/train_model.py
python scripts/neural_network_simple.py
```

### Using Utilities
```python
from utils.data_loader import load_iris_data
from utils.utils import evaluate_classification

X_train, X_test, y_train, y_test = load_iris_data()
# Your training code...
evaluate_classification(y_test, y_pred)
```

## Learning Path

1. Start with **01_data_exploration.ipynb** (30 mins)
2. Move to **02_supervised_learning.ipynb** (45 mins)
3. Explore **03_neural_networks.ipynb** (60 mins)
4. Review **scripts/** for reference implementations
5. Use **utils/** functions in your own experiments

---

**Recommended**: Follow notebooks in order (01, 02, 03) for best learning experience!
