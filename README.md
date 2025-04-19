# iNaturalist Image Classification

This repository contains Python scripts for training and evaluating Convolutional Neural Networks (CNNs) on the iNaturalist 12K dataset for image classification.

## 📂 Repository Structure

```
.
├── A1.py
├── A2.py
└── A4.py
```

- `A1.py`: Defines a customizable CNN architecture and a basic training setup.
- `A2.py`: Implements a more modular CNN with various configuration options, including hyperparameter sweeping using Weights & Biases (wandb).
- `A4.py`: Provides a flexible CNN implementation with options for different activation functions, filter organizations, and includes training, evaluation, and visualization functionalities.

## ⚙️ Installation

To run these scripts, you need to have Python 3.6+ installed on your system. We recommend creating a virtual environment to manage the dependencies.

### 🐍 Creating a Virtual Environment

Open your terminal or command prompt and navigate to the repository directory.

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 📦 Installing Dependencies

Once the virtual environment is activated, install the required libraries using pip:

```bash
pip install torch torchvision torchaudio scikit-learn matplotlib wandb
```

**Explanation of Dependencies:**

- `torch`: PyTorch library for tensor computations and building neural networks.
- `torchvision`: Provides datasets, model architectures, and image transformations for computer vision.
- `torchaudio`: (While not explicitly used in these scripts, it's good practice to install it if you plan to extend the project with audio data later).
- `scikit-learn`: Useful for machine learning tasks like stratified train-test splitting.
- `matplotlib`: For creating visualizations, such as the prediction grid in `A3.py`.
- `wandb`: Weights & Biases library for experiment tracking and hyperparameter optimization (used in `A2.py`).

## 🚀 Running the Code

Before running the scripts, ensure that you have downloaded the iNaturalist 12K dataset and placed the `inaturalist_12K` folder in the same directory as the Python scripts, or update the `DATA_PATH` variables in the scripts to the correct location. The dataset should have `train` and `val` subdirectories containing the image data organized by class.

### ▶️ Running `A1.py`

This script defines a basic CNN and trains it on a subset of the iNaturalist 12K dataset.

```bash
python A1.py
```

This will:
1. Prepare the data loaders for training, validation, and testing.
2. Instantiate the `CustomCNN` model.
3. Define the loss function and optimizer.
4. Print the model architecture and the total number of parameters.

**Note:** This script runs a basic setup without explicit training loops. You would need to add a training loop to see the model learn.

### ▶️ Running `A2.py`

This script utilizes Weights & Biases for hyperparameter sweeping. You need to have a wandb account and be logged in.

1. **Install wandb:** If you haven't already, install it with `pip install wandb`.
2. **Log in to wandb:** Run `wandb login` in your terminal and follow the instructions.

Then, run the script:

```bash
python A2.py
```

This will initiate a hyperparameter sweep defined in the `SWEEP_CFG` dictionary. Wandb will run multiple experiments with different hyperparameter combinations and track the results.

**Note:** Ensure that the `DATA_PATH` in `A2.py` correctly points to your iNaturalist 12K training data directory.

### ▶️ Running `A3.py`

This script provides a more complete training and evaluation pipeline, including visualization of predictions.

```bash
python A3.py
```

This will:
1. Load the iNaturalist 12K dataset and create data loaders.
2. Define and train the `ConvNet` model based on the provided `best` configuration.
3. Evaluate the trained model on the test set and print the accuracy.
4. Generate a visualization (`class_predictions_grid.png`) showing sample images with their ground truth labels and model predictions.

**Note:** Ensure that the `train_path` and `test_path` variables in the `if __name__ == '__main__':` block of `A3.py` point to the correct locations of your training and validation datasets, respectively.

## 💾 Dataset

These scripts are designed to work with the iNaturalist 12K dataset. You can download it from various sources online (e.g., Kaggle datasets). The expected directory structure is:

```
inaturalist_12K/
├── train/
│   ├── class_folder_1/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── class_folder_2/
│   │   ├── image2.png
│   │   └── ...
│   └── ...
└── val/
    ├── class_folder_1/
    │   ├── image_a.jpg
    │   └── ...
    ├── class_folder_2/
    │   ├── image_b.png
    │   └── ...
    └── ...
```

Make sure to adjust the `DATA_PATH`, `train_path`, and `test_path` variables in the Python scripts if your dataset is located elsewhere.

## 🔭 Further Exploration

You can modify the configurations in the scripts (e.g., network architecture, hyperparameters, data augmentation techniques) to experiment with different models and training strategies. The `A2.py` script provides a good starting point for exploring hyperparameter optimization using Weights & Biases. The `A3.py` script offers a more structured training and evaluation framework that you can extend.
```
