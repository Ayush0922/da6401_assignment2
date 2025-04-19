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
- `matplotlib`: For creating visualizations, such as the prediction grid in `A4.py`.
- `wandb`: Weights & Biases library for experiment tracking and hyperparameter optimization (used in `A2.py`).

## 🚀 Running the Code

Before running the scripts, ensure that you have downloaded the iNaturalist 12K dataset and placed the `inaturalist_12K` folder in the same directory as the Python scripts, or update the `DATA_PATH`, `train_path`, and `test_path` variables within the scripts to the correct locations of your dataset. The dataset should have `train` and `val` subdirectories containing the image data organized by class.

### ▶️ Running `A1.py`

This script defines a basic CNN. To train and evaluate it, you would need to add a training loop and evaluation steps. Here's a basic outline of how you might extend `A1.py`:

1.  **Add a Training Loop:** Iterate over the training data loader, perform forward passes, calculate the loss using `loss_fn`, perform backpropagation, and update the model's weights using the `optimiser`.
2.  **Implement Validation:** After each epoch (or at intervals), evaluate the model on the validation data loader to monitor performance and prevent overfitting.
3.  **Evaluate on Test Set:** Once training is complete, evaluate the model on the `test_dl` to get the final performance metrics.

You can refer to the training and evaluation logic in `A4.py` for a more detailed implementation. To run the extended `A1.py` (after adding the training and evaluation code):

```bash
python A1.py
```

### ▶️ Running `A2.py`

`A2.py` is designed for hyperparameter sweeping using Weights & Biases. Training and evaluation are handled within the `experiment` function, which is called by the wandb agent.

1.  **Set up wandb:** Ensure you have installed and logged into wandb (`pip install wandb` followed by `wandb login`).
2.  **Run the sweep:** Execute the script:

```bash
python A2.py
```

Wandb will manage the training and evaluation process for different hyperparameter configurations defined in `SWEEP_CFG`. The best performing models based on the validation accuracy will be tracked by wandb.

### ▶️ Running `A4.py`

`A4.py` includes a complete training and evaluation pipeline.

1.  **Ensure correct data paths:** Verify that `train_path` and `test_path` in the `if __name__ == '__main__':` block point to your training and validation datasets, respectively.
2.  **Run the script:**

```bash
python A4.py
```

This script will:
- Load the data.
- Train the `ConvNet` model using the configuration in the `best` dictionary.
- Evaluate the model on the validation set during training and save the best weights.
- Evaluate the final trained model on the test set and print the test accuracy.
- Generate a visualization of predictions on the test set.

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

You can modify the configurations in the scripts (e.g., network architecture, hyperparameters, data augmentation techniques) to experiment with different models and training strategies. `A1.py` can be extended with custom training and evaluation loops. `A2.py` provides a framework for automated hyperparameter tuning. `A4.py` offers a more structured end-to-end training, evaluation, and visualization process that can be adapted and expanded upon.
