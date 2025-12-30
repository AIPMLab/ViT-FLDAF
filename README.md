# Federated Vision Transformer with Adaptive Focal Loss (ViT-FLDAF)

This repository implements a Federated Learning framework for medical image classification using a Vision Transformer (ViT) backbone and a novel Dynamic Adaptive Focal Loss (DAFL). This approach is designed to address the challenges of data heterogeneity and class imbalance inherent in federated medical imaging datasets.

## Methodology

### 1. Vision Transformer Backbone
We utilize a standard Vision Transformer (`vit_small_patch16_224`) as the global model. The self-attention mechanism of ViT allows the model to capture long-range dependencies and global context, which is critical for identifying subtle lesions in medical images.

### 2. Dynamic Adaptive Focal Loss (DAFL)
To tackle class imbalance and hard samples, we propose the Dynamic Adaptive Focal Loss. The loss function is defined as:

$$ L = - (1+c_{f,t}) \cdot (1-p_t)^\gamma \cdot \log(p_t) $$

Where:
- $p_t$ is the predicted probability of the true class.
- $\gamma$ is the focusing parameter (learnable, initialized to 2.0), which down-weights easy examples and focuses training on hard negatives.
- $c_{f,t}$ is the dynamic imbalance coefficient for the target class $t$.

The imbalance coefficient $c_{f,t}$ combines client-level and class-level statistics:

$$ c_{f,t} = \lambda \cdot c_{f,\text{client}} + (1-\lambda) \cdot c_{f,i} $$

- **$c_{f,i}$ (Class-level)**: Measures the global scarcity of class $i$ across all clients.
- **$c_{f,\text{client}}$ (Client-level)**: Measures the overall data distribution skewness within specific clients.
- **$\lambda$**: A balancing coefficient (set to 0.5) to weigh the importance of client vs. class imbalance.

### 3. Client-Aware Aggregation
Instead of simple averaging (FedAvg), we employ a weighted aggregation strategy where client contributions are weighted by their local dataset size. This ensures that the global model better represents the underlying data distribution.

## Requirements

- Python 3.8+
- PyTorch
- Torchvision
- Timm (PyTorch Image Models)
- NumPy
- Scikit-learn

Install dependencies via:
```bash
pip install torch torchvision timm numpy scikit-learn
```

## Usage

1. **Prepare Data**: Ensure your dataset is organized in the standard ImageFolder structure:
   ```
   /path/to/data/
       client_0/
           class_A/
           class_B/
       client_1/
           ...
   ```
   Update the `pathes` variable in `main.py` to point to your dataset root.

2. **Run Training**:
   ```bash
   python main.py
   ```

## Key Parameters

- `Num_clients`: Number of federated clients (default: 4).
- `gamma`: Focusing parameter for Focal Loss (learnable).
- `lambda_val`: Balancing coefficient for imbalance parameters (default: 0.5).
- `epochs`: Number of local training epochs per round.
- `communication_rounds`: Number of global communication rounds (default: 50).

## File Structure

- `main.py`: The main entry point containing the FL simulation, ViT model definition, DAFL implementation, and training loops.
