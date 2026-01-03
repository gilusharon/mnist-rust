# MNIST Classifier in Rust

A neural network implementation for classifying handwritten digits from the MNIST dataset, built with Rust and the [Candle](https://github.com/huggingface/candle) machine learning framework.

## Features

- **3-layer fully connected neural network** with ReLU activations
- **Automatic MNIST dataset loading** via `candle-datasets`
- **Model persistence** using safetensors format
- **Configurable hyperparameters** via command-line arguments
- **Hands on testing** with a drawing app

## Architecture

The model consists of:
- Input layer: 784 neurons (28×28 flattened MNIST images)
- Hidden layer 1: 128 neurons with ReLU activation
- Hidden layer 2: 64 neurons with ReLU activation
- Output layer: 10 neurons (one for each digit 0-9)

## Requirements

- Rust 1.70 or later
- Cargo

## Installation

Clone the repository and build the project:

```bash
cargo build --release
```

## Usage

### Training

Train the model with default settings (10 epochs, batch size 64, learning rate 0.001):

```bash
cargo run --release
```

Or with custom parameters:

```bash
cargo run --release -- --epochs 20 --batch-size 128 --learning-rate 0.0005
```

### Testing

Test a trained model:

```bash
cargo run --release -- test
```

### Command-Line Options

- `--epochs <N>`: Number of training epochs (default: 10)
- `--batch-size <N>`: Batch size for training (default: 64)
- `--learning-rate <RATE>`: Learning rate for SGD optimizer (default: 0.001)
- `test`: Run in test mode (evaluates existing model)

## Model Persistence

The trained model is automatically saved to `model.safetensors` after training. If this file exists when starting training, the model weights will be loaded from it (useful for resuming training).

## Performance

With default settings, the model typically achieves:
- Training loss decreases from ~2.3 to ~0.1-0.3 over 10 epochs
- Test accuracy: ~90-95% (depending on training duration and hyperparameters)

## Project Structure

```
MNIST_Classifier/
├── Cargo.toml          # Project dependencies
├── src/
│   └── main.rs         # Main implementation
└── README.md           # This file
```

## Dependencies

- `candle-core`: Core tensor operations
- `candle-nn`: Neural network layers and optimizers
- `candle-datasets`: MNIST dataset loader
- `anyhow`: Error handling
- `clap`: Command-line argument parsing

