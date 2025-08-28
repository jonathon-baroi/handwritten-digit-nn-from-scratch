# Handwritten Digit Recognition — Neural Network from Scratch

This repository contains a **neural network framework built from scratch** (using only `numpy`) to classify handwritten digits.  
It demonstrates the fundamental steps of building, training, and evaluating a neural network — without relying on high-level machine learning libraries like TensorFlow or PyTorch.

---

## ✨ Features

- Define model architectures with:
  - `Linear` (fully connected) layers
  - `ReLU` activation
  - `Softmax` output
- **Training pipeline** with:
  - Cross-Entropy Loss
  - SGD Optimizer (with learning rate scheduling support)
  - Mini-batch gradient descent
- **Evaluation metrics**: accuracy and loss
- Forward & backward propagation implemented manually

---

## 🧩 Model Architecture Example

```python
model = Sequential([
    Linear(n_features, 64),
    ReLU(),
    Linear(64, 32),
    ReLU(),
    Linear(32, n_classes)
])

model.compile(
    optimizer=SGD(learning_rate=0.05),
    loss=CrossEntropyLoss()
)

model.fit(X_train, y_train, epochs=1500, batch_size=128)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {test_accuracy*100:.2f}% | Loss: {test_loss:.4f}")
