import numpy as np
import matplotlib.pyplot as plt
from MLP import MLP_Classifier
import pandas as pd

# Get data from xlsx file
data = pd.read_excel("dadosmamografia.xlsx").values
X = data[:, :-1]
y = data[:, -1].reshape(-1, 1)


TEST_SPLIT = 0.2

# Make a test split
num_samples = X.shape[0]
num_validation_samples = int(num_samples * TEST_SPLIT)
indices = np.random.permutation(num_samples)
X_train = X[indices[:-num_validation_samples]]
y_train = y[indices[:-num_validation_samples]]
X_test = X[indices[-num_validation_samples:]]
y_test = y[indices[-num_validation_samples:]]

# Create a MLP model with 5 input units, 4 hidden units, and 1 output unit

model = MLP_Classifier(
    input_units=5, hidden_units=40, output_units=1, normalize=True, activation="sigmoid"
)
(training_losses, validation_losses, best_weights) = model.backpropagation(
    X_train,
    y_train,
    learning_rate=0.001,
    epochs=0,
    verbose=True,
    verbose_step=1000,
    early_stopping=True,
    patience=100,
    validation_split=0.2,
)

# Predict the output for the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = np.mean((y_pred > 0.5) == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")


SMOOTH_GRAPH = True

if SMOOTH_GRAPH:
    training_losses = np.convolve(training_losses, np.ones(10) / 10, mode="valid")
    validation_losses = np.convolve(validation_losses, np.ones(10) / 10, mode="valid")

    plt.semilogy(training_losses, label="Training Loss")
    plt.semilogy(validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()

    plt.text(
        0.97,
        0.80,
        f"Test Score: {accuracy * 100:.2f}%",
        transform=plt.gca().transAxes,
        ha="right",
    )

    plt.show()
