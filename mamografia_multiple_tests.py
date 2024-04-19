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

hidden_layer_sizes = [4, 8, 16, 32, 64, 128]
activation_functions = ["sigmoid"]
best_accuracy = 0
best_model = None
best_training_losses = None
best_validation_losses = None


for hidden_units in hidden_layer_sizes:
    for activation in activation_functions:
        model = MLP_Classifier(
            input_units=X_train.shape[1],
            hidden_units=hidden_units,
            output_units=1,
            normalize=True,
            activation=activation,
        )
        (training_losses, validation_losses, best_weights, best_validation_loss) = (
            model.backpropagation(
                X_train,
                y_train,
                learning_rate=0.001,
                epochs=0,
                verbose=False,
                early_stopping=True,
                patience=100,
                validation_split=0.2,
            )
        )

        # Calculate the accuracy of the model
        y_pred = model.predict(X_test)
        accuracy = np.mean((y_pred > 0.5) == y_test)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_training_losses = training_losses
            best_validation_losses = validation_losses

        print(
            f"Hidden Units: {hidden_units}, Activation: {activation}, Accuracy: {accuracy * 100:.2f}%"
        )


# Predict the output for the test set
y_pred = best_model.predict(X_test)

# Calculate the accuracy of the model
accuracy = np.mean((y_pred > 0.5) == y_test)
print(f"Accuracy of the best model: {accuracy * 100:.2f}%")


SMOOTH_GRAPH = True

if SMOOTH_GRAPH:
    training_losses = np.convolve(best_training_losses, np.ones(10) / 10, mode="valid")
    validation_losses = np.convolve(
        best_validation_losses, np.ones(10) / 10, mode="valid"
    )

    plt.semilogy(training_losses, label="Training Loss")
    plt.semilogy(validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses of Best model")
    plt.legend()

    plt.text(
        0.97,
        0.80,
        f"Test Score: {accuracy * 100:.2f}%",
        transform=plt.gca().transAxes,
        ha="right",
    )

    plt.show()
