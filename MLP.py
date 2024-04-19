import numpy as np


class MLP_Classifier:
    """
    MLP_Classifier class.

    This class represents a neural network model with a specified number of input units, hidden units, and output units.
    It provides methods for initializing the model, getting the weights, performing forward propagation, and implementing backpropagation.

    Attributes:
        input_units (int): Number of input units.
        hidden_units (int): Number of hidden units.
        output_units (int): Number of output units.
        hidden_weights (ndarray): Weights of the connections between the input and hidden layers.
        output_weights (ndarray): Weights of the connections between the hidden and output layers.
        normalize (bool): Whether to normalize the input data.
        activation (str): Activation function to use. Options: "sigmoid", "relu", "tanh".

    Methods:
        __init__(self, input_units=2, hidden_units=4, output_units=1, weights=None): Initializes the neural network model.
        get_weights(self): Returns the weights of the model.
        activation_function(self, x): Applies the activation function to the given input.
        activation_derivative(self, x): Computes the derivative of the activation function.
        predict(self, X): Predicts the output values for the given input data.
        backpropagation(self, X, y, learning_rate, epochs, early_stopping=False, patience=100, tol=1e-4, verbose=False, verbose_step=1000, validation_split=0.2): Performs backpropagation to train the model.

    """

    def __init__(
        self,
        input_units=2,
        hidden_units=4,
        output_units=1,
        weights=None,
        normalize=False,
        activation="sigmoid",
    ):
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.normalize = normalize
        self.activation = activation

        if weights:
            self.hidden_weights, self.output_weights = weights
        else:
            # Initialize weights with random values
            self.hidden_weights = np.random.uniform(size=(input_units, hidden_units))
            self.output_weights = np.random.uniform(size=(hidden_units, output_units))

    def get_weights(self):
        """
        Returns the weights of the model.

        Returns:
            tuple: A tuple containing the hidden weights and output weights.
        """
        return self.hidden_weights, self.output_weights
    
    def set_weights(self, weights):
        self.hidden_weights, self.output_weights = weights

    def _sigmoid(self, x):
        """
        Applies the sigmoid activation function to the given input.

        Args:
            x (ndarray): Input values.

        Returns:
            ndarray: Output values after applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        """
        Computes the derivative of the sigmoid function for the given input.

        Args:
            x (ndarray): Input values.

        Returns:
            ndarray: Derivative values of the sigmoid function.
        """
        return x * (1 - x)

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def _tanh(self, x):
        return np.tanh(x)

    def _tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def activation_function(self, x):
        if self.activation == "sigmoid":
            return self._sigmoid(x)
        elif self.activation == "relu":
            return self._relu(x)
        elif self.activation == "tanh":
            return self._tanh(x)
        else:
            raise ValueError("Invalid activation function")

    def activation_derivative(self, x):
        if self.activation == "sigmoid":
            return self._sigmoid_derivative(x)
        elif self.activation == "relu":
            return self._relu_derivative(x)
        elif self.activation == "tanh":
            return self._tanh_derivative(x)
        else:
            raise ValueError("Invalid activation function")

    def predict(self, X):
        """
        Predicts the output values for the given input data.

        Args:
            X (ndarray): Input data.

        Returns:
            ndarray: Predicted output values.
        """
        if self.normalize:
            X = (X - X.mean(axis=0)) / X.std(axis=0)
        hidden_layer_input = np.dot(X, self.hidden_weights)
        hidden_layer_output = self.activation_function(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.output_weights)
        output = self.activation_function(output_layer_input)

        return output

    def backpropagation(
        self,
        X,
        y,
        learning_rate,
        epochs,
        early_stopping=False,
        patience=100,
        verbose=False,
        verbose_step=1000,
        validation_split=0.2,
    ):
        """
        Performs backpropagation to train the model.

        Args:
            X (ndarray): Input data.
            y (ndarray): Target output data.
            learning_rate (float): Learning rate for updating the weights.
            epochs (int): Number of training epochs. 0 means early stopping.
            early_stopping (bool): Whether to use early stopping.
            patience (int): Number of epochs to wait before stopping if no improvement is seen.
            verbose (bool): Whether to print the loss at each epoch.
            verbose_step (int): Step size for printing the loss.

        Returns:
            tuple: A tuple containing the training losses, validation losses, best weights, and the best validation loss.
        """
        if self.normalize:
            X = (X - X.mean(axis=0)) / X.std(axis=0)
        # Randomly split the data into training and validation sets
        num_samples = X.shape[0]
        num_validation_samples = int(num_samples * validation_split)
        indices = np.random.permutation(num_samples)
        X_train = X[indices[:-num_validation_samples]]
        y_train = y[indices[:-num_validation_samples]]
        X_val = X[indices[-num_validation_samples:]]
        y_val = y[indices[-num_validation_samples:]]
        patience_count = 0
        early_stopping_prev_loss = np.inf
        epoch = 0
        training_losses = []
        validation_losses = []
        best_weights = (self.hidden_weights, self.output_weights)
        while True:
            epoch += 1
            # Forward propagation
            hidden_layer_input = np.dot(X_train, self.hidden_weights)
            hidden_layer_output = self.activation_function(hidden_layer_input)

            output_layer_input = np.dot(hidden_layer_output, self.output_weights)
            output = self.activation_function(output_layer_input)

            # Backpropagation
            output_error = y_train - output
            training_losses.append(np.mean(np.square(output_error)))

            output_delta = output_error * self.activation_derivative(output)
            hidden_error = output_delta.dot(self.output_weights.T)
            hidden_delta = hidden_error * self.activation_derivative(
                hidden_layer_output
            )

            # Update weights
            self.output_weights += (
                hidden_layer_output.T.dot(output_delta) * learning_rate
            )
            self.hidden_weights += X_train.T.dot(hidden_delta) * learning_rate

            validation_loss = np.mean(np.square(y_val - self.predict(X_val)))
            validation_losses.append(validation_loss)
            if early_stopping:
                if validation_loss < early_stopping_prev_loss:
                    best_weights = (self.hidden_weights, self.output_weights)
                    early_stopping_prev_loss = validation_loss
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count == patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break
            else:
                if validation_loss < early_stopping_prev_loss:
                    early_stopping_prev_loss = validation_loss
                    best_weights = (self.hidden_weights, self.output_weights)

            if verbose and epoch % verbose_step == 0:
                loss = np.mean(np.abs(output_error))
                print(
                    f"Epoch: {epoch}, Loss: {loss}, Validation Loss: {validation_loss}"
                )

            if epochs != 0 and epoch == epochs:
                print(f"Training completed at epoch {epoch}")
                break

        self.hidden_weights, self.output_weights = best_weights
        return (
            training_losses,
            validation_losses,
            best_weights,
            early_stopping_prev_loss,
        )
