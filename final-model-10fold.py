import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
# Load dataset from the given link
# Load the dataset
data = pd.read_csv("housing.csv")

# Define input features (RM, LSTAT, PTRATIO)
X = data[['RM', 'LSTAT', 'PTRATIO']].values

# Define target (MEDV)
y = data['MEDV'].values

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights with random values
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            # Forward Propagation
            hidden_input = np.dot(X, self.weights_input_hidden)
            hidden_output = self.sigmoid(hidden_input)

            output = np.dot(hidden_output, self.weights_hidden_output)

            # Backpropagation
            error = y - output
            output_delta = error
            hidden_error = output_delta.dot(self.weights_hidden_output.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

            # Update Weights
            self.weights_hidden_output += hidden_output.T.dot(output_delta) * self.learning_rate
            self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate

    def predict(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden)
        hidden_output = self.sigmoid(hidden_input)
        output = np.dot(hidden_output, self.weights_hidden_output)
        return output


# Define parameters
hidden_layers = [3, 4, 5]
learning_rates = [0.01, 0.001, 0.0001]
epochs = 1000

for hidden_size in hidden_layers:
    for learning_rate in learning_rates:
        kf = KFold(n_splits=10)  # 10-fold cross-validation
        accuracies = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = NeuralNetwork(input_size=3, hidden_size=hidden_size, output_size=1, learning_rate=learning_rate)
            model.train(X_train, y_train.reshape(-1, 1), epochs)

            predictions = model.predict(X_test).flatten()
            accuracy = np.mean(np.abs(predictions - y_test) / y_test)
            accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies)
        print(f"Hidden Neurons: {hidden_size}, Learning Rate: {learning_rate}, Mean Absolute Error: {mean_accuracy:.4f}")

