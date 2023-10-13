import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Load the dataset
data = pd.read_csv("housing.csv")
target_column = "MEDV"

# Normalize the data
data = (data - data.mean()) / data.std()

# Extract features and target variable
X = data.drop(columns=[target_column]).values
y = data[[target_column]].values

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Function to initialize weights with random numbers
def initialize_weights(input_neurons, hidden_neurons, output_neurons):
    hidden_layer_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
    output_layer_weights = np.random.uniform(size=(hidden_neurons, output_neurons))
    return hidden_layer_weights, output_layer_weights

# Function for forward and backward propagation
def train_neural_network(X, y, hidden_layer_weights, output_layer_weights, learning_rate, epochs):
    for _ in range(epochs):
        # Forward Propagation
        hidden_layer_input = np.dot(X, hidden_layer_weights)
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, output_layer_weights)
        predicted_output = sigmoid(output_layer_input)

        # Backpropagation
        error = y - predicted_output
        output_error = error * sigmoid_derivative(predicted_output)
        hidden_layer_error = output_error.dot(output_layer_weights.T) * sigmoid_derivative(hidden_layer_output)

        # Updating weights
        output_layer_weights += hidden_layer_output.T.dot(output_error) * learning_rate
        hidden_layer_weights += X.T.dot(hidden_layer_error) * learning_rate

    return predicted_output

# Function for cross-validation and accuracy calculation
def cross_validation(X, y, hidden_neurons, learning_rate, epochs, folds=5):
    kf = KFold(n_splits=folds)
    accuracies = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        hidden_layer_weights, output_layer_weights = initialize_weights(X_train.shape[1], hidden_neurons, y_train.shape[1])

        predicted_output = train_neural_network(X_train, y_train, hidden_layer_weights, output_layer_weights, learning_rate, epochs)
        accuracy = np.mean((np.abs(np.round(predicted_output.flatten()) - y_test.flatten()) < 0.5).astype(int)) * 100
        accuracies.append(accuracy)

    return np.mean(accuracies)

# Main function
def main():
    cases = [
        {"hidden_neurons": 3, "learning_rate": 0.01},
        {"hidden_neurons": 4, "learning_rate": 0.001},
        {"hidden_neurons": 5, "learning_rate": 0.0001}
    ]

    for case in cases:
        hidden_neurons = case["hidden_neurons"]
        learning_rate = case["learning_rate"]
        epochs = 1000

        accuracy_5_fold = cross_validation(X, y, hidden_neurons, learning_rate, epochs, folds=5)
        accuracy_10_fold = cross_validation(X, y, hidden_neurons, learning_rate, epochs, folds=10)
        print(f"Average accuracy for {hidden_neurons} neurons and learning rate {learning_rate}:")
        print(f"5-Fold Cross Validation: {accuracy_5_fold:.2f}%")
        print(f"10-Fold Cross Validation: {accuracy_10_fold:.2f}%")
        print("---")

# Run the main function
if __name__ == "__main__":
    main()
