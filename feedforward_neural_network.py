import numpy as np
import pandas as pd
import zipfile
import os

class NeuralNetwork:
    '''
    A class that implements a simple feedforward neural network
    '''
    w: list[np.ndarray] = []    # The weights of the network
    b: list[np.ndarray] = []    # The biases of the network
    layers: list[int]   = []    # The number of neurons in each layer
    L: int = 0                  # The number of layers in the network

    def __init__(self, layers: list[int]):
        '''
        Initialize the neural network

        Args:
            layers (list[int]): The number of neurons in each layer
        '''
        self.layers = layers
        self.L = len(layers)
        for l in range(1, self.L):
            self.w.append(np.random.randn(self.layers[l], self.layers[l-1]) * 0.01)     # We initialize the weights to small random values
            self.b.append(np.zeros((self.layers[l], 1)))                                # We initialize the biases to 0

    def __ReLU(self, Z: np.ndarray) -> np.ndarray:
        '''Applies the ReLU activation function to the input'''
        return np.maximum(Z, 0)

    def __forward_prop(self, X: np.ndarray) -> list[np.ndarray]:
        '''
        Performs forward propagation

        Args:
            X (np.ndarray): The input data

        Returns:
            A (list[np.ndarray]): The outputs of the activation functions for each layer
        '''
        A = [X.T] # The input data is transposed to be a column vector

        for l in range(1, self.L):
            z = self.b[l-1] + self.w[l-1] @ A[l-1]  # Linear transformation of the input data
            A.append(self.__ReLU(z))                # Apply an activation function to make the output non-linear

        return A
    
    def __one_hot(self, Y: np.ndarray) -> np.ndarray:
        '''
        One-hot encodes the labels

        Args:
            Y (np.ndarray): Flat list of labels

        Returns:
            np.ndarray: The one-hot encoded labels
        '''
        one_hot_Y = np.zeros((Y.size, int(Y.max() + 1)))    # Create a vector with 10 zeros for each observation in the dataset
        one_hot_Y[np.arange(Y.size), Y] = 1                 # For each observation, set the value of the one-hot vector to 1 at the index of the label
        return one_hot_Y.T                                  # Transpose the one-hot encoded labels to be a column vector

    def __backward_prop(self, A: list[np.ndarray], Y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        '''
        Performs backward propagation

        Args:
            A (list[np.ndarray]): The activation functions of the input data
            Y (np.ndarray): The one-hot encoded labels
        
        Returns:
            dw (list[np.ndarray]): The gradients of the weights
            db (list[np.ndarray]): The gradients of the biases
        '''
        dw = []
        db = []
        for l in range(self.L - 1):
            one_hot_Y = self.__one_hot(Y)
            m = one_hot_Y.shape[0]
            dZ = A[l+1] - one_hot_Y
            dw.append(dZ @ A[l].T / m)
            db.append(np.sum(dZ) / m)
        return dw, db
    
    def __update_parameters(self, dw: list[np.ndarray], db: list[np.ndarray], alpha: float):
        '''Updates the parameters of the neural network (weights and biases) using gradient descent.'''
        for l in range(self.L - 1):
            self.w[l] = self.w[l] - alpha * dw[l]
            self.b[l] = self.b[l] - alpha * db[l]

    def __get_accuracy(self, A: list[np.ndarray], Y: np.ndarray) -> float:
        '''
        Calculates the accuracy of the model based on the expected outputs (labels) and the predicted outputs (activations)

        Args:
            A (list[np.ndarray]): The outputs of the activation functions for each layer
            Y (np.ndarray): The expected outputs (labels)

        Returns:
            float: The accuracy of the model - a number between 0 and 1 (the percentage of correct predictions)
        '''
        predictions = np.argmax(A[-1], axis=0) # Get the index of the maximum value in the output layer (1 means the model predicts the digit 1, 2 means the model predicts the digit 2, etc.)
        return np.sum(predictions == Y) / Y.size # Calculate the accuracy of the model on the training data

    def measure_final_accuracy_on_test_data(self, X: np.ndarray, Y: np.ndarray) -> float:
        '''
        Calculates the accuracy of the model by converting the test data X into activations and then comparing them with the expected outputs (labels)

        Args:
            X (np.ndarray): The input test data
            Y (np.ndarray): The expected outputs (labels)

        Returns:
            float: The accuracy of the model - a number between 0 and 1 (the percentage of correct predictions)
        '''
        A = self.__forward_prop(X)
        return self.__get_accuracy(A, Y)

    def gradient_descent(self, X: np.ndarray, Y: np.ndarray, alpha: float, num_iters: int):
        '''
        Performs gradient descent on the model

        Args:
            X (np.ndarray): The input training data
            Y (np.ndarray): The expected outputs (labels)
            alpha (float): The learning rate
            num_iters (int): The number of iterations to run the gradient descent
        '''
        for i in range(num_iters):
            A = self.__forward_prop(X)
            dw, db = self.__backward_prop(A, Y)
            self.__update_parameters(dw, db, alpha)
            # Print the accuracy of the model on the training data
            if i % 50 == 0:
                print(f"\tIteration #{i} accuracy: {round(self.__get_accuracy(A, Y), 4)}")

def main():
    # Unpack the archive.zip file, if it is not already unpacked
    if not os.path.isfile('data/mnist_train.csv'):
        print("0. Unpacking the archive.zip file...")
        with zipfile.ZipFile('data/archive.zip', 'r') as zip_ref:
            zip_ref.extractall('data')

    # Load the training data
    print("1. Loading the training data...")
    train_data = pd.read_csv('data/mnist_train.csv')

    # Remove the label column from the data
    X = train_data.drop(columns=['label'])
    Y = train_data['label']

    # Convert to numpy arrays
    X = np.asarray(X, dtype="float64")
    X = X / 255. # Normalize the pixel values to be between 0 and 1
    Y = np.asarray(Y, dtype="int32")

    # Initialize the neural network
    nn = NeuralNetwork([784, 10, 10])

    # Perform gradient descent on the model
    learning_rate = 0.00003
    iterations = 200
    print(f"2. Running gradient descent for {iterations} iterations with a learning rate of {learning_rate}...")
    nn.gradient_descent(X, Y, learning_rate, iterations)

    # Load the test data
    print("3. Loading the test data...")
    test_data = pd.read_csv('data/mnist_test.csv')
    X_test = test_data.drop(columns=['label'])
    Y_test = test_data['label']

    # Convert to numpy arrays
    X_test = np.asarray(X_test, dtype="float64")
    X_test = X_test / 255. # Normalize the pixel values to be between 0 and 1
    Y_test = np.asarray(Y_test, dtype="int32")

    # Measure the final accuracy of the model on the test data
    print("4. Measuring the final accuracy of the model on the test data...")
    accuracy = nn.measure_final_accuracy_on_test_data(X_test, Y_test)
    print(f"\tFinal accuracy: {round(accuracy*100, 2)}%")

if __name__ == "__main__":
    main()
