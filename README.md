# Handwritten Digit Recognition

## Backpropagation 1
In this project a simple `feedforward neural network` is developed from scratch (without `PyTorch` or `TensorFlow`). The project contains the code and data needed for training the neural network with `backpropagation` to recognize handwritten digits with 90% accuracy.

Better results could be achieved with:
 - different size of the hidden layer
 - more hidden layers
 - extra activation functions (e.g. `softmax` to be used on the final layer)
 - playing with the learning rate and number of iterations

Feel free to play with these recommendations on your own. :)

NB: There is also one bug in the code, which does not affect the efficiency of the model at the moment.


## Backpropagation 2
Another `from scratch` model with different backprop implementation and `softmax` activation function

## Backpropagation 3
This is a fork of the first implementation, but adapted to work 4-layer neural network.

## Accuracy

With 1000 iterations we achieve:
* Backpropagation #1 -> 91%
* Backpropagation #2 -> 89%
* Backpropagation #3 -> 92%

With 10 000 iterations we achieve:
* Backpropagation #1 -> 91%
* Backpropagation #2 -> 94%
* Backpropagation #3 -> 98%
