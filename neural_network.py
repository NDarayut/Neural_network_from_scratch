import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
from loss_function import L2_loss

data = pd.read_csv('mnist_digits.csv')

# convert the data into numpy array for processing
data = np.array(data)

m, n = data.shape

'''
This will be our test set, which will contain 500 instances

Y_test will be our training label (0,1,2...10)
X_test will be our feature vectors (pixel value)

'''

data_test = data[0:2000].T # our test set will be 500 instances and again we transpose it so our label will be the first row
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.

'''
This will be our training set, which will contain 1000 instances

Y_train will be our training label (0,1,2...10)
X_train will be our feature vectors (pixel value)

'''

data_train = data[2000:m].T # transpose the matrix so all the label will be in the first row
Y_train = data_train[0] # take the first row as training label
X_train = data_train[1:n] # take the second row (which is the first pixel) until the 785th pixel
X_train = X_train / 255. # rescale the pixel value between 0-1

'''
Our Neural Network will have 1 Input layer which is the 784 pixel input and each input will map to a neuron
All of the connection will have a weight associate with it and all the neuron will have a bias

Forward propagation:

    Z1 = sum( W1 * X ) + b1
    A(Z1) = ReLu(Z1) = max(Z1, 0)
    Z2 = sum( W2 * A(Z1) ) + b2
    Y_hat = softmax( Z2 ) = exp(Z2) / sum(exp(Z2))

Where:

    --X is the pixel input
    --W1 is the weights in the first connection layer
    --b1 is the biases of all the neuron in the first layer
    --Z1 is the value after perform linear regression on the first hidden layer
    --A(Z1) is the value of the first layer after applying the activation function ReLu
    --Z2 is the the value after perform another linear regression on the second layer (output layer)
    --Y_hat is the probability of a class after applying the activation function softmax

'''

def initialization_parameters():

    '''
    --np.random.randn(10, 784) will create weights for 10 neuron that correspond to 784 connection
    --If we want to add more neuron we can change the number 10

    --We initialize our weight using He initialization

    '''
    
    W1 = np.random.randn(10, 784) * np.sqrt(2/784)
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * np.sqrt(2/10)
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2
    

def ReLu(Z1):

    '''
    ReLu activation function will return any value that is greater 0

    '''

    return np.maximum(Z1, 0)

def softmax(Z2):

    '''
    Softmax will return the probability of each class
    Classes with the highest probability will be the predicted outcome

    '''

    A = np.exp(Z2)/sum(np.exp(Z2))
    return A
            
def forward_propagation(X, W1, b1, W2, b2):

    '''
    --X input value (784 pixel)
    --W1 the weights of the first connection layer
    --b1 the biases of the neuron in the first layer
    --W2 the weights of second connection layer (output layer
    --b2 the biases of the neuron in the second layer (output layer)

    --Z1 the linear combination of the first layer
    --A1 the activated linear combination (ReLu)
    --Z2 the linear combination of the second layer
    --Y_hat the output after applying softmax activation function

    '''

    Z1 = np.dot(W1, X ) + b1
    A1 =  ReLu(Z1)
    Z2 = np.dot(W2, A1 ) + b2
    Y_hat = softmax(Z2)
    return  Z1, A1, Z2, Y_hat


def derivative_of_relu(Z1):

    '''
    The derivative of ReLu return 1 as long as the value is greater than 0

    '''

    return Z1 > 0

def one_hot_encoding(Y, num_classes):

    '''
    This function encode every correct label as 1
    e.g Label: 5, encoded_label: [0,0,0,0,0,1,0,0,0,0]
        Label: 1, encoded_label: [0,1,0,0,0,0,0,0,0,0]
    '''
    
    Y_true = np.zeros((len(Y), num_classes))
    Y_true[np.arange(len(Y)), Y] = 1
    return Y_true.T


def backward_propagation(Z1, A1, W2, Y_hat, X, Y):

    '''
    --Z1 is the linear combination of the first layer
    --A1 is the activated linear combination (ReLU)
    --W2 is the weights of the second connection layer
    --Y_hat is the probability of each class
    --X is the features
    --Y is the target (True value after onehot encoded)
    --m is the number of instances

    --dW2 is the derivative of the loss function with respect to weights in the second layer
    --db2 is the derivative of the loss function with respect to biases in the second layer
    --dW1 is the derivative of the loss function with respect to weights in the first layer
    --db1 is the derivative of the loss function with respect to biases in the first layer

    '''

    dL2_loss = (Y_hat - Y)*2 
    dW2 = ( np.dot(dL2_loss, A1.T) ) / m
    db2 = ( np.sum(dL2_loss) ) / m
    dZ1 = W2.T.dot(dL2_loss) * derivative_of_relu(Z1)
    dW1 = ( dZ1.dot(X.T) ) / m
    db1 = ( np.sum(dZ1) ) / m

    return dW2, db2, dW1, db1
    

def gradient_descent(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):

    '''
    Gradient descent is an optimization algorithm that change the weights and biases
    through steps (learning rate)

    --W1 the weights from the first layer
    --b1 the biases from the first layer
    --W2 the weights from the second layer
    --b2 the biases from the second layer
    --dW1 the change in weights from the first layer
    --db1 the change in biases from the first layer
    --dW2 the change in weights from the second layer
    --db2 the change in biases from the second layer
    --learning_rate the step in the opposite direction of the gradient

    After running through the algorithm we will have a new set of weights and biases shifted
    at the opposite direction of the gradient

    '''

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    return W1, b1, W2, b2

def predict(Y_hat):

    '''
    This function return the index with the highest probability
    e.g [0.2, 0.3, 0.9] would return [2] since index 2 has the highest probability

    '''
    return np.argmax(Y_hat, 0)

def accuracy(Y_hat, Y):

    '''
    This function give us the proportion of correct prediction
    in terms of the actual label

    --Y_hat the predicted outcome
    --Y the true label
    --Y.size the total number of label (4000+)

    '''

    return np.sum(Y_hat == Y) / Y.size

def fit(X, Y, learning_rate, iteration):

    '''
    This function will fit the model with the X, Y, learning_rate and number of the iteration
    After the model is trained, it will output a set of weights and biases where we can use to
    test our prediction on the test set

    --X the features of the train set
    --Y the labels of the train set
    --learning_rate the amount of steps to take in the opposite of the gradient
    --iteration number of times to iterate over the set

    --W1 the new set of weights for the first layer
    --b1 the new set of biases for the first layer
    --W2 the new set of weights for the second layer
    --b2 the new set of biases for the second layer

    '''

    Y_true = one_hot_encoding(Y,10)

    W1, b1, W2, b2 = initialization_parameters() # Weights and biases initialization
    for i in range(iteration):
        Z1, A1, Z2, Y_hat = forward_propagation(X, W1, b1, W2, b2)
        dW2, db2, dW1, db1 = backward_propagation(Z1, A1, W2, Y_hat, X, Y_true)
        W1, b1, W2, b2 = gradient_descent(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        if i % 10 == 0:
            print("Iteration:", i)
            print(f"accuracy: {accuracy(predict(Y_hat), Y)*100}%")
            print("Loss:", L2_loss(Y_true, Y_hat))
            print("")
    return W1, b1, W2, b2

def test_accuracy(X_test, Y_test, W1, b1, W2, b2):

    '''
    This function will only perform forward propagation to get the predicted value
    The value is then use to evaluate the accuracy of the model on the test set

    Note: All the weights and biases are after the training is complete
          so all the weights and biases are optimize to get the highest accuracy possible
    '''
    
    _, _, _, Y_pred = forward_propagation(X_test, W1, b1, W2, b2)
    Y_pred = predict(Y_pred)
    test_accuracy = accuracy(Y_pred, Y_test)
    print(f"Test accuracy: {test_accuracy*100}%")

def make_predictions(X, W1, b1, W2, b2):

    '''
    This function is used to make prediction on a single image
    '''
    _, _, _, Y_pred = forward_propagation(X, W1, b1, W2, b2)
    predictions = predict(Y_pred)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    
    '''
    This function is used to test the prediction and plot the actual image of the label
    and compare it to the predicted label
    '''

    current_image = X_test[:, index, None]
    prediction = make_predictions(X_test[:, index, None], W1, b1, W2, b2)
    label = Y_test[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()







    









