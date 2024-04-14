import pickle
import random
from neural_network import X_test, Y_test, test_accuracy, test_prediction

with open('parameters.pkl', 'rb') as f:
    # Use pickle.load to deserialize and load each variable
    W1 = pickle.load(f)
    b1 = pickle.load(f)
    W2 = pickle.load(f)
    b2 = pickle.load(f)


# Evaluate the model on the test set
test_accuracy(X_test, Y_test, W1, b1, W2, b2)

# randomly pick 10 images from the test set and predict it
for i in range(10):
    n = random.randint(1, 2000)
    test_prediction(n, W1, b1, W2, b2)