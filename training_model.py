from neural_network import *
import pickle

# Fitting the model with the train set and run it for 100 iteration over the set
W1, b1, W2, b2 = fit(X_train, Y_train, 0.1, 100)

with open('parameters.pkl', 'wb') as f:
    # Use pickle.dump to serialize each variable and write them to the file
    pickle.dump(W1, f)
    pickle.dump(b1, f)
    pickle.dump(W2, f)
    pickle.dump(b2, f)
  

