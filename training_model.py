from neural_network import *
import pickle
import numpy as np

# Fitting the model with the train set and run it for 100 iteration over the set
W1, b1, W2, b2 = fit(X=X_train, Y=Y_train, learning_rate=0.3, Epochs=100, batch_size=192)

layer_1 = np.count_nonzero(W1) + np.count_nonzero(b1)
layer_2 = np.count_nonzero(W2) + np.count_nonzero(b2)
print(f"Layer 1 parameters: {layer_1}")
print(f"Layer 2 parameters: {layer_2}")
print(f"Total trainable parameters: {layer_1 + layer_2}\n")

with open('parameters.pkl', 'wb') as f:
    # Use pickle.dump to serialize each variable and write them to the file
    pickle.dump(W1, f)
    pickle.dump(b1, f)
    pickle.dump(W2, f)
    pickle.dump(b2, f)
  

