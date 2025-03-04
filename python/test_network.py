import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt

# Load the model architecture
layers = get_lenet()
params = init_convnet(layers)

# Load the network
data = loadmat('../results/lenet.mat')
params_raw = data['params']

for params_idx in range(len(params)):
    raw_w = params_raw[0,params_idx][0,0][0]
    raw_b = params_raw[0,params_idx][0,0][1]
    assert params[params_idx]['w'].shape == raw_w.shape, 'weights do not have the same shape'
    assert params[params_idx]['b'].shape == raw_b.shape, 'biases do not have the same shape'
    params[params_idx]['w'] = raw_w
    params[params_idx]['b'] = raw_b

# Load data
fullset = False
xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = load_mnist(fullset)


# Testing the network
#### Modify the code to get the confusion matrix ####
all_preds = []
for i in range(0, xtest.shape[1], 100):
    cptest, P = convnet_forward(params, layers, xtest[:,i:i+100], test=True)

# hint: 
#     you can use confusion_matrix from sklearn.metrics (pip install -U scikit-learn)
#     to compute the confusion matrix. Or you can write your own code :)
    
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
all_labels = []
for i in range(0, xtest.shape[1], 100):
    cptest, P = convnet_forward(params, layers, xtest[:,i:i+100], test=True)
    # print(f"Shape of P: {P.shape}")
    # print(f"Shape of ytest batch: {ytest[i:i + 100].shape}")
    preds = np.argmax(P, axis=0)
    all_preds.append(preds)
    all_labels.append(ytest[i:i + 100])

# Convert into arrays
all_preds = np.array(all_preds).flatten()
all_labels = [arr for arr in all_labels if arr.size > 0] # drop out empty arrays
all_labels = np.concatenate(all_labels).ravel()

conf_matrix = confusion_matrix(all_labels, all_preds)
# print("Confusion Matrix:\n", conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.arange(10))
disp.plot(cmap='inferno', colorbar=True)

plt.title('Confusion Matrix', fontsize=12)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Ground Truth', fontsize=14)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.savefig('../results/confusion matrix.png')
# plt.show()

