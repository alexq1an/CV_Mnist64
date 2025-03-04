import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os

layers = get_lenet()
params = init_convnet(layers)

data = loadmat('../results/lenet.mat')
params_raw = data['params']

for params_idx in range(len(params)):
    raw_w = params_raw[0, params_idx][0, 0][0]
    raw_b = params_raw[0, params_idx][0, 0][1]
    assert params[params_idx]['w'].shape == raw_w.shape, 'weights do not have the same shape'
    assert params[params_idx]['b'].shape == raw_b.shape, 'biases do not have the same shape'
    params[params_idx]['w'] = raw_w
    params[params_idx]['b'] = raw_b

img_folder = "../results/hand_draw_img"
image_files = [f'handraw_{i}.png' for i in range(10)]

images = []
for image_file in image_files:
    img_path = os.path.join(img_folder, image_file)
    if os.path.exists(img_path):
        images.append(Image.open(img_path))
    else:
        print(f"Image {img_path} not found.")

# Invert colors and preprocess images
invert_img = []
for img in images:
    gr_img = ImageOps.grayscale(img)  # Convert to grayscale
    invert_img.append(ImageOps.invert(gr_img))  # Invert colors

# Resize images
for i in range(len(invert_img)):
    invert_img[i] = invert_img[i].resize((28, 28), Image.LANCZOS)

img_arr = []
for img in invert_img:
    img_arr.append((np.asarray(img)) / 255)  # Normalize between 0 and 1

# Save images from img_arr back into PNG format in ../results/hand_draw_img_result
output_folder = "../results/hand_draw_img_result"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for i, img in enumerate(img_arr):
    output_path = os.path.join(output_folder, f'processed_handraw_{i}.png')
    img_to_save = Image.fromarray(np.uint8(img * 255))  # Convert to uint8 format
    img_to_save.save(output_path)
print("Saved processed_handraw images")

# Set the batch size for the network
layers[0]['batch_size'] = 1

# predictions
my_pred = []
for img in img_arr:
    input_data = img.reshape((-1, 28 * 28))  # Reshape for a single sample
    cptest, P = convnet_forward(params, layers, input_data, test=True)
    my_pred.append(P)

# Extract predicted classes
my_pred_val = [np.argmax(pred) for pred in my_pred]

# Output results
print("Ground Truth: ")
print("[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]") 

print("Prediction: ")
print(my_pred_val)
