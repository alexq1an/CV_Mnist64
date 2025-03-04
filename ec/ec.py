import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(1, '../python/')
from utils import get_lenet
from init_convnet import init_convnet
from conv_net import convnet_forward
from scipy.io import loadmat

# Load trained network
layers = get_lenet()
params = init_convnet(layers)
layers[0]['batch_size'] = 1

# Load trained network weights
data = loadmat('../results/lenet.mat')
params_raw = data['params']

for params_idx in range(len(params)):
    raw_w = params_raw[0, params_idx][0, 0][0]
    raw_b = params_raw[0, params_idx][0, 0][1]
    assert params[params_idx]['w'].shape == raw_w.shape, 'weights do not have the same shape'
    assert params[params_idx]['b'].shape == raw_b.shape, 'biases do not have the same shape'
    params[params_idx]['w'] = raw_w
    params[params_idx]['b'] = raw_b

# Function to preprocess image for neural network input
def preprocess_image(img):
    img = cv2.resize(img, (28, 28))
    img = img.astype('float32') / 255
    img = img.reshape((28 * 28, 1))
    return img


# Function to recognize digits in a real-world image
def recognize_digits(image_path, name):
    # Read and convert image to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    
    # find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours left-to-right
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    digit_images = []  # List to store digit images
    predicted_labels = []  # List to store predicted labels
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # skip the digit that is way too small(noise reduction)
        if w < 7 or h < 7:
            continue

        digit = img[y:y + h, x:x + w]

        # save image before invert it
        digit_images.append(digit)

        # Invert the image colors
        digit = cv2.bitwise_not(digit)

        # threshold
        _, digit = cv2.threshold(digit, 100, 255, cv2.THRESH_TOZERO)
        
        # Pad and resize the digit image to 28x28
        pad_size = max(w, h)
        padded_digit = np.zeros((pad_size, pad_size), dtype='uint8')
        padded_digit[(pad_size - h) // 2:(pad_size - h) // 2 + h,
                     (pad_size - w) // 2:(pad_size - w) // 2 + w] = digit
        
        # leave the edge of the image blank for better prediction
        padded_digit = np.pad(padded_digit, pad_width=max(w, h) // 4, mode='constant', constant_values=0)

        kernel = np.ones((3, 3), np.uint8)
        padded_digit = cv2.dilate(padded_digit, kernel, iterations=2)

        # Apply Gaussian Blur for bold digits
        padded_digit = cv2.GaussianBlur(padded_digit, (3, 3), 0.2)

        # Preprocess the digit for the network
        input_data = preprocess_image(padded_digit)
        
        # Run through the neural network
        cptest, P = convnet_forward(params, layers, input_data, test=True)

        predicted_label = np.argmax(P, axis=0)

        # Store prediction
        predicted_labels.append(predicted_label)
        
        # # Draw bounding box and prediction on the original image
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.putText(img, str(predicted_label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

    # Create subplots for digit images and predictions
    num_digits= int(np.sqrt(len(digit_images)))
    fig, axes = plt.subplots(num_digits, num_digits+2, figsize=(15, 10))

    # prepare the image for
    for i, ax in enumerate(axes.flat):
        if i < len(digit_images):  # Plot only for available images
            ax.imshow(digit_images[i], cmap='gray')
            ax.set_title(predicted_labels[i])
        else:
            ax.axis('off')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'../results/{name}.png')

# Path to the test image
image_path = '../images/image4.jpg'

# Recognize digits in the test image
recognize_digits('../images/image1.jpg', 'recognition_image1')
recognize_digits('../images/image2.jpg', 'recognition_image2')
recognize_digits('../images/image3.png', 'recognition_image3')
recognize_digits('../images/image4.jpg', 'recognition_image4')