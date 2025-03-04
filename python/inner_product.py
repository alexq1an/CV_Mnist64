import numpy as np


def inner_product_forward(input, layer, param):
    """
    Forward pass of inner product layer.

    Parameters:
    - input (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """

    d, k = input["data"].shape # 25, 2
    n = param["w"].shape[1] # 25

    ###### Fill in the code here ######

    data = np.zeros((n, k)) # 25,2

    for i in range(k):
        # data[:,i] = np.dot(input["data"][:,i], param["w"].T) + param["b"]
        data[:,i] = np.dot(param["w"].T, input["data"][:,i])
        data[:,i] += param["b"][0]

    # Initialize output data structure
    output = {
        "height": n,
        "width": 1,
        "channel": 1,
        "batch_size": k,
        "data": data # replace 'data' value with your implementation
    }

    return output


def inner_product_backward(output, input_data, layer, param):
    """
    Backward pass of inner product layer.

    Parameters:
    - output (dict): Contains the output data.
    - input_data (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """
    param_grad = {}
    ###### Fill in the code here ######
    # Replace the following lines with your implementation.
    param_grad['b'] = np.zeros_like(param['b'])
    param_grad['w'] = np.zeros_like(param['w'])
    input_od = None

    input_od = np.dot(param['w'], output['diff'])
    param_grad['b'] = np.sum(output['diff'], axis=1, keepdims=True).T
    param_grad['w'] = np.dot(input_data['data'], output['diff'].T)
    



    return param_grad, input_od