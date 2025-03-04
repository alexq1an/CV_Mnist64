import numpy as np

def relu_forward(input_data):
    output = {
        'height': input_data['height'],
        'width': input_data['width'],
        'channel': input_data['channel'],
        'batch_size': input_data['batch_size'],
    }

    ###### Fill in the code here ######
    # Replace the following line with your implementation.
    # print("output['data'].shape",output['data'].shape)

    output['data'] = np.zeros((input_data['channel'], input_data['height'], input_data['width'], input_data['batch_size']))

    input_reshape = input_data['data'].reshape(((input_data['channel'], input_data['height'], input_data['width'], input_data['batch_size'])))
    for batch in range(output['batch_size']):
        for c in range(output['channel']):
            for h in range(output['height']):      
                for w in range(output['width']):                    
                    output['data'][c,h,w,batch] = np.maximum(0,input_reshape[c,h,w,batch])

    output['data'] = output['data'].reshape((input_data['channel']*input_data['height']*input_data['width'],input_data['batch_size']))
    return output

def relu_backward(output, input_data, layer):
    ###### Fill in the code here ######
    # Replace the following line with your implementation.

    diff_re = np.where(input_data['data'] >= 0, 1, 0)
    input_od = np.multiply(output['diff'], diff_re)


    return input_od
