import os
import subprocess
import numpy as np
import cv2

def tf_image_preprocess(input_image, output_image, image_dim):
    r""" Preprocess image for TenforFlow inference 
    Parameters
    ----------
    input_image:  str
        Input coded image file
    output_image: str
        Output raw image file name
    image_dim: list
        Image dimension: HxWxC
    """

    # get image height and weight
    output_image_size = image_dim[0:2]
    #if input_layout == 'HWCN' or input_layout == 'HWNC':
    #    output_image_size = image_dim[0:2]
    #elif input_layout == 'NCHW':
    #    output_image_size = image_dim[2:4]
    #elif input_layout == 'NHWC':
    #    output_image_size = image_dim[1:3]
    #else:
    #    print('Input layout is not supported')
    #    return
  
    image_BGR = cv2.imread(filename = input_image)
    
    # convert to RGB format - openCV reads image as BGR format
    image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
  
    # crop and resize image 
    orig_H = image.shape[0]
    orig_W = image.shape[1]
    factor = 0.875
    crop_W = orig_W * factor
    crop_H = orig_H * factor
    half_W = orig_W/2
    half_H = orig_H/2
    start_x = half_H - crop_H/2
    start_y = half_W - crop_W/2
    end_x = start_x + crop_H
    end_y = start_y + crop_W
    x0 = round(start_x)
    x1 = round(end_x)
    y0 = round(start_y)
    y1 = round(end_y)
    cropped_image = image[x0:x1,y0:y1]
    resized_image = cv2.resize(cropped_image, output_image_size, interpolation = cv2.INTER_AREA)

    # serialize data to be written to a file
    r,g,b = cv2.split(resized_image)
    total_per_plane = output_image_size[0]*output_image_size[1];
    rr = r.reshape(1,total_per_plane)
    rg = g.reshape(1,total_per_plane)
    rb = b.reshape(1,total_per_plane)
    y = np.hstack((rr,rg,rb))
    
    # subtract all pixels by 128 -> convert to int8 
    mean = np.full(y.shape,128)
    y = y.astype(np.int32)
    mean = mean.astype(np.int32)
    y_sub_mean = cv2.subtract(y,mean)
    np.clip(y_sub_mean,-128,127)
    y_sub_mean.astype('int8').tofile(output_image)

def onnx_image_preprocess(input_image, output_image, output_dim):
    r""" Preprocess image for TenforFlow inference 
    Parameters
    ----------
    input_image:  str
        Input coded image file
    output_image: str
        Output raw image file name
    image_dim: list
        Image dimension        
    """

    # read image and convert to RGB format - openCV reads image as BGR format
    image_BGR = cv2.imread(filename = input_image)   
    image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)

    output_image_size = output_dim[0:2] # get image height and weight

    resized_image = cv2.resize(image, output_image_size, interpolation = cv2.INTER_AREA)

    # normalize image according to ONNX spec
    r,g,b = cv2.split(resized_image)
    total_per_plane = output_image_size[0]*output_image_size[1];
    rr = r.reshape(1,total_per_plane)
    rg = g.reshape(1,total_per_plane)
    rb = b.reshape(1,total_per_plane)
    mean_r = np.full(rr.shape, 123.675)
    mean_g = np.full(rg.shape, 116.28)
    mean_b = np.full(rb.shape, 103.53)
    rr = rr.astype(np.float)
    rg = rg.astype(np.float)
    rb = rb.astype(np.float)
    rr_sub_mean = cv2.subtract(rr,mean_r)
    rg_sub_mean = cv2.subtract(rg,mean_g)
    rb_sub_mean = cv2.subtract(rb,mean_b)
    rr_norm = rr_sub_mean * 0.82381
    rg_norm = rg_sub_mean * 0.84220
    rb_norm = rb_sub_mean * 0.83846

    y = np.hstack((rr_norm, rg_norm, rb_norm))
    np.clip(y, -128,127)
    y.astype('int8').tofile(output_image);

def float_nd_to_int8_1d(input):
    r""" Convert float32 nd array to int8 1d array
    Parameters
    ----------
    input: N dimension float32 array in "NCHW" format
    output: 1 dimension int8 array HxWxCxN
    """

    scale = 128.0/max(abs(np.amin(input)), np.amax(input))
    print('Input quantization scaling factor: ')
    print(scale)
    y = np.multiply(input, scale)
    z = np.rint(y)
    z = np.clip(z,-128,127)
    output = z.flatten()

    return output, scale

def tensor_quant(input_tensor):
    r""" Convert float32 n-d array to int8 or uint8 1-d array
    Parameters
    ----------
    input: float32 array in "CxHxW" format
    output: 1 dimension int8 or uint8 array
    """

    if np.amin(input_tensor) >= 0:
        # quantize to Uint8
        sign  = 0
        scale = 255.0/np.amax(input_tensor)
        quant_min = 0
        quant_max = 255
    else:
        # quantize to Int8
        sign  = 1
        scale = 128.0/max(abs(np.amin(input_tensor)), np.amax(input_tensor))
        quant_min = -128
        quant_max = 127
        
    y = np.multiply(input_tensor, scale)
    z = np.rint(y)
    z = np.clip(z, quant_min, quant_max)
    output = z.flatten()

    return output, scale, sign
