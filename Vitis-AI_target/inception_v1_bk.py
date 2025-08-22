"""
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import pathlib
import xir
import os
import math
import threading
import time
import sys
import cProfile

"""
Calculate softmax
data: data to be calculated
size: data size
return: softamx result
"""


def CPUCalcSoftmax(data, size, scale):
    sum = 0.0
    result = [0 for i in range(size)]
    for i in range(size):
        result[i] = math.exp(data[i] * scale)
        sum += result[i]
    for i in range(size):
        result[i] /= sum
    return result


def get_script_directory():
    path = os.getcwd()
    return path


"""
Get topk results according to its probability
datain: data result of softmax
filePath: filePath in witch that records the infotmation of kinds
"""

def TopK(datain, size, filePath):

    cnt = [i for i in range(size)]
    pair = zip(datain, cnt)
    pair = sorted(pair, reverse=True)
    softmax_new, cnt_new = zip(*pair)
    fp = open(filePath, "r")
    data1 = fp.readlines()
    fp.close()
    for i in range(5):
        flag = 0
        for line in data1:
            if (flag+1) == cnt_new[i]:
                print("Top[%d] %d %s" % (i, flag, (line.strip)("\n")))
            flag = flag + 1


"""
pre-process for resnet50 (caffe)
"""
_B_MEAN = 127.5
_G_MEAN = 127.5
_R_MEAN = 127.5
MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]
SCALES = [0.007843137, 0.007843137, 0.007843137]

def resize_shortest_edge(image, size):
    H, W = image.shape[:2]
    if H >= W:
        nW = size
        nH = int(float(H)/W * size)
    else:
        nH = size
        nW = int(float(W)/H * size)
    #print("nW:nH=",nW,nH)
    return cv2.resize(image,(nW,nH))
def central_crop(image, crop_height, crop_width):
    image_height = image.shape[0]
    image_width = image.shape[1]
    offset_height = (image_height - crop_height) // 2
    offset_width = (image_width - crop_width) // 2
    return image[offset_height:offset_height + crop_height, offset_width:offset_width + crop_width, :]

def preprocess_one_image_fn(image_path, fix_scale, width=224, height=224):
    means = MEANS
    scales = SCALES
    image = cv2.imread(image_path)
    #image = cv2.resize(image, (width, height))
    image = resize_shortest_edge(image,256)
    image = central_crop(image, height, width)
    B, G, R = cv2.split(image)
    B = (B - means[0]) * scales[0] * fix_scale
    G = (G - means[1]) * scales[1] * fix_scale
    R = (R - means[2]) * scales[2] * fix_scale
    #image = cv2.merge([B, G, R])
    image = cv2.merge([R, G, B])
    image = image.astype(np.int8)
    return image


SCRIPT_DIR = get_script_directory()
calib_image_dir = SCRIPT_DIR + "/../images/"

global threadnum
threadnum = 0

"""
run inception_v1 with batch
dpu: dpu runner
img: imagelist to be run
cnt: threadnum
"""


def runInceptionV1(dpu: "Runner", img, cnt):
    """get tensor"""
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    shapeIn = tuple(inputTensors[0].dims)
    shapeOut = tuple(outputTensors[0].dims)
    pre_output_size = int(outputTensors[0].get_data_size() / shapeIn[0])

    output_fixpos = outputTensors[0].get_attr("fix_point")
    output_scale = 1 / (2**output_fixpos)
    count = 0
    n_of_images = len(img)
    while count < cnt:
        runSize = shapeIn[0]
        """prepare batch input/output """
        outputData = [np.empty(shapeOut, dtype=np.int8, order="C")]
        inputData = [np.empty(shapeIn, dtype=np.int8, order="C")]
        """init input image to input buffer """
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(shapeIn[1:])
        """run with batch """
        job_id = dpu.execute_async(inputData, outputData)
        dpu.wait(job_id)
        """softmax calculate with batch """
        """Benchmark DPU FPS performance over Vitis AI APIs execute_async() and wait() """
        """Uncomment the following code snippet to include softmax calculation for model’s end-to-end FPS evaluation """
        #for j in range(runSize):
        #    softmax = CPUCalcSoftmax(outputData[0][j], pre_output_size, output_scale)
        #    TopK(softmax, pre_output_size, "./words.txt")

        count = count + runSize

def float_to_q2_5_int8(float_val):
    """
    Converts a floating-point number (or array of numbers) to its Q2.5 fixed-point
    representation, returned as an 8-bit signed integer (np.int8).

    Q2.5 format: 1 sign bit, 2 integer bits, 5 fractional bits = 8 bits total.
    Range for the floating-point input: approximately [-4.0, 3.96875].
    The scaling factor used is 2^5 = 32.

    Args:
        float_val (float or np.ndarray): The floating-point number(s) to convert.
                                        If an array, elements outside the Q2.5
                                        float range [-4, 3.96875] will be clamped.

    Returns:
        np.int8 or np.ndarray of np.int8: The Q2.5 fixed-point representation.
    """
    n_integer_bits = 2 # m
    n_fractional_bits = 5 # n

    scale = 2**n_fractional_bits  # 2^5 = 32

    # Ensure input is a NumPy array for consistent handling of scalars or lists
    float_array = np.array(float_val, dtype=np.float32)

    # 1. Multiply by the scale factor to shift the fractional part into integer range
    scaled_value = float_array * scale

    # 2. Round to the nearest integer
    # np.round handles rounding correctly for positive and negative numbers
    rounded_value = np.round(scaled_value)

    # 3. Clamp the rounded integer value to the valid range for an 8-bit signed integer [-128, 127]
    # This also implicitly handles clamping for values outside the Q2.5 float range.
    # For example, a float 5.0 (outside Q2.5 max 3.96875) would be scaled to 5*32=160,
    # then clamped to 127.
    int8_value = np.clip(rounded_value, -128, 127).astype(np.int8)

    return int8_value

def run_keras_dnn(dpu: "Runner", img, choosen):
    """get tensor"""
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    shapeIn = tuple(inputTensors[0].dims)
    shapeOut = tuple(outputTensors[0].dims)
    # for inputTensor in inputTensors:
    #     print(inputTensor.name)
    #     print(inputTensor.dims)
    #     print(inputTensor.dtype)
    # for outputTensor in outputTensors:
    #     print(outputTensor.name)
    #     print(outputTensor.dims)
    #     print(outputTensor.dtype)
    pre_output_size = int(outputTensors[0].get_data_size() / shapeIn[0])

    output_fixpos = outputTensors[0].get_attr("fix_point")
    output_scale = 1 / (2**output_fixpos)
    count = 0
    score = 0
    n_of_images = len(img)

    data = [
        [-0.51121936,  0.47090117,  0.38540931,  0.07026399,
        -0.04159453, -0.1419771,  -0.86861754, -0.55332437,
        -0.86861754, -0.78398404, -0.80951885, -0.84826608,
        -0.96489331, -0.8356734,   0.18723389,  0.46561524],
        [0.7483717,  -0.49861702,  0.12397022,  0.03641154,  
            0.56179779,  0.06696305,  0.72380229, -0.09968861,
            0.72380229, -0.51549956, -0.65733668, -0.50218422,
        -0.19640946, -0.43477071,  0.26307061, -0.87207752]
    ]

    fixed_data = [
        [0b10010011,  0b00001111,  0b00001100,  0b00000010,
        0b10000001,  0b10000100,  0b10011011,  0b10010100,
        0b10011011,  0b10011001,  0b10011001,  0b10011011,
        0b10011110,  0b10011010,  0b10000101,  0b00001111],
        [0b00011000, 0b10001111,  0b00000100,  0b00000001,
        0b00010100,  0b00000001,  0b00011001,  0b10000011,
        0b00011000,  0b10010011,  0b10010111,  0b10010000,
        0b10000101,  0b10001110,  0b00001000,  0b10011011],
    ]

    X_train_val = np.load('X_train_val.npy')
    y_train_val = np.load('y_train_val.npy')

    if choosen == 2:
        X_train_val = np.load('X_test.npy')
        y_train_val = np.load('y_test.npy')

    # print(X_train_val[0:2])
    # print(y_train_val[0:2])

    int8_value = float_to_q2_5_int8(X_train_val)
    # print(int8_value[0:2])

    while count < 100:
        runSize = shapeIn[0]
        """prepare batch input/output """
        outputData = [np.empty(shapeOut, dtype=np.int8, order="C")]
        inputData = [np.empty(shapeIn, dtype=np.int8, order="C")]
        """init input image to input buffer """
        # Create an int8 array to store the bits
        # Define the data as a Python list

        # Create the NumPy array with the specified shape (1, 16)
        # NumPy will automatically infer the dtype as float64 since the data contains floats.
        my_array = []
        my_array = np.array([int8_value[count]], dtype=np.int8, order="C") # Using float32 for potentially smaller memory footprint
        
        inputData = [my_array]

        # for j in range(runSize):
        #     imageRun = inputData[0]
        #     imageRun[j, ...] = img[(count + j) % n_of_images].reshape(shapeIn[1:])
        
        """run with batch """
        job_id = dpu.execute_async(inputData, outputData)
        dpu.wait(job_id)
        """softmax calculate with batch """
        """Benchmark DPU FPS performance over Vitis AI APIs execute_async() and wait() """
        """Uncomment the following code snippet to include softmax calculation for model’s end-to-end FPS evaluation """
        # print(outputData)

        if np.argmax(outputData) == np.argmax(y_train_val[count]):
            score += 1
        #for j in range(runSize):
        #    softmax = CPUCalcSoftmax(outputData[0][j], pre_output_size, output_scale)
        #    TopK(softmax, pre_output_size, "./words.txt")

        count = count + 1
    #print("%.2f accuracy" % score)


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (
        root_subgraph is not None
    ), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

list_record = []


def main(argv):
    global threadnum

    """create runner """

    # listimage = os.listdir(calib_image_dir)
    threadAll = []
    threadnum = int(argv[1])
    i = 0
    # global runTotall
    # runTotall = len(listimage)
    g = xir.Graph.deserialize(argv[2])
    subgraphs = get_child_subgraph_dpu(g)
    assert len(subgraphs) == 1  # only one DPU kernel

    choosen = int(argv[3])

    all_dpu_runners = []
    for i in range(int(threadnum)):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))
    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    output_fixpos = all_dpu_runners[0].get_output_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos
    # print(input_fixpos, output_fixpos)
    
    """ tensor list to be run """
    tensor16 = [0x01e2fdf4, 0x0047018a, 0xff6effd5, 0xfdc9fc86, 0xfcddfc86, 0xfc9bfcc3, 0xfca8fc23, 0x01dc00bf]
    
    """
      The cnt variable is used to control the number of times a single-thread DPU runs.
      Users can modify the value according to actual needs. It is not recommended to use
      too small number when there are few input images, for example:
      1. If users can only provide very few images, e.g. only 1 image, they should set
         a relatively large number such as 360 to measure the average performance;
      2. If users provide a huge dataset, e.g. 50000 images in the directory, they can
         use the variable to control the test time, and no need to run the whole dataset.
    """
    cnt = 360
    """run with batch """
    time_start = time.time()
    # for i in range(int(threadnum)):
    #     t1 = threading.Thread(
    #         target=run_keras_dnn, args=(all_dpu_runners[i], tensor16, choosen)
    #     )
    #     threadAll.append(t1)
    # for x in threadAll:
    #     x.start()
    # for x in threadAll:
    #     x.join()
    run_keras_dnn(all_dpu_runners[i], tensor16, choosen);

    del all_dpu_runners
    time_end = time.time()
    total = cnt * int(threadnum)
    timetotal = time_end - time_start
    list_record.append(timetotal)
    fps = float(total / timetotal)
    #print("%.2f VPS" % fps)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage : python3 inception_v1.py <thread_number> <inception_v1_model_file> <train: 1/test: 2>")
    else:
        #main(sys.argv)
        for i in range(100):
            #cProfile.run("main(sys.argv)")
            main(sys.argv)
        with open('output.txt', 'w') as file:
            for item in list_record:
                file.write(str(item) + '\n')
