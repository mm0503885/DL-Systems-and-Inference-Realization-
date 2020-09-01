from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

import torch
import torchvision
import torchvision.transforms as transforms

import time
BATCH_SIZEs = [1, 2, 4, 8, 16, 32, 64]
latencies = []
FPSs = []

import matplotlib.pyplot as plt

TRT_LOGGER = trt.Logger()

data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

test_set = torchvision.datasets.ImageFolder(root='food11/evaluation/',transform=data_transforms)

def argmax(lst):
    return max(range(len(lst)), key=lst.__getitem__)

def get_engine(onnx_file_path, engine_file_path="", batch_size=1):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30 # 256MiB
            builder.max_batch_size = batch_size
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [batch_size, 3, 224, 224]
            #network.get_input.shape = [batch_size, 3, 224, 224]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def main():
    for bs in BATCH_SIZEs:
        onnx_file_path = 'lab5_model.onnx'
        engine_file_path = ("model%d.trt" % (bs))
        # Do inference with TensorRT
   
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False)
        trt_outputs = []
        with get_engine(onnx_file_path, engine_file_path, bs) as engine, engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            # Do inference
            print('Running inference ')
            print('Batch size: %d' %(bs))
            right = 0
            number = 0
            start_time = time.time()
            # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
            for i, (images, labels) in enumerate(test_loader, 0):
                images = images.numpy()
                inputs[0].host = images
                [results] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                for j in range(len(labels)):
                    result = results[j*11:(j+1)*11]
                    pred = argmax(result)
                    if(pred == labels[j]):
                        right += 1
                    number += 1
            latency = time.time() - start_time
            print('Time elapsed: %.4f' %(latency))
            print(right / number)
            latencies.append(latency)
            FPSs.append(len(test_set)/(latency))
    
    plt.subplot(2, 1, 1)
    plt.plot(BATCH_SIZEs, latencies)
    plt.ylabel('latency (s)')
    plt.subplot(2, 1, 2)
    plt.plot(BATCH_SIZEs, FPSs)
    plt.xlabel('batch size')
    plt.ylabel('FPS')
    plt.savefig('./test.png')

if __name__ == '__main__':
    main()
