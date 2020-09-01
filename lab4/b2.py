from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore
from os import listdir
from os.path import join, splitext, basename, isfile, isdir
import PIL
from PIL import Image
import time

Image.MAX_IMAGE_PIXELS = 1000000000

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    args.add_argument("-i", "--input", help="Required. Customized. Input should be directory under food11/food11re/",
                      required=True,
                      type=str, nargs="+")
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                           " kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)

    return parser


def main():
    start_all_time = time.time()
    # ======================================
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    #log.info("Creating Inference Engine")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")

    # Read IR
    #log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = ie.read_network(model=model_xml, weights=model_bin)

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    #log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    #net.batch_size = len(args.input)
    net.batch_size = 1

    # Loading model to the plugin
    #log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)
    root_folder = args.input[0]
    correct = 0
    number = 0

    # ======================================
    overhead_time = int((time.time() - start_all_time) * 1000)
    #print(overhead_time)
    loading_and_preprocessing_time = 0
    inference_time = 0
    total_latency_start = time.time()

    for folder in listdir(root_folder):
        for file_name in listdir(root_folder + folder):
            
            # Read and pre-process input images
            n, c, h, w = net.inputs[input_blob].shape
            images = np.ndarray(shape=(n, c, h, w))

            start_time = time.time()
            label = int(folder)

            #image = cv2.imread(root_folder + folder + '/' + file_name)
            #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = Image.open(root_folder + folder + '/' + file_name)
            
            if image.size[:-1] != (h, w):
                #log.warning("Image {} is resized from {} to {}".format(file_name, image.shape[:-1], (h, w)))
                #image = cv2.resize(image, (w, h))
                image = image.resize((h, w), Image.BILINEAR)
            
            image = np.array(image)
            image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            image = image - 127.5
            image = image / 127.5
            images[0] = image
            loading_and_preprocessing_time += int((time.time() - start_time) * 1000)

            start_time = time.time()
            res = exec_net.infer(inputs={input_blob: images})
            inference_time += int((time.time() - start_time) * 1000)

            res = res[out_blob]
            
            probs = np.squeeze(res)
            top_ind = np.argsort(probs)[-1:][::-1]
            
            if(int(top_ind[0]) == label):
                correct += 1
            number += 1
            
    total_latency = int((time.time() - total_latency_start) * 1000)
    total_time = int(time.time() - start_all_time)
    print('Accuracy: %.3f, (%d/ %d)' % (correct/number, correct, number))
    print('Total execution time: %d s' % total_time)
    print('Overhead: %d ms' % overhead_time)
    print('Loading and preprocessing average time: %.3f ms' % (loading_and_preprocessing_time/number))
    print('Inference average time: %.3f ms' % (inference_time/number))
    print('average lantency: %.3f ms' % (total_latency/number))
    print('FPS without overhead: %.3f' % (1000 / (total_latency/number)))
    print('FPS with overhead: %.3f' % (number/total_time))
    

if __name__ == '__main__':
    sys.exit(main() or 0)