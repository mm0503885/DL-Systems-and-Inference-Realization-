import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torchvision.models as models

device = 'cpu'
print(device)

torch_model = models.resnet34(pretrained=True)
num_ftrs = torch_model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
torch_model.fc = nn.Linear(num_ftrs, 11)
torch_model = torch_model.to(device)
torch_model.load_state_dict(torch.load('lab4_model.pht'))
torch_model.eval()

# Input to the model
x = torch.randn(4, 3, 224, 224, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "lab4_model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})

import onnx

onnx_model = onnx.load("lab4_model.onnx")
onnx.checker.check_model(onnx_model)


