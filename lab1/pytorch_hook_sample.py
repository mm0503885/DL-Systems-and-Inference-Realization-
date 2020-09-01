import torch
import torch.nn as nn
import torch.nn.functional as F

class SampleNet(nn.Module):
    def __init__(self):
        super(SampleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        
    def forward(self, input):
        x = F.relu(self.conv1(input))
        return x
    
def my_hook_function(self, input, output):
    print("Op:{}".format(str(self.__class__.__name__)))
    for param in self.parameters():
        print("params shape: {}".format(list(param.size())))
        
def main():
    model = SampleNet()
    model.conv1.register_forward_hook(my_hook_function)
    input_data = torch.randn(1, 3, 224, 224)
    out = model(input_data)

if __name__ == '__main__':
    main()

