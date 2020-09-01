
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
total_params=0
total_MACs=0
total_FLOPs=0

def my_hook_function(self, input, output):
    global total_params,total_MACs,total_FLOPs
    if(str(self.__class__.__name__)=="Conv2d"):
        params = sum(p.numel() for p in self.parameters())
        MACs = list(output.size())[2]*list(output.size())[3]*self.out_channels*self.kernel_size[0]*self.kernel_size[1]*self.in_channels
        FLOPs = 2*list(input[0].size())[2]*list(input[0].size())[3]*(self.in_channels*self.kernel_size[0]*self.kernel_size[1]+1)*self.out_channels
        print("%s %s %s %d %d %d" % (str(self.__class__.__name__), list(input[0].size()), list(output.size()), params, MACs ,FLOPs))
        total_params+=params
        total_MACs+=MACs
        total_FLOPs+=FLOPs
    elif(str(self.__class__.__name__)=="Linear"):
        params = sum(p.numel() for p in self.parameters())
        MACs = list(input[0].size())[1]*list(output.size())[1]
        FLOPs = (2*list(input[0].size())[1]-1)*list(output.size())[1]
        print("%s %s %s %d %d %d" % (str(self.__class__.__name__), list(input[0].size()), list(output.size()), params, MACs ,FLOPs))
        total_params+=params
        total_MACs+=MACs
        total_FLOPs+=FLOPs        
def main():
    global total_params,total_MACs,total_FLOPs
    print("resnet50:\n")
    print("%s | %s | %s | %s | %s | %s" % ("op_type", "input_shape", "output_shape", "params", "MACs", "FLOPs"))
    print("--------|-------------|--------------|--------|------|------")
    model = models.resnet50()
    modules = model.named_modules() # 
    for name, module in modules:
        module.register_forward_hook(my_hook_function)
    input_data = torch.randn(1, 3, 224, 224)
    out = model(input_data)
    print("\nTotal_params: %.2fM" %(total_params/(1000 ** 2)))
    print("Total_MACs: %.2fG" %(total_MACs/(1000 ** 3)))
    print("Total_FLOPs: %.2fG\n" %(total_FLOPs/(1000 ** 3)))
    total_params=0
    total_MACs=0
    total_FLOPs=0
    print("\nmobilenet_v2:\n")
    print("%s | %s | %s | %s | %s | %s" % ("op_type", "input_shape", "output_shape", "params", "MACs", "FLOPs"))
    print("--------|-------------|--------------|--------|------|------")
    model2 = models.mobilenet_v2()
    modules2 = model2.named_modules() # 
    for name, module in modules2:
        module.register_forward_hook(my_hook_function)
    input_data2 = torch.randn(1, 3, 224, 224)
    out2 = model2(input_data2)
    print("\nTotal_params: %.2fM" %(total_params/(1000 ** 2)))
    print("Total_MACs: %.2fG" %(total_MACs/(1000 ** 3)))
    print("Total_FLOPs: %.2fG\n" %(total_FLOPs/(1000 ** 3)))
if __name__ == '__main__':
    main()






