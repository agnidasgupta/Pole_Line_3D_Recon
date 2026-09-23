"""Opt-in CUDA inference Conv3d input cast/layout copy fusion."""
import torch
from torch import nn

class InputLayoutConv3d(nn.Conv3d):
    def forward(self,x):
        if (not self.training and not torch.is_grad_enabled() and x.is_cuda
                and x.ndim==5 and x.dtype==torch.float32
                and torch.is_autocast_enabled()
                and torch.get_autocast_gpu_dtype()==torch.bfloat16
                and x.is_contiguous() and not x.is_contiguous(memory_format=torch.channels_last_3d)
                and self.weight.is_contiguous(memory_format=torch.channels_last_3d)):
            x=x.to(dtype=torch.bfloat16,memory_format=torch.channels_last_3d)
        return super().forward(x)

def enable_conv_input_layout(model):
    if model.training:raise ValueError('Conv input layout optimization requires model.eval()')
    count=0
    for name,module in list(model.named_children()):
        if type(module) is nn.Conv3d:
            new=InputLayoutConv3d(module.in_channels,module.out_channels,module.kernel_size,
                stride=module.stride,padding=module.padding,dilation=module.dilation,
                groups=module.groups,bias=module.bias is not None,padding_mode=module.padding_mode)
            new.weight=module.weight;new.bias=module.bias;new.eval();setattr(model,name,new);count+=1
        else:count+=enable_conv_input_layout(module)
    return count
