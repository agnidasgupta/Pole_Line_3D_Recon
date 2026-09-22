"""Opt-in inference-only GroupNorm input copy optimization.

Keep native FP32 GroupNorm and its reduction order. Combine the autocast's
BF16/FP16 -> FP32 cast and native GroupNorm's NCDHW contiguous copy into one
Tensor.to operation. Parameters, state-dict names and normalization math stay
unchanged. This is validated on Torch 2.4.1/H100; production weights remain
required for deployment acceptance.
"""
import torch
from torch import nn


class InputLayoutGroupNorm(nn.GroupNorm):
    def forward(self, x):
        if (not self.training and not torch.is_grad_enabled()
                and x.is_cuda and x.ndim == 5
                and torch.is_autocast_enabled()
                and x.dtype in (torch.bfloat16, torch.float16)
                and x.is_contiguous(memory_format=torch.channels_last_3d)
                and not x.is_contiguous()
                and (self.weight is None or self.weight.dtype == torch.float32)
                and (self.bias is None or self.bias.dtype == torch.float32)):
            x = x.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        return super().forward(x)


def enable_groupnorm_input_layout(model):
    """Explicitly opt an evaluation model in; no global monkey-patching."""
    if model.training:
        raise ValueError('GroupNorm layout optimization requires model.eval()')
    count = 0
    for name, module in list(model.named_children()):
        if type(module) is nn.GroupNorm:
            replacement = InputLayoutGroupNorm(module.num_groups, module.num_channels,
                                               eps=module.eps, affine=module.affine)
            replacement.weight = module.weight
            replacement.bias = module.bias
            replacement.eval()
            setattr(model, name, replacement)
            count += 1
        else:
            count += enable_groupnorm_input_layout(module)
    return count
