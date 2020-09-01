import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class NestDepthConv(nn.Module):

    def __init__(self, C_in, kernel_size=3, stride=1, padding=1):
        super(NestDepthConv, self).__init__()

        self.C_in = C_in
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(
            self.C_in, self.C_in, self.kernel_size, self.stride, groups=self.C_in,
            bias=False
        )

        scale_param = {}
        param_name = 'transf_matrix'
        scale_param[param_name] = Parameter(torch.eye(self.kernel_size**2))
        self.register_parameter(param_name, scale_param[param_name])

    def get_act_filter(self, has_residual):
        filters = self.conv.weight[:, :, :, :]

        if has_residual:
            _input_filter = self.conv.weight[:, :, :, :]
            _input_filter = _input_filter.contiguous()
            _input_filter = _input_filter.view(_input_filter.size(0), _input_filter.size(1), -1)
            _input_filter = _input_filter.view(-1, _input_filter.size(2))
            _input_filter = F.linear(
                _input_filter, self.__getattr__('transf_matrix')
            )
            _input_filter = _input_filter.view(filters.size(0), filters.size(1),
                                                self.kernel_size**2)
            _input_filter = _input_filter.view(filters.size(0), filters.size(1),
                                                self.kernel_size, self.kernel_size)
            filters = _input_filter
        return filters

    def forward(self, x, has_residual=False):
        filters = self.get_act_filter(has_residual).contiguous()

        y = F.conv2d(
            x, filters, None, self.stride, self.padding, groups=self.C_in
        )
        return y
