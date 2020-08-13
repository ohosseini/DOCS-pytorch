import torch
from torch.nn.modules.module import Module
from torch.autograd import Function
import correlation_cuda

class CorrelationFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2, param_dict):
        ctx.save_for_backward(input1, input2)

        ctx.pad_size = param_dict["pad_size"]
        ctx.kernel_size = param_dict["kernel_size"]
        ctx.max_disp = param_dict["max_disp"]
        ctx.stride1 = param_dict["stride1"]
        ctx.stride2 = param_dict["stride2"]
        ctx.corr_multiply = param_dict["corr_multiply"]

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()

            correlation_cuda.forward(input1, input2, rbot1, rbot2, output,
                ctx.pad_size, ctx.kernel_size, ctx.max_disp, ctx.stride1, ctx.stride2, ctx.corr_multiply)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
                ctx.pad_size, ctx.kernel_size, ctx.max_disp, ctx.stride1, ctx.stride2, ctx.corr_multiply)

        return grad_input1, grad_input2, None

class Correlation(Module):
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply
        self.out_channels = ((max_displacement/stride2)*2 + 1) * ((max_displacement/stride2)*2 + 1)

    def forward(self, input1, input2):
        param_dict = {'pad_size':self.pad_size, 'kernel_size':self.kernel_size, 
                      'max_disp':self.max_displacement, 'stride1':self.stride1, 
                      'stride2':self.stride2, 'corr_multiply':self.corr_multiply}

        result = CorrelationFunction.apply(input1, input2, param_dict)

        return result

