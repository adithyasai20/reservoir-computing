import torch
import torch.nn as nn

class SurrogareSpikeFunction(torch.autograd.Function):
    """Surrogate gradient function for non-differentiable spikes"""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input>0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        sigmoid_grad = torch.sigmoid(input) * (1 - torch.sigmoid(input))

        return grad_input * sigmoid_grad
    
    

class SNNLayer(nn.Module):
    """Single SNN layer with `output_size` LIF neurons, which takes input from a `input_size` sized layer"""
    def __init__(self, input_size, output_size, threshold=1.0, decay=0.9):
        super(SNNLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.threshold = threshold
        self.decay = decay
        self.spike_fn = SurrogareSpikeFunction.apply
    
    def forward(self, x:torch.tensor):
        mem_potential = torch.zeros(x.size(0), self.fc.out_features, device=x.device)
        spike_train = mem_potential.unsqueeze(0).repeat(x.size(-1), 1, 1)

        for t in range(x.size(-1)):
            
            input_t = x[:, :, t]
            mem_potential = self.decay * mem_potential + self.fc(input_t)
            spikes = self.spike_fn(mem_potential - self.threshold)
            spike_train[t] = spikes
            mem_potential = (1 - spikes) * mem_potential

        return spike_train.permute(1, 2, 0)









