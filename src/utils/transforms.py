
import torch

class AddSingletonChannel:
    """Add single channel dimension to input. Assumes three spatial dimensions"""
    def forward(self, x):
        return x.unsqueeze(-4)
    
    def reverse(self, x):
        return x.squeeze(-4)

class Center:
    """Shift and scale a tensor into the range [0,1] given min value `lo` and max value `hi`"""

    def __init__(self, lo, hi, indices=None):
        self.lo = torch.tensor(lo)
        self.hi = torch.tensor(hi)
        self.indices = indices
        if indices is not None:
            self.lo = self.lo[indices]
            self.hi = self.hi[indices]

    def forward(self, x):
        self.lo = self.lo.to(x.device)
        self.hi = self.hi.to(x.device)
        if self.indices is not None:
            x = x[..., sorted(self.indices)]
        return (x - self.lo)/(self.hi - self.lo)
    
    def reverse(self, x):
        self.lo = self.lo.to(x.device)
        self.hi = self.hi.to(x.device)      
        return x*(self.hi - self.lo) + self.lo
    
class Clamp:
    """Apply a symmetric log scaling to the input."""
    def forward(self, x):
        return x.abs().add(1).log()*x.sign()
    
    def reverse(self, x):
        return x.abs().exp().add(-1)*x.sign()