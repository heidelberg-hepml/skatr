
import torch

class AddSingletonChannel:
    """Add single channel dimension to input. Assumes three spatial dimensions"""
    def forward(self, x):
        return x.unsqueeze(-4)
    
    def reverse(self, x):
        return x.squeeze(-4)

class Center:

    def __init__(self, lo, hi):
        self.lo = torch.tensor(lo)
        self.hi = torch.tensor(hi)

    def forward(self, x):
        self.lo = self.lo.to(x.device)
        self.hi = self.hi.to(x.device)
        return (x - self.lo)/(self.hi - self.lo)
    
    def reverse(self, x):
        self.lo = self.lo.to(x.device)
        self.hi = self.hi.to(x.device)      
        return x*(self.hi - self.lo) + self.lo
    
class Clamp:

    def forward(self, x):
        return x.abs().add(1).log()*x.sign()
    
    def reverse(self, x):
        return x.abs().exp().add(-1)*x.sign()