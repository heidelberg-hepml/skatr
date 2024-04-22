import torch

class AddSingletonChannel:
    """Add single channel dimension to input. Assumes three spatial dimensions"""
    def forward(self, lightcone, theta):
        lightcone_t = lightcone.unsqueeze(-4)
        return lightcone_t, theta
    
    def reverse(self, lightcone, theta):
        lightcone_t = lightcone.squeeze(-4)
        return lightcone_t, theta

class Center:

    def __init__(self):
        self.lo_lc = -120.
        self.hi_lc = -1.
        self.lo_th = torch.tensor([0.55, 0.20,  100., 38., 4.0, 10.6])
        self.hi_th = torch.tensor([10.0, 0.40, 1500., 42., 5.3, 250.])

    def forward(self, lightcone, theta):
        self.lo_th = self.lo_th.to(theta.device)
        self.hi_th = self.hi_th.to(theta.device)

        lightcone_t = (lightcone - self.lo_lc)/(self.hi_lc - self.lo_lc)
        theta_t     = (theta     - self.lo_th)/(self.hi_th - self.lo_th)
        return lightcone_t, theta_t
    
    def reverse(self, lightcone, theta):
        self.lo_th = self.lo_th.to(theta.device)
        self.hi_th = self.hi_th.to(theta.device)        
        lightcone_t = lightcone*(self.hi_lc - self.lo_lc) + self.lo_lc
        theta_t     =     theta*(self.hi_th - self.lo_th) + self.lo_th
        return lightcone_t, theta_t
