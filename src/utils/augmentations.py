import random
import torch

class RotateAndReflect:
    """Random rotation + reflection, avoiding double counting."""

    def __init__(self, include_identity=False):
        """
        :param include_identity: Whether or not to allow the identity
        """
        
        # construct options
        self.idcs = [(i, j) for i in range(2) for j in range(4)]
        if not include_identity:
            self.idcs.remove((0,0))

    def __call__(self, x):
        """
        :param x: A tensor containing a batch of lightcones.
        """
        # select from options
        ref_idx, rot_idx = random.choice(self.idcs)

        # apply transformations
        x = torch.rot90(x, rot_idx, dims=[2,3])
        if ref_idx:
            x = x.transpose(2, 3)
        
        return x
    
    def enumerate(self, x):
        """
        Returns a list containing all transformations of the input

        :param x: A tensor containing a batch of lightcones.
        """
        
        outs = []

        # select from options
        for ref_idx, rot_idx in self.idcs:

            # apply transformations
            x = torch.rot90(x, rot_idx, dims=[2,3])
            if ref_idx:
                x = x.transpose(2, 3)

            outs.append(x)
        
        return outs
    
class Upsample:
    """TODO: Fill docstring"""

    def __init__(self, factor=5, mode='trilinear'):
        self.upsampler = torch.nn.Upsample(scale_factor=factor, mode=mode)
    
    def __call__(self, x):
        return self.upsampler(x)