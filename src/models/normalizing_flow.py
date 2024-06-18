import FrEIA.framework as ff
import FrEIA.modules as fm
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from scipy.stats import special_ortho_group
from typing import Callable, Iterable, Type, Union

from src.models.base_model import Model
from src.networks import ViT

class INN(Model):
    """
    Class implementing a standard conditional INN
    """
    def __init__(self, cfg:DictConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.sum_net = self.bb if cfg.backbone else ViT(cfg.summary_net)
        self.build_inn()

    def construct_subnet(self, x_in: int, x_out: int) -> nn.Module:
        subnet = Subnet(
            self.cfg.inn.layers_per_block, x_in, x_out,
            internal_size=self.cfg.inn.internal_size, dropout=self.cfg.inn.dropout
        )
        return subnet

    def build_inn(self):
        """
        Construct the INN
        """
        permute_soft = self.cfg.inn.permute_soft
        if self.cfg.inn.latent_space == "gaussian":
            upper_bound = self.cfg.inn.spline_bound
            lower_bound = -upper_bound
        elif self.cfg.inn.latent_space == "uniform":
            lower_bound = 0
            upper_bound = 1
            if permute_soft:
                raise ValueError(
                    "Soft permutations not supported for uniform latent space"
                )
        block_kwargs = {
            "num_bins": self.cfg.inn.num_bins,
            # "subnet_constructor": constructor_fct,
            "subnet_constructor": self.construct_subnet,
            "left": lower_bound,
            "right": upper_bound,
            "bottom": lower_bound,
            "top": upper_bound,
            "permute_soft": permute_soft
        }

        self.inn = ff.SequenceINN(self.cfg.dim)
        for _ in range(self.cfg.inn.num_blocks):
            self.inn.append(
                RationalQuadraticSplineBlock, cond=0, cond_shape=(self.cfg.summary_dim,),
                # **self.get_coupling_block_kwargs()
                **block_kwargs
            )

    def latent_log_prob(self, z: torch.Tensor) -> Union[torch.Tensor, float]:
        """
        Returns the log probability for a tensor in latent space

        Args:
            z: latent space tensor, shape (n_events, dims_in)
        Returns:
            log probabilities, shape (n_events, )
        """
        if self.cfg.inn.latent_space == "gaussian":
            return -(z**2 / 2 + 0.5 * math.log(2 * math.pi)).sum(dim=1)
        elif self.cfg.inn.latent_space == "uniform":
            return 0.0

    def log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log probability

        Args:
            x: input tensor, shape (n_events, dims_in)
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            log probabilities, shape (n_events, ) if not bayesian
                               shape (1+self.bayesian_samples, n_events) if bayesian
        """
        z, jac = self.inn(x, (c,)) # TODO: summarize c first
        return self.latent_log_prob(z) + jac

    @torch.inference_mode()
    def sample_batch(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples and log probabilities for the given condition

        Args:
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            x: generated samples, shape (n_events, dims_in)
            log_prob: log probabilites, shape (n_events, )
        """
        latent_sampler = (
            torch.randn if self.cfg.inn.latent_space == "gaussian" else 
            torch.rand if self.cfg.inn.latent_space == "uniform" else None
        )
        z = latent_sampler((c.shape[0], self.cfg.dim), dtype=c.dtype, device=c.device)    
        
        c = self.sum_net(c)
        x, jac = self.inn(z, (c,), rev=True)
        log_prob = self.latent_log_prob(z) - jac
        return x.detach().cpu(), log_prob.detach().cpu()

    def transform_hypercube(
        self, r: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes values and jacobians for the given condition and numbers on the unit
        hypercube

        Args:
            r: points on the the unit hypercube, shape (n_events, dims_in)
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            x: generated samples, shape (n_events, dims_in)
            jac: jacobians, shape (n_events, )
        """
        if self.cfg.inn.latent_space == "gaussian":
            z = torch.erfinv(2 * r - 1) * math.sqrt(2)
        elif self.cfg.inn.latent_space == "uniform":
            z = r
        x, jac = self.inn(z, (c,), rev=True)
        return x, -self.latent_log_prob(z) + jac

    def batch_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log probability

        Args:
            x: input tensor, shape (n_events, dims_in)
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            loss: batch loss
        """
        c, x = batch
        c = self.sum_net(c)
        return -self.log_prob(x, c).mean() / self.cfg.dim


class Subnet(nn.Module):
    """
    Standard MLP or bayesian network to be used as a trainable subnet in INNs
    """

    def __init__(
        self,
        num_layers: int,
        size_in: int,
        size_out: int,
        internal_size: int,
        dropout: float = 0.0,
        layer_class: Type = nn.Linear,
        layer_args: dict = {},
    ):
        """
        Constructs the subnet.

        Args:
            num_layers: number of layers
            size_in: input size of the subnet
            size: output size of the subnet
            internal_size: hidden size of the subnet
            dropout: dropout chance of the subnet
            layer_class: class to construct the linear layers
            layer_args: keyword arguments to pass to the linear layer
        """
        super().__init__()
        if num_layers < 1:
            raise (ValueError("Subnet size has to be 1 or greater"))
        self.layer_list = []
        for n in range(num_layers):
            input_dim, output_dim = internal_size, internal_size
            if n == 0:
                input_dim = size_in
            if n == num_layers - 1:
                output_dim = size_out

            self.layer_list.append(layer_class(input_dim, output_dim, **layer_args))

            if n < num_layers - 1:
                if dropout > 0:
                    self.layer_list.append(nn.Dropout(p=dropout))
                self.layer_list.append(nn.ReLU())

        self.layers = nn.Sequential(*self.layer_list)

        for name, param in self.layer_list[-1].named_parameters():
            if "logsig2_w" not in name:
                param.data *= 0.02

    def forward(self, x):
        return self.layers(x)

class RationalQuadraticSplineBlock(fm.InvertibleModule):
    """
    Implementation of rational quadratic spline coupling blocks
    (https://arxiv.org/pdf/1906.04032.pdf) as a FrEIA invertible module,
    based on the implementation in https://github.com/bayesiains/nflows
    """

    DEFAULT_MIN_BIN_WIDTH = 1e-3
    DEFAULT_MIN_BIN_HEIGHT = 1e-3
    DEFAULT_MIN_DERIVATIVE = 1e-3

    def __init__(
        self,
        dims_in: Iterable[tuple[int]],
        dims_c=Iterable[tuple[int]],
        subnet_constructor: Callable = None,
        num_bins: int = 10,
        left: float = 0.0,
        right: float = 1.0,
        bottom: float = 0.0,
        top: float = 1.0,
        permute_soft: bool = False,
        min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
        min_derivative: float = DEFAULT_MIN_DERIVATIVE,
    ):
        """
        Initializes the RQS coupling block

        Args:
            dims_in: shapes of the inputs
            dims_c: shapes of the conditions
            subnet_constructor: function that constructs the coupling block subnet
            num_bins: number of spline bins
            left: lower input bound (forward)
            right: upper input bound (forward)
            bottom: lower input bound (inverse)
            top: upper input bound (inverse)
            permute_soft: if True, insert rotations matrix instead of permutation after
                          the coupling block
            min_bin_width: minimal spline bin width
            min_bin_height: minimal spline bin height
            min_derivative: minimal derivative at bin boundary
        """
        super().__init__(dims_in, dims_c)
        channels = dims_in[0][0]
        # rank of the tensors means 1d, 2d, 3d tensor etc.
        self.input_rank = len(dims_in[0]) - 1
        # tuple containing all dims except for batch-dim (used at various points)
        self.sum_dims = tuple(range(1, 2 + self.input_rank))
        if len(dims_c) == 0:
            self.conditional = False
            self.condition_channels = 0
        else:
            assert tuple(dims_c[0][1:]) == tuple(
                dims_in[0][1:]
            ), f"Dimensions of input and condition don't agree: {dims_c} vs {dims_in}."
            self.conditional = True
            self.condition_channels = sum(dc[0] for dc in dims_c)

        split_len1 = channels - channels // 2
        split_len2 = channels // 2
        self.splits = [split_len1, split_len2]
        self.num_bins = num_bins
        if self.DEFAULT_MIN_BIN_WIDTH * self.num_bins > 1.0:
            raise ValueError("Minimal bin width too large for the number of bins")
        if self.DEFAULT_MIN_BIN_HEIGHT * self.num_bins > 1.0:
            raise ValueError("Minimal bin height too large for the number of bins")
        self.in_channels = channels
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            w = np.zeros((channels, channels))
            for i, j in enumerate(np.random.permutation(channels)):
                w[i, j] = 1.0

        self.w_perm = nn.Parameter(
            torch.FloatTensor(w).view(channels, channels, *([1] * self.input_rank)),
            requires_grad=False,
        )
        self.w_perm_inv = nn.Parameter(
            torch.FloatTensor(w.T).view(channels, channels, *([1] * self.input_rank)),
            requires_grad=False,
        )

        if subnet_constructor is None:
            raise ValueError(
                "Please supply a callable subnet_constructor"
                "function or object (see docstring)"
            )
        self.subnet = subnet_constructor(
            self.splits[0] + self.condition_channels,
            (3 * self.num_bins + 1) * self.splits[1],
        )

    def forward(
        self,
        x: Iterable[torch.Tensor],
        c: Iterable[torch.Tensor] = [],
        rev: bool = False,
        jac: bool = True,
    ) -> tuple[tuple[torch.Tensor], torch.Tensor]:
        """
        Computes the coupling transformation

        Args:
            x: Input tensors
            c: Condition tensors
            rev: If True, compute inverse transformation
            jac: Not used, Jacobian is always computed

        Returns:
            Output tensors and log jacobian determinants
        """

        (x,) = x

        if rev:
            x = F.linear(x, self.w_perm_inv)

        x1, x2 = torch.split(x, self.splits, dim=1)

        if self.conditional:
            x1c = torch.cat([x1, *c], dim=1)
        else:
            x1c = x1

        theta = self.subnet(x1c).reshape(
            x1c.shape[0], self.splits[1], 3 * self.num_bins + 1
        )
        x2, log_jac_det = unconstrained_rational_quadratic_spline(
            x2,
            theta,
            rev=rev,
            num_bins=self.num_bins,
            left=self.left,
            right=self.right,
            top=self.top,
            bottom=self.bottom,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
        x_out = torch.cat((x1, x2), dim=1)

        if not rev:
            x_out = F.linear(x_out, self.w_perm)

        return (x_out,), log_jac_det

    def output_dims(self, input_dims: list[tuple[int]]) -> list[tuple[int]]:
        """
        Defines the output shapes of the coupling block

        Args:
            input_dims: Shapes of the inputs

        Returns:
            Shape of the outputs
        """
        return input_dims


def unconstrained_rational_quadratic_spline(
    inputs: torch.Tensor,
    theta: torch.Tensor,
    rev: bool,
    num_bins: int,
    left: float,
    right: float,
    bottom: float,
    top: float,
    min_bin_width: float,
    min_bin_height: float,
    min_derivative: float,
    periodic: bool = False,
    sum_jacobian: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transform inputs using RQ splines defined by theta.

    Args:
        inputs: Input tensor
        theta: tensor with bin widths, heights and derivatives
        rev: If True, compute inverse transformation

    Returns:
        Transformed tensor and log of jacobian determinant
    """
    if not rev:
        inside_interval_mask = torch.all((inputs >= left) & (inputs <= right), dim=-1)
    else:
        inside_interval_mask = torch.all((inputs >= bottom) & (inputs <= top), dim=-1)
    outside_interval_mask = ~inside_interval_mask
    masked_outputs = torch.zeros_like(inputs)
    masked_logabsdet = torch.zeros(
        inputs.shape[:(1 if sum_jacobian else 2)], dtype=inputs.dtype, device=inputs.device
    )
    masked_outputs[outside_interval_mask] = inputs[outside_interval_mask]
    masked_logabsdet[outside_interval_mask] = 0

    inputs = inputs[inside_interval_mask]
    theta = theta[inside_interval_mask, :]

    unnormalized_widths = theta[..., :num_bins]
    unnormalized_heights = theta[..., num_bins : num_bins * 2]
    unnormalized_derivatives = theta[..., num_bins * 2 :]

    # unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    # constant = np.log(np.exp(1 - min_derivative) - 1)
    # unnormalized_derivatives[..., 0] = constant
    # unnormalized_derivatives[..., -1] = constant

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = (min_derivative + F.softplus(unnormalized_derivatives)) / (
        min_derivative + math.log(2)
    )
    if periodic:
        derivatives[...,-1] = derivatives[...,0]
        periodic_shift = (right - left) / 2 * torch.tanh(unnormalized_derivatives[...,-1])

        if not rev:
            inputs = torch.remainder(inputs + periodic_shift - left, right - left) + left
            infi = inputs[(inputs < left) | (inputs > right) | ~torch.isfinite(inputs)]
            if len(infi) > 0:
                print(infi)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if rev:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if rev:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (torch.isnan(discriminant) | (discriminant >= 0)).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths
        if periodic:
            outputs = torch.remainder(outputs - periodic_shift - left, right - left) + left

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = -torch.log(derivative_numerator) + 2 * torch.log(denominator)

    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    if sum_jacobian:
        logabsdet = torch.sum(logabsdet, dim=1)

    masked_outputs[inside_interval_mask] = outputs
    masked_logabsdet[inside_interval_mask] = logabsdet

    return masked_outputs, masked_logabsdet

def searchsorted(
    bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1    