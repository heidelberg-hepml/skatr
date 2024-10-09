import torch
import torch.nn.functional as F
import torchsort

from omegaconf import DictConfig

from src import networks
from src.models.base_model import Model


class NPE(Model):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg = cfg
        sum_net_cls = getattr(networks, cfg.summary_net.arch)
        self.summary_net = sum_net_cls(cfg.summary_net)

        if cfg.use_extra_summary_mlp:
            self.extra_mlp = networks.MLP(cfg.extra_mlp)
        if cfg.use_attn_pool:
            self.attn_pool = networks.AttentiveHead(cfg.attn_pool)

        self.pool_summary = not (
            hasattr(self.summary_net, "head")
            or hasattr(self, "attn_pool")
            or cfg.summary_net.arch == "CNN"
        )

        self.inn = self.net

    def summarize(self, c: torch.Tensor) -> torch.Tensor:

        if not self.cfg.data.summarize:
            c = self.summary_net(c)
            if self.pool_summary:
                c = c.mean(1)  # (B, T, D) -> (B, D)

        if self.cfg.use_extra_summary_mlp:
            c = self.extra_mlp(c)

        if self.cfg.use_attn_pool:
            c = self.attn_pool(c)

        return c

    def log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c = self.summarize(c)
        return self.inn.log_prob(x, c)

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
        c = self.summarize(c)
        samples, logprobs = self.inn.sample_batch(c)
        return samples, logprobs

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

        return -self.log_prob(x, c).mean() / self.cfg.net.dim


class CalibratedNPE(NPE):

    def __init__(self, cfg: DictConfig):

        self.calibration_num_samples = cfg.calibration_num_samples
        self.calibration_weight = cfg.calibration_weight
        self.conservative = cfg.conservative

        super().__init__(cfg)

    def batch_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log probability

        Args:
            x: input tensor, shape (n_events, dims_in)
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            loss: batch loss
        """
        ranks = self.get_ranks(batch)
        coverage, expected = get_coverage(ranks, device=batch[0].device)
        if self.conservative:
            regularizer = F.relu(expected - coverage).pow(2).mean()
        else:
            regularizer = (expected - coverage).pow(2).mean()

        return super().batch_loss(batch) + self.calibration_weight * regularizer

    def get_ranks(self, batch, logits=False):

        c, x = batch

        c = self.summarize(c)
        # evaluate true param likelihoods
        param_logprobs = self.inn.log_prob(x, c)

        # sample the posterior for each test point in parallel
        c = c.repeat_interleave(self.calibration_num_samples, 0)
        _, posterior_logprobs = self.inn.sample_batch(c)
        posterior_logprobs = posterior_logprobs.reshape(
            len(x), self.calibration_num_samples
        )

        return STEFunctionRanksq.apply(
            param_logprobs.unsqueeze(1) - posterior_logprobs
        ).mean(1)


# from https://github.com/DMML-Geneva/calibrated-posterior
class STEFunctionRanksq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


def get_coverage(ranks, device):
    # Source: https://github.com/montefiore-ai/balanced-nre/blob/main/demo.ipynb
    # As a sample at a given rank belongs to the credible regions at levels 1-rank and below,
    # the coverage at level 1-alpha is the proportion of samples with ranks alpha and above.
    ranks = ranks[~ranks.isnan()]
    alpha = torchsort.soft_sort(ranks.unsqueeze(0)).squeeze()
    return (
        torch.linspace(0.0, 1.0, len(alpha) + 2, device=device)[1:-1],
        1 - torch.flip(alpha, dims=(0,)),
    )
