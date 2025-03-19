from typing import Callable
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    if not inverse:
        inside_interval_mask = torch.all(
            (inputs >= left) & (inputs <= right), dim=-1)
    else:
        inside_interval_mask = torch.all(
            (inputs >= bottom) & (inputs <= top), dim=-1)

    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0

    (
        outputs[inside_interval_mask],
        logabsdet[inside_interval_mask],
    ) = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    assert (
        inputs.shape[0] == 0 or
        (left <= torch.min(inputs) and torch.max(inputs) <= right)
    ), f"Inputs < 0 or > 1. Min: {torch.min(inputs)}, Max: {torch.max(inputs)}"

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = ((min_derivative + F.softplus(unnormalized_derivatives))
                   / (min_derivative + math.log(2)))

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[...,
                                             1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

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
        logabsdet = torch.log(derivative_numerator) - \
            2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives *
            theta_one_minus_theta
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
        logabsdet = torch.log(derivative_numerator) - \
            2 * torch.log(denominator)

        return outputs, logabsdet


class MLP(nn.Module):
    def __init__(
        self,
        features_in: int,
        features_out: int,
        layers: int,
        units: int,
        activation=nn.ReLU,
        layer_constructor=nn.Linear,
    ):
        super().__init__()
        input_dim = features_in
        layer_list = []
        for i in range(layers - 1):
            layer_list.append(layer_constructor(input_dim, units))
            layer_list.append(activation())
            input_dim = units
        layer_list.append(layer_constructor(input_dim, features_out))
        nn.init.zeros_(layer_list[-1].weight)
        nn.init.zeros_(layer_list[-1].bias)
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class ConditionalFlow(nn.Module):
    def __init__(
        self,
        dims_in: int,
        dims_c: int,
        layers: int = 3,
        units: int = 32,
        bins: int = 10,
        activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.dims_in = dims_in

        def subnet_constructor(features_in, features_out): return MLP(
            features_in, features_out, layers, units, activation
        )

        n_perms = int(np.ceil(np.log2(dims_in)))
        blocks = int(2 * n_perms)
        self.masks = torch.tensor([
            [int(i) for i in np.binary_repr(i, n_perms)] for i in range(dims_in)
        ]).flip(dims=(1,)).bool().t().repeat_interleave(2, dim=0)
        self.masks[1::2, :] ^= True

        self.subnets = nn.ModuleList()
        for mask in self.masks:
            dims_cond = torch.count_nonzero(mask)
            self.subnets.append(subnet_constructor(
                dims_cond + dims_c,
                (dims_in - dims_cond) * (3 * bins + 1)
            ))

    def transform(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        inverse: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.clone()
        jac = 0.
        if inverse:
            blocks = zip(reversed(self.masks), reversed(self.subnets))
        else:
            blocks = zip(self.masks, self.subnets)
        for mask, subnet in blocks:
            inv_mask = ~mask
            x_trafo = x[:, inv_mask]
            x_cond = torch.cat((x[:, mask], c), dim=1)
            subnet_out = subnet(x_cond).reshape(
                (x.shape[0], x_trafo.shape[1], -1))
            bins = subnet_out.shape[-1] // 3
            x_out, block_jac = unconstrained_rational_quadratic_spline(
                x_trafo,
                subnet_out[:, :, :bins],
                subnet_out[:, :, bins:2*bins],
                subnet_out[:, :, 2*bins:],
                inverse,
            )
            x[:, inv_mask] = x_out
            jac += block_jac.sum(dim=1)
        return x, jac

    def log_prob(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        if c is None:
            c = x[:, :0]
        z, jac = self.transform(x, c, False)
        return jac

    def sample(
        self, c: torch.Tensor | None = None, n: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if c is not None:
            n = len(c)
        else:
            c = torch.zeros((n, 0))
        z = torch.rand((n, self.dims_in))
        x, jac = self.transform(z, c, True)
        return x, -jac


class Integrator:
    def __init__(
        self,
        integrand: Callable[[torch.Tensor], torch.Tensor],
        dimensions: int,
        lr: float = 3e-4,
        loss: str = "variance",
    ):
        self.integrand = integrand
        self.flow = ConditionalFlow(dimensions, 0)
        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=lr)
        if loss == "variance":
            self.loss = lambda f, g, log_g, q: torch.mean(
                g / q * (f / g - 1) ** 2)
        elif loss == "kl":
            self.loss = lambda f, g, log_g, q: - torch.mean(f / q * log_g)
        else:
            raise ValueError(f"unknown loss {loss}")

    def train(self, batches: int, batch_size: int = 1024, log_interval: int = 20):
        losses = []
        for batch in range(batches):
            with torch.no_grad():
                r, log_prob = self.flow.sample(n=batch_size)
            q = log_prob.exp()
            f_unnorm = self.integrand(r)
            f = f_unnorm / f_unnorm.mean()
            self.optimizer.zero_grad()
            log_g = self.flow.log_prob(r)
            g = log_g.exp()
            loss = self.loss(f, g, log_g, q)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            if (batch + 1) % log_interval == 0:
                print(f"Batch {batch + 1}, loss {np.mean(losses)}")

    def sample(self, n: int):
        with torch.no_grad():
            r, log_prob = self.flow.sample(n=n)
            return self.integrand(r) / log_prob.exp()
