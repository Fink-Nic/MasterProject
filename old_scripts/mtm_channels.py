# type: ignore
from dataclasses import dataclass
import momtrop
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from madnis.nn import Flow

n_samples = 10
batch_size = min(n_samples, 1024)
n_iterations = 1000
log_interval = 10
print_out = n_iterations < log_interval
plt_prog = not print_out
save_fig = False


class MaskedMLP(nn.Module):
    def __init__(
        self,
        input_dims: list[int],
        output_dims: list[int],
        layers: int = 3,
        nodes_per_feature: int = 8,
        activation=nn.LeakyReLU,
    ):
        super().__init__()

        in_degrees = []
        for i, in_dim in enumerate(input_dims):
            in_degrees.extend([i] * in_dim)
        hidden_degrees = torch.repeat_interleave(
            torch.arange(len(input_dims)), nodes_per_feature)
        out_degrees = []
        for i, out_dim in enumerate(output_dims):
            out_degrees.extend([i] * out_dim)
        hidden_layers = layers - 1
        layer_degrees = [
            torch.tensor(in_degrees), *([hidden_degrees]
                                        * hidden_layers), torch.tensor(out_degrees)
        ]

        self.in_slices = [[slice(0)] * layers]
        self.out_slices = [[slice(0)] * layers]
        hidden_dims = [nodes_per_feature] * hidden_layers
        for in_dim, out_dim in zip(input_dims, output_dims):
            self.in_slices.append([
                slice(0, prev_slice_in.stop + deg_in)
                for deg_in, prev_slice_in in zip([in_dim, *hidden_dims], self.in_slices[-1])
            ])
            self.out_slices.append([
                slice(prev_slice_out.stop, prev_slice_out.stop + deg_out)
                for deg_out, prev_slice_out in zip([*hidden_dims, out_dim], self.out_slices[-1])
            ])
        self.in_slices.pop(0)
        self.out_slices.pop(0)

        self.masks = nn.ParameterList()
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for deg_in, deg_out in zip(layer_degrees[:-1], layer_degrees[1:]):
            self.masks.append(
                nn.Parameter(
                    (deg_out[:, None] >= deg_in[None, :]).float(), requires_grad=False)
            )
            self.weights.append(nn.Parameter(
                torch.empty((len(deg_out), len(deg_in)))))
            self.biases.append(nn.Parameter(torch.empty((len(deg_out),))))

        self.activation = activation()
        self.reset_parameters()

    def reset_parameters(self):
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(bias, -bound, bound)
        nn.init.zeros_(self.weights[-1])
        nn.init.zeros_(self.biases[-1])

    def forward(self, x: torch.Tensor):
        for weight, bias, mask in zip(self.weights[:-1], self.biases[:-1], self.masks[:-1]):
            x = self.activation(F.linear(x, mask * weight, bias))
        return F.linear(x, self.masks[-1] * self.weights[-1], self.biases[-1])

    def forward_cached(
        self, x: torch.Tensor, feature: int, cache: list[torch.Tensor] | None = None
    ):
        if cache is None:
            cache = [None] * len(self.weights)
        new_cache = []
        in_slices = self.in_slices[feature]
        out_slices = self.out_slices[feature]
        first = True
        for weight, bias, in_slice, out_slice, x_cached in zip(
            self.weights, self.biases, in_slices, out_slices, cache
        ):
            if first:
                first = False
            else:
                x = self.activation(x)
            if x_cached is not None:
                x = torch.cat((x_cached, x), dim=1)
            new_cache.append(x)
            x = F.linear(x, weight[out_slice, in_slice], bias[out_slice])
        return x, new_cache


Cache = tuple[torch.Tensor, list[torch.Tensor] | None]


class TropicalFlow(nn.Module):
    def __init__(
        self,
        continuous_dim: int,
        discrete_dims: list[int],
        conditional_dim: int,
        continuous_kwargs: dict,
        discrete_kwargs: dict,
    ):
        super().__init__()
        self.flow = Flow(
            dims_in=continuous_dim,
            dims_c=sum(discrete_dims) + conditional_dim,
            **continuous_kwargs
        )
        self.masked_net = MaskedMLP(
            input_dims=[conditional_dim, *discrete_dims[:-1]],
            output_dims=discrete_dims,
            **discrete_kwargs
        )
        self.discrete_dims = discrete_dims
        self.max_dim = max(discrete_dims)
        discrete_indices = []
        one_hot_mask = []
        for i, dim in enumerate(discrete_dims):
            discrete_indices.extend([i] * dim)
            one_hot_mask.extend([True] * dim + [False] * (self.max_dim - dim))
        self.register_buffer('discrete_indices',
                             torch.tensor(discrete_indices))
        self.register_buffer('one_hot_mask', torch.tensor(one_hot_mask))

    def log_prob(
        self,
        indices: torch.Tensor,
        x: torch.Tensor,
        discrete_probs: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:

        x_discrete = F.one_hot(
            indices, self.max_dim
        ).to(x.dtype).flatten(start_dim=1)[:, self.one_hot_mask]
        if condition is None:
            input_disc = x_discrete
        else:
            input_disc = torch.cat((condition, x_discrete), dim=1)
        unnorm_prob_disc = self.masked_net(
            input_disc[:, :-self.discrete_dims[-1]]
        ).exp() * discrete_probs
        prob_norms = torch.zeros_like(indices, dtype=x.dtype).scatter_add_(
            1, self.discrete_indices[None, :].expand(
                x.shape[0], -1), unnorm_prob_disc
        )
        prob_sums = torch.zeros_like(prob_norms).scatter_add_(
            1, self.discrete_indices[None, :].expand(
                x.shape[0], -1), unnorm_prob_disc * x_discrete
        )
        prob_disc = torch.prod(prob_sums / prob_norms, dim=1)
        log_prob_cont = self.flow.log_prob(x, c=input_disc)
        return prob_disc.log() + log_prob_cont

    def init_cache(self, n: int, condition: torch.Tensor | None = None) -> Cache:
        return (torch.zeros((n, 0)) if condition is None else condition, torch.ones((n, )), None)

    def sample_discrete(
        self, dim: int, pred_probs: torch.Tensor, cache: Cache, get_ch_probs
    ) -> tuple[torch.Tensor, torch.Tensor, Cache]:
        x, prob, net_cache = cache
        y, net_cache = self.masked_net.forward_cached(x, dim, net_cache)
        unnorm_probs = y.exp() * pred_probs
        cdf = unnorm_probs.cumsum(dim=1)
        norm = cdf[:, -1]
        cdf = cdf / norm[:, None]
        r = torch.rand((y.shape[0], 1))
        if get_ch_probs and dim == 0:
            samples = torch.tensor([0, 0, 1, 1, 2, 2])
        elif get_ch_probs and dim == 1:
            # In this case we need only return the probabilities
            samples = torch.tensor([1, 2, 0, 2, 0, 1])
            return torch.gather(unnorm_probs, 1, samples[:, None])[:, 0] / norm * prob
        else:
            samples = torch.searchsorted(cdf, r)[:, 0]
        prob = torch.gather(unnorm_probs, 1, samples[:, None])[
            :, 0] / norm * prob
        x_one_hot = F.one_hot(samples, self.discrete_dims[dim]).to(y.dtype)
        return samples, (x_one_hot, prob, net_cache)

    def sample_continuous(self, cache: Cache) -> tuple[torch.Tensor, torch.Tensor]:
        x, prob, net_cache = cache
        condition = torch.cat((net_cache[0], x), dim=1)
        flow_samples, flow_log_prob = self.flow.sample(
            c=condition, return_log_prob=True)
        return flow_samples, prob.log() + flow_log_prob


@dataclass
class SampleBatch:
    x: torch.Tensor
    indices: torch.Tensor
    prob: torch.Tensor
    discrete_probs: torch.Tensor
    func_val: torch.Tensor


class TropicalIntegrator:
    def __init__(self, integrand, lr=3e-4, batch_size=batch_size, continuous_kwargs={}, discrete_kwargs={}):
        self.integrand = integrand
        self.flow = TropicalFlow(
            continuous_dim=integrand.continuous_dim,
            discrete_dims=integrand.discrete_dims,
            conditional_dim=0,
            continuous_kwargs=continuous_kwargs,
            discrete_kwargs=discrete_kwargs,
        )
        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr)
        self.batch_size = batch_size

    def sample(self, n: int, get_ch_probs=False):
        with torch.no_grad():
            if get_ch_probs:
                # We only need to probe each channel once
                n = 6
            discrete_count = len(self.integrand.discrete_dims)
            indices = torch.zeros((n, discrete_count), dtype=torch.int64)
            cache = self.flow.init_cache(n)
            discrete_probs = []
            for i in range(discrete_count):
                pred_probs = self.integrand.predict_discrete_probs(
                    i, indices[:, :i])
                discrete_probs.append(pred_probs)
                if get_ch_probs and i == 1:
                    return self.flow.sample_discrete(i, pred_probs, cache, get_ch_probs)
                indices[:, i], cache = self.flow.sample_discrete(
                    i, pred_probs, cache, get_ch_probs)

            x, log_prob = self.flow.sample_continuous(cache)
            func_val = self.integrand(indices, x)

            return SampleBatch(
                x, indices, log_prob.exp(), torch.cat(discrete_probs, dim=1), func_val
            )

    def optimization_step(self, samples: SampleBatch) -> float:
        self.optimizer.zero_grad()
        log_prob = self.flow.log_prob(
            samples.indices, samples.x, samples.discrete_probs)
        loss = -torch.mean(samples.func_val.abs() / samples.prob * log_prob)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, iterations: int, log_interval=100, plt_prog: bool = False):
        loss_scaling = 5
        legend_loc = 'upper right'
        loss = 0.
        probs = []
        losses = []
        probs.append(self.sample(6, True).numpy())
        for i in range(iterations):
            samples = self.sample(self.batch_size)
            loss += self.optimization_step(samples)
            if (i + 1) % log_interval == 0:
                print(f"Batch {i+1}: loss={loss / log_interval:.6f}")
                prob = self.sample(6, True)
                probs.append(prob.numpy())
                losses.append(loss/log_interval)
                loss = 0.

        # Plotting the probability evolution
        if plt_prog:
            probs = np.array(probs)
            losses = np.array(losses)*loss_scaling
            log_iters = np.arange(start=0,
                                  stop=iterations+1, step=log_interval)
            plt.style.use('ggplot')
            channel_sigs = ['12', '13', '21', '23', '31', '32']
            fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
            fig.suptitle(
                f'Channel Probability Evolution using {n_samples} samples')
            for ch in range(3):
                channel = probs[:, 2*ch] + probs[:, 2*ch+1]
                lbl = f'Ch{ch+1}'
                ax = axs[ch % 2, ch//2]
                ax.plot(log_iters, channel, label=lbl+'x')
                ax.plot(log_iters, probs[:, 2*ch],
                        label=f'Ch{channel_sigs[2*ch]}')
                ax.plot(log_iters, probs[:, 2*ch+1],
                        label=f'Ch{channel_sigs[2*ch+1]}')
                ax.set_title(lbl)
                ax.legend(loc=legend_loc)
                axs[1, 1].plot(log_iters, channel, label=lbl+'x')

            axs[1, 1].plot(log_iters[1:], losses,
                           color='black', label=f'{loss_scaling}*L')
            axs[1, 1].legend(loc=legend_loc)
            axs[1, 1].set_title('Summary')

            plt.ylim([0, loss_scaling/10.])
            if save_fig:
                filename = 'probs_evolution_' + \
                    datetime.now().strftime('%Y_%m_%d-%H_%M_%S')+'.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()

    def integrate(self, n: int) -> tuple[float, float]:
        samples = self.sample(n)
        weights = samples.func_val / samples.prob
        integral = weights.mean().item()
        error = weights.std().item() / math.sqrt(n)
        return integral, error

    def predict_discrete_probs(self, dim: int, indices: torch.Tensor) -> torch.Tensor:
        new_index = torch.arange(self.discrete_dims[dim])
        unnorm_probs = indices.sum(
            dim=1, keepdims=True) + new_index[None, :] + dim + 1

        return_val = unnorm_probs / unnorm_probs.sum(dim=1, keepdims=True)
        return return_val

    def __call__(self, indices: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        prob = self.probs[indices[:, 0], indices[:, 1], indices[:, 2]]
        mu = (indices + 0.5) / (torch.tensor(self.discrete_dims) + 1)
        return_val = (
            prob
            * torch.exp(- (x - mu).square().sum(dim=1) / (2 * self.sigma**2))
            / (2 * math.pi * self.sigma**2)**(3/2)
            / self.integral
        )

        return return_val


class TriangleIntegrand:
    continuous_dim = 7
    discrete_dims = [3, 3]

    def __init__(self):
        edge_1 = momtrop.Edge((0, 1), False, 0.66)
        edge_2 = momtrop.Edge((1, 2), False, 0.77)
        edge_3 = momtrop.Edge((2, 0), False, 0.88)

        assym_graph = momtrop.Graph([edge_1, edge_2, edge_3], [0, 1, 2])
        signature = [[1], [1], [1]]

        self.sampler = momtrop.Sampler(assym_graph, signature)
        self.edge_data = momtrop.EdgeData([0.0, 0.0, 0.0], [momtrop.Vector(
            0.0, 0.0, 0.0), momtrop.Vector(3., 4., 5.), momtrop.Vector(9., 11., 13.)])
        self.settings = momtrop.Settings(False, False)

    def predict_discrete_probs(self, dim: int, indices: torch.Tensor) -> torch.Tensor:
        rust_result = self.sampler.predict_discrete_probs(indices.tolist())

        return torch.tensor(rust_result)

    def get_subgraph_from_edges_removed(self, edges_removed):
        result = []

        for edge in [0, 1, 2]:
            if edge not in edges_removed:
                result.append(edge)

        return result

    def __call__(self, indices: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        rust_result = self.sampler.call(
            indices.tolist(), x.tolist(), self.edge_data, self.settings)

        return torch.tensor(rust_result)


def integrate_flat(integrand, n):
    indices = torch.stack(
        [torch.randint(0, dim, (n, )) for dim in integrand.discrete_dims], dim=1
    )
    prob = 1 / torch.tensor(integrand.discrete_dims).prod()
    x = torch.rand((n, integrand.continuous_dim))
    weights = integrand(indices, x) / prob
    integral = weights.mean().item()
    error = weights.std().item() / math.sqrt(n)
    return integral, error


def main():
    torch.set_default_dtype(torch.float64)
    integrand = TriangleIntegrand()
    integrator = TropicalIntegrator(integrand)
    print("Running_training")
    integrator.train(n_iterations, log_interval, plt_prog)
    print("Training done \n")

    # int_flat, err_flat = integrate_flat(integrand, n_samples)
    # rsd_flat = err_flat / int_flat * math.sqrt(n_samples)
    int_flow, err_flow = integrator.integrate(n_samples)
    rsd_flow = err_flow / int_flow * math.sqrt(n_samples)
    # print(f"Flat integration: {
    #     int_flat:.8f} +- {err_flat:.8f}, RSD = {rsd_flat:.2f}")
    print(f"Trained flow:     {
        int_flow:.8f} +- {err_flow:.8f}, RSD = {rsd_flow:.2f}")


if __name__ == "__main__":
    main()
