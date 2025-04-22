# type: ignore
from madnis.nn import Flow
import math
import os
from dataclasses import dataclass
import momtrop
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats
import math
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from enum import StrEnum
from triangle import ltd_triangle, prop_factor


# Integrand functions
def const_f(m_psi: float, k: list[float], q: list[float], p: list[float], weight: float) -> float:
    return prop_factor(m_psi, k, q, p, weight)


def triangle_f(m_psi: float, k: list[float], q: list[float], p: list[float], weight: float) -> float:
    return prop_factor(m_psi, k, q, p, weight)*ltd_triangle(m_psi, k, q, p)


# Logging related setup
class Colour(StrEnum):
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


logging.basicConfig(
    format=f'{Colour.GREEN}%(levelname)s{Colour.END} {Colour.BLUE}%(funcName)s l.%(lineno)d{
        Colour.END} {Colour.CYAN}t=%(asctime)s.%(msecs)03d{Colour.END} > %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
)
logger = logging.getLogger('mtm')


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
        self.register_buffer("discrete_indices",
                             torch.tensor(discrete_indices))
        self.register_buffer("one_hot_mask", torch.tensor(one_hot_mask))

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
        self, dim: int, pred_probs: torch.Tensor, cache: Cache
    ) -> tuple[torch.Tensor, torch.Tensor, Cache]:
        x, prob, net_cache = cache
        y, net_cache = self.masked_net.forward_cached(x, dim, net_cache)
        unnorm_probs = y.exp() * pred_probs
        cdf = unnorm_probs.cumsum(dim=1)
        norm = cdf[:, -1]
        cdf = cdf / norm[:, None]
        r = torch.rand((y.shape[0], 1))
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
    def __init__(self, integrand, lr=3e-4, batch_size=1024, continuous_kwargs={}, discrete_kwargs={}):
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

    def sample(self, n: int):
        with torch.no_grad():
            discrete_count = len(self.integrand.discrete_dims)
            indices = torch.zeros((n, discrete_count), dtype=torch.int64)
            cache = self.flow.init_cache(n)
            discrete_probs = []
            for i in range(discrete_count):
                pred_probs = self.integrand.predict_discrete_probs(
                    i, indices[:, :i])
                discrete_probs.append(pred_probs)
                indices[:, i], cache = self.flow.sample_discrete(
                    i, pred_probs, cache)

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

    def train(self, iterations: int, n_log=100, **opts):
        loss = 0.
        for i in range(iterations):
            samples = self.sample(self.batch_size)
            loss += self.optimization_step(samples)
            if (i + 1) % n_log == 0:
                print(f"Batch {i+1}: loss={loss / n_log:.6f}")
                loss = 0.

    def integrate(self, n: int) -> tuple[float, float]:
        samples = self.sample(n)
        weights = samples.func_val / samples.prob
        integral = weights.mean().item()
        error = weights.std().item() / math.sqrt(n)
        return integral, error

    def plot(self, **opts):
        pass


class TIChannels(TropicalIntegrator):
    probs = []
    losses = []

    def train(self, iterations: int, n_log=50, **opts):
        loss = 0.
        loss_prog = 0.
        self.probs.append(self.peek_discrete_channels().numpy())
        for i in range(iterations):
            samples = self.sample(self.batch_size)
            it_loss = self.optimization_step(samples)
            loss += it_loss
            loss_prog += it_loss
            if (i + 1) % opts['plotting_interval'] == 0:
                self.probs.append(self.peek_discrete_channels().numpy())
                self.losses.append(loss_prog/opts['plotting_interval'])
                loss_prog = 0.
            if (i + 1) % n_log == 0:
                logger.info(f"Batch {i+1}: loss={loss / n_log:.6f}")
                loss = 0.

    def plot(self, **opts):
        legend_loc = 'upper right'
        probs = np.array(self.probs)
        losses = np.array(self.losses)
        loss_scaling = np.max(probs)/np.max(losses)
        losses *= loss_scaling
        log_iters = np.arange(start=0,
                              stop=opts['n_iterations']+1, step=opts['plotting_interval'])
        plt.style.use('ggplot')
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.suptitle(
            f'Channel Probability Evolution using {opts['batch_size']} samples')
        channel_sigs = ['12', '13', '21', '23', '31', '32']
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
                       color='black', label=f'{loss_scaling:.1f}*L')
        axs[1, 1].legend(loc=legend_loc)
        axs[1, 1].set_title('Summary')

        plt.ylim([0, np.ceil(np.max(probs)*20)/10])
        if not opts['nosave_fig']:
            filename = 'channel_probs_evolution_' + \
                datetime.now().strftime('%Y_%m_%d-%H_%M_%S')+'.png'
            plt.savefig(os.path.join(os.getcwd(), opts['file_path'],
                        filename), dpi=300, bbox_inches='tight')
        plt.show()

    def peek_discrete_channels(self) -> torch.tensor:
        with torch.no_grad():
            discrete_count = len(self.integrand.discrete_dims)
            n = sum(self.integrand.discrete_dims)
            indices = torch.zeros((n, discrete_count), dtype=torch.int64)
            cache = self.flow.init_cache(n)
            discrete_probs = []
            for dim in range(discrete_count):
                pred_probs = self.integrand.predict_discrete_probs(
                    dim, indices[:, :dim])
                discrete_probs.append(pred_probs)
                x, prob, net_cache = cache
                y, net_cache = self.flow.masked_net.forward_cached(
                    x, dim, net_cache)
                unnorm_probs = y.exp() * pred_probs
                cdf = unnorm_probs.cumsum(dim=1)
                norm = cdf[:, -1]
                cdf = cdf / norm[:, None]
                r = torch.rand((y.shape[0], 1))
                if dim == 0:
                    samples = torch.tensor([0, 0, 1, 1, 2, 2])
                elif dim == 1:
                    samples = torch.tensor([1, 2, 0, 2, 0, 1])
                else:
                    logger.critical(
                        "The discrete channel peeker entered an unintended state, dim>2.")

                prob = torch.gather(unnorm_probs, 1, samples[:, None])[
                    :, 0] / norm * prob
                x_one_hot = F.one_hot(
                    samples, self.integrand.discrete_dims[dim]).to(y.dtype)
                indices[:, dim], cache = samples, (x_one_hot, prob, net_cache)

            return prob


class TI1DSlice(TropicalIntegrator):
    def plot(self, **opts):
        force_z = torch.rand(self.integrand.continuous_dim)
        force_z = force_z.repeat((opts['n_samples'], 1))
        if opts['slice_dim'] is None:
            slice_dim = torch.randint(
                size=(1,), high=self.integrand.continuous_dim).item()
        else:
            slice_dim = opts['slice_dim']
        slice_grid = torch.linspace(0, 1, opts['n_samples'])
        force_z[:, slice_dim] = slice_grid
        slice_grid = slice_grid.numpy()

        plt.style.use('ggplot')
        fig, ax = plt.subplots()
        channels = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
        integrals = []
        for channel in channels:
            slice_samples = self.sample_1dslice(
                opts['n_samples'], channel, force_z)
            slice_weights = slice_samples.func_val / slice_samples.prob
            slice_integral = slice_weights.mean().item()
            integrals.append(slice_integral*slice_samples.discrete_probs[0])
            slice_weights.numpy()

            ax.plot(slice_grid, slice_weights,
                    label=f'wch{channel[0]}{channel[1]}')
            # ax.plot(slice_grid, slice_samples.func_val,
            #        label=f'uch{channel[0]}{channel[1]}')

        s = (self.integrand.continuous_dim*'{:.3f} ')[:-1]
        s = s.format(*tuple(force_z[0].tolist()))
        fig.suptitle(
            f"Int: {np.sum(integrals):.4f}, Slice: [{s}]")
        ax.set_xlabel("Sliced dimension z-value")
        ax.set_ylabel("Function value")
        ax.legend()
        if not opts['nosave_fig']:
            filename = 'latent_space_slicing_' + \
                datetime.now().strftime('%Y_%m_%d-%H_%M_%S')+'.png'
            plt.savefig(os.path.join(os.getcwd(), opts['file_path'],
                        filename), dpi=300, bbox_inches='tight')
        plt.show()

    def sample_1dslice(self, n: int, channel, force_z):
        with torch.no_grad():
            discrete_count = len(self.integrand.discrete_dims)
            indices = torch.zeros((n, discrete_count), dtype=torch.int64)
            cache = self.flow.init_cache(n)
            discrete_probs = []
            for i in range(discrete_count):
                pred_probs = self.integrand.predict_discrete_probs(
                    i, indices[:, :i])
                discrete_probs.append(pred_probs)
                indices[:, i], cache = self.sample_discrete_1dslice(
                    i, pred_probs, cache, channel)

            x, log_prob = self.sample_continuous_1dslice(cache, force_z)
            func_val = self.integrand(indices, x)
            return SampleBatch(
                x, indices, log_prob.exp(), torch.cat(discrete_probs, dim=1), func_val
            )

    def sample_discrete_1dslice(
        self, dim: int, pred_probs: torch.Tensor, cache: Cache, channel: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, Cache]:
        x, prob, net_cache = cache
        y, net_cache = self.flow.masked_net.forward_cached(x, dim, net_cache)
        unnorm_probs = y.exp() * pred_probs
        cdf = unnorm_probs.cumsum(dim=1)
        norm = cdf[:, -1]
        cdf = cdf / norm[:, None]
        samples = torch.tensor([channel[dim]]).repeat(y.shape[0])
        prob = torch.gather(unnorm_probs, 1, samples[:, None])[
            :, 0] / norm * prob
        x_one_hot = F.one_hot(
            samples, self.integrand.discrete_dims[dim]).to(y.dtype)
        return samples, (x_one_hot, prob, net_cache)

    def sample_continuous_1dslice(self, cache: Cache, force_z) -> tuple[torch.Tensor, torch.Tensor]:
        x, prob, net_cache = cache
        condition = torch.cat((net_cache[0], x), dim=1)
        flow_samples, jac = self.flow.flow.transform(
            x=force_z, inverse=True, c=condition)
        log_prob_latent = self.flow.flow._latent_log_prob(force_z)
        flow_log_prob = log_prob_latent - jac
        return flow_samples, prob.log() + flow_log_prob


class TriangleIntegrand(object):
    continuous_dim = 7
    discrete_dims = [3, 3]
    mt_weight = 0.7

    def __init__(self, m_psi: float = 0.02, p: list[float] = [0.005, 0., 0., 0.005], q: list[float] = [0.005, 0., 0., -0.005],
                 integrand=triangle_f, weight: float = 0.5):
        isMassive = m_psi > 1e-10
        self.m_psi = m_psi
        self.p = p
        self.q = q
        self.weight = weight
        self.integrand = integrand
        edge_1 = momtrop.Edge((0, 1), isMassive, self.mt_weight)
        edge_2 = momtrop.Edge((1, 2), isMassive, self.mt_weight)
        edge_3 = momtrop.Edge((2, 0), isMassive, self.mt_weight)

        assym_graph = momtrop.Graph([edge_1, edge_2, edge_3], [0, 1, 2])
        signature = [[1], [1], [1]]

        self.sampler = momtrop.Sampler(assym_graph, signature)
        self.edge_data = momtrop.EdgeData(
            [m_psi, m_psi, m_psi],
            [momtrop.Vector(0., 0., 0.),
             momtrop.Vector(-q[1], -q[2], -q[3]),
             momtrop.Vector(p[1], p[2], p[3])
             ])
        self.settings = momtrop.Settings(False, False)
        self.continuous_dim = self.sampler.get_dimension()

    def predict_discrete_probs(self, dim: int, indices: torch.Tensor) -> torch.Tensor:
        rust_result = self.sampler.predict_discrete_probs(indices.tolist())

        return torch.tensor(rust_result)

    def get_subgraph_from_edges_removed(self, edges_removed):
        result = []

        for edge in [0, 1, 2]:
            if edge not in edges_removed:
                result.append(edge)

        return result

    def momtrop_parameterize(self, xs: list[float], force_sector: list[float] | None = None) -> tuple[list[float], float]:
        sample = self.sampler.sample_point(
            xs, self.edge_data, self.settings, force_sector)
        k = [sample.loop_momenta[0].x(),
             sample.loop_momenta[0].y(),
             sample.loop_momenta[0].z()]
        # Our LTD expression has a factor of 2 for each propagator
        const_factor = 1./8.

        return k, sample.jacobian/const_factor

    def integrate_point(self, xs: list[float], indices: list[float] | None = None) -> float:
        force_sector = indices + self.get_subgraph_from_edges_removed(indices)
        k, jac = self.momtrop_parameterize(xs, force_sector)
        jac *= self.sampler.get_sector_prob(force_sector)

        return self.integrand(self.m_psi, k, self.q, self.p, self.weight-self.mt_weight)*jac

    def __call__(self, indices: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        result = list(map(self.integrate_point, x.tolist(), indices.tolist()))

        return torch.tensor(result)


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
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='mtm')

    parser.add_argument('--command', '-c', type=str, choices=[
                        'channel_prog', '1dslice', 'integrate'], default='integrate', help='Perform an implemented command.')
    parser.add_argument('--n_samples', '-ns', type=int, default=10000,
                        help='Set number of samples taken from trained model')
    parser.add_argument('--n_iterations', '-ni', type=int, default=1000,
                        help='Set number of training iterations')
    parser.add_argument('--batch_size', '-bs', type=int, default=1024,
                        help='Set batch size per training iteration')
    parser.add_argument('--log_interval', '-li', type=int, default=100,
                        help='Set number of batches between logging')
    parser.add_argument('--verbosity', '-v', type=str, choices=[
                        'debug', 'info', 'critical'], default='info', help='Set verbosity level')
    parser.add_argument('--nosave_fig', action='store_true', default=False,
                        help='Enable to only show plotted figure')
    parser.add_argument('--file_path', '-fp', type=str, default='',
                        help='Specify a relative path to a folder to save the plot at.')

    subparser = parser.add_subparsers(
        title="commands", dest="command", help='Various commands available')

    # integrate subparser
    parser_integrate = subparser.add_parser(
        'integrate', help='Integrate the integrand.'
    )

    # channel_prog subparser
    parser_channel_prog = subparser.add_parser(
        'channel_prog', help='Inspect the evolution of the discrete probabilities for each channel.'
    )
    parser_channel_prog.add_argument('--plotting_interval', '-pi', type=int,
                                     default=100, help='Over-ride the log_interval for the channel_prog plot.')

    # 1dslice subparser
    parser_1dslice = subparser.add_parser(
        '1dslice', help='Inspect the latent space along a one-dimensional slice.')
    parser_1dslice.add_argument('--slice_dim', '-sd', type=int, default=2,
                                help='Optional: specify the dimension along which to slice for the 1dslice analysis.')

    args = parser.parse_args()

    match args.verbosity:
        case 'debug': logger.setLevel(logging.DEBUG)
        case 'info': logger.setLevel(logging.INFO)
        case 'critical': logger.setLevel(logging.CRITICAL)

    subfolder_path = os.path.join(os.getcwd(), args.file_path)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
        logger.info(
            f"Created target folder {subfolder_path} for plotting results, please make sure this is the intended location.")

    torch.set_default_dtype(torch.float64)
    integrand = TriangleIntegrand()
    match args.command:
        case 'integrate': integrator = TropicalIntegrator(integrand, batch_size=args.batch_size)
        case 'channel_prog': integrator = TIChannels(integrand, batch_size=args.batch_size)
        case '1dslice': integrator = TI1DSlice(integrand, batch_size=args.batch_size)

    logger.info("Running_training")
    integrator.train(args.n_iterations, args.log_interval, **vars(args))
    logger.info("Training done \n")

    int_flow, err_flow = integrator.integrate(args.n_samples)
    rsd_flow = err_flow / int_flow * math.sqrt(args.n_samples)
    logger.info(f"Trained flow:     {
        int_flow:.8f} +- {err_flow:.8f}, RSD = {rsd_flow:.2f}")
    integrator.plot(**vars(args))


if __name__ == "__main__":
    main()
