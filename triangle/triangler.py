#!/usr/bin/env python3
# type: ignore
from __future__ import annotations
from typing import Any, Callable, Iterator
from pprint import pprint, pformat
from itertools import repeat  # For the multi
from enum import StrEnum

import momtrop
import argparse
import math
import random
import time
import logging
import multiprocessing
import copy
import numpy
import torch

try:
    import vegas
except:
    pass

try:
    from symbolica import Sample, NumericalIntegrator
except:
    pass

try:
    from triangle import ltd_triangle, prop_factor  # , get_ch_wgt
except:
    pass

from vectors import LorentzVector, Vector
from mtm import TriangleIntegrand, TropicalIntegrator


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
logger = logging.getLogger('Triangler')

TOLERANCE: float = 1e-10

RESCALING: float = 10.


class TrianglerException(Exception):
    pass


def chunks(a_list: list[Any], n: int) -> Iterator[list[Any]]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(a_list), n):
        yield a_list[i:i + n]


def error_fmter(value, error, prec: int | None = None):
    log10v, log10e = math.log10(value), math.log10(error)

    # The default is showing the error to a precision of 2
    if prec is None:
        error_prec = 2
        prec = error_prec + math.floor(log10v) - math.floor(log10e)
    else:
        error_prec = prec - math.floor(log10v) + math.floor(log10e)

    # Case of small error, user should increase precision or use default
    if error_prec <= 0:
        print(
            f'Automatically adjusted precision for the error formatter to {prec - error_prec + 1}.')
        prec = prec - error_prec + 1
        error_prec = 1

    # General string formatter returns non-scientific representation for -4<log10(value)<5
    # 'Hashtag' option for g formatter forces trailing zeros
    if log10v < prec-1 and log10v >= -2:
        value_str = f'{value:#.{prec}g}'
        if log10e >= 0:
            error_str = f'{error:#.{error_prec}g}'
            return f'{value_str}({error_str})'
        else:
            error_str = f'{error:.{error_prec}e}'
            return f'{value_str}({error_str[0]}{error_str[2:error_prec+1]})'

    # In case of scientific representation, reduce prec by one to match actual shown digits with prec
    prec -= 1
    error_prec -= 1

    value_str = f'{value:.{prec}e}'
    error_str = f'{error:.{error_prec}e}'

    return f'{value_str[:prec+2]}({error_str[0]}{error_str[2:error_prec+2]}){value_str[prec+2:]}'


class SymbolicaSample(object):
    def __init__(self, sample: Sample):
        self.c: list[float] = sample.c
        self.d: list[int] = sample.d


class IntegrationResult(object):

    def __init__(self,
                 central_value: float, error: float, n_samples: int = 0, elapsed_time: float = 0.,
                 max_wgt: float | None = None,
                 max_wgt_point: list[float] | None = None):
        self.n_samples = n_samples
        self.central_value = central_value
        self.error = error
        self.max_wgt = max_wgt
        self.max_wgt_point = max_wgt_point
        self.elapsed_time = elapsed_time

    def combine_with(self, other):
        """ Combine self statistics with all those of another IntegrationResult object."""
        self.n_samples += other.n_samples
        self.elapsed_time += other.elapsed_time
        self.central_value += other.central_value
        self.error += other.error
        if other.max_wgt is not None:
            if self.max_wgt is None or abs(self.max_wgt) > abs(other.max_wgt):
                self.max_wgt = other.max_wgt
                self.max_wgt_point = other.max_wgt_point

    def normalize(self):
        """ Normalize the statistics."""
        self.central_value /= self.n_samples
        self.error = math.sqrt(
            abs(self.error / self.n_samples - self.central_value**2)/self.n_samples)

    def str_report(self, target: float | None = None) -> str:

        if self.central_value == 0. or self.n_samples == 0:
            return 'No integration result available yet'

        # First printout sample and timing statitics
        report = [f'Integration result after {Colour.GREEN}{self.n_samples}{
            Colour.END} evaluations in {Colour.GREEN}{self.elapsed_time:.2f} CPU-s{Colour.END}']
        if self.elapsed_time > 0.:
            report[-1] += f' {Colour.BLUE}({1.0e6*self.elapsed_time /
                                            self.n_samples:.1f} µs / eval){Colour.END}'

        # Also indicate max weight encountered if provided
        if self.max_wgt is not None and self.max_wgt_point is not None:
            report.append(f"Max weight encountered = {self.max_wgt:.5e} at xs = [{
                          ' '.join(f'{x:.16e}' for x in self.max_wgt_point)}]")

        # Finally return information about current best estimate of the central value
        report.append(f'{Colour.GREEN}Central value{Colour.END} : {
                      self.central_value:<+25.16e} +/- {self.error:<12.2e}')

        err_perc = self.error/self.central_value*100
        if err_perc < 1.:
            report[-1] += f' ({Colour.GREEN}{err_perc:.3f}%{Colour.END})'
        else:
            report[-1] += f' ({Colour.RED}{err_perc:.3f}%{Colour.END})'

        # Also indicate distance to target if specified
        if target is not None and target != 0.:
            report.append(f'    vs target : {
                          target:<+25.16e} Δ = {self.central_value-target:<+12.2e}')
            diff_perc = (self.central_value-target)/target*100
            if abs(diff_perc) < 1.:
                report[-1] += f' ({Colour.GREEN}{diff_perc:.3f}%{Colour.END}'
            else:
                report[-1] += f' ({Colour.RED}{diff_perc:.3f}%{Colour.END}'
            if abs(diff_perc/err_perc) < 3.:
                report[-1] += f' {Colour.GREEN} = {
                    abs(diff_perc/err_perc):.2f}σ{Colour.END})'
            else:
                report[-1] += f' {Colour.RED} = {abs(diff_perc/err_perc):.2f}σ{
                    Colour.END})'

        # Join all lines and return
        return '\n'.join(f'| > {line}' for line in report)


class Triangle(object):

    def __init__(self, m_psi: float, m_s: float, p: LorentzVector, q: LorentzVector, alpha: float, weight: float):
        self.m_psi = m_psi
        self.m_s = m_s
        self.p = p
        self.q = q
        self.alpha = alpha
        self.weight = weight

        # Only perform sanity checks if in the physical region
        if (self.p+self.q).squared() > 0. or self.p.squared() > 0. or self.q.squared() > 0.:
            if m_s <= 0.:
                raise TrianglerException('m_s must be positive.')
            if abs(p.squared()) / m_s > TOLERANCE:
                raise TrianglerException('p must be on-shell.')
            if abs(q.squared()) / m_s > TOLERANCE:
                raise TrianglerException('q must be on-shell.')
            if abs((p+q).squared()-m_s**2)/m_s**2 > TOLERANCE:
                raise TrianglerException('p+q must be on-shell.')

        # Set up the momtrop sampler
        self.mt_weight = 0.7
        isMassive = m_psi > TOLERANCE
        graph = momtrop.Graph([momtrop.Edge((0, 1), isMassive, self.mt_weight),
                               momtrop.Edge((1, 2), isMassive, self.mt_weight),
                               momtrop.Edge((2, 0), isMassive, self.mt_weight)], [0, 1, 2])
        # This signature should result in edges (k+offset)
        signature = [[1], [1], [1]]
        self.mt_sampler = momtrop.Sampler(graph, signature)
        # EdgeData first takes the masses for each edge in an array, then the loop momentum offsets for the edges
        # These offsets should result in edges (k), (k-q), (k+p)
        self.edge_data = momtrop.EdgeData([m_psi, m_psi, m_psi],
                                          [momtrop.Vector(0.0, 0.0, 0.0),
                                           momtrop.Vector(-q.x, -q.y, -q.z),
                                           momtrop.Vector(p.x, p.y, p.z)])

        self.mt_settings = momtrop.Settings(False, False)
        self.mt_dim = self.mt_sampler.get_dimension()

    def parameterize(self, xs: list[float], parameterisation: str, origin: Vector | None = None) -> tuple[Vector, float]:
        match parameterisation:
            case 'cartesian':
                self.mt_weight = 0.0
                return self.cartesian_parameterize(xs, origin)
            case 'spherical':
                self.mt_weight = 0.0
                return self.spherical_parameterize(xs, origin)
            case 'momtrop': return self.momtrop_parameterize(xs)
            case _: raise TrianglerException(f'Parameterisation {parameterisation} not implemented.')

    def cartesian_parameterize(self, xs: list[float], origin: Vector | None = None) -> tuple[Vector, float]:
        x, y, z = xs
        scale = 10*self.m_s

        # All coordinates use the same conformal mapping log(c/(1-c))
        def conf(c: float):
            return math.log(c/(1-c))

        # Derivate of the conformal mapping
        def d_conf(c: float):
            return 1/(c-c**2)

        cartesian = scale*Vector(conf(x), conf(y), conf(z))
        # Scale factor enters determinant as a third power
        jacobian = scale**3*d_conf(x)*d_conf(y)*d_conf(z)

        if origin is not None:
            return cartesian-origin, jacobian

        return cartesian, jacobian

    def spherical_parameterize(self, xs: list[float], origin: Vector | None = None) -> tuple[Vector, float]:
        x, y, z = xs
        scale = 10*self.m_s

        r = x/(1-x)
        cos_az = 2*y-1
        sin_az = math.sqrt(1 - cos_az**2)
        pol = 2*math.pi*z
        spherical = scale*r*Vector(sin_az*math.cos(pol),
                                   sin_az*math.sin(pol), cos_az)
        # Calculate the jacobian determinant
        jacobian = 4*math.pi*scale**3*x**2/(1-x)**4

        if origin is not None:
            return spherical-origin, jacobian

        return spherical, jacobian

    def momtrop_parameterize(self, xs: list[float]) -> tuple[Vector, float]:
        sample = self.mt_sampler.sample_point(
            xs, self.edge_data, self.mt_settings)
        k = Vector(sample.loop_momenta[0].x(),
                   sample.loop_momenta[0].y(),
                   sample.loop_momenta[0].z())
        # Our LTD expression has a factor of 2 for each propagator
        const_factor = 1./8.

        return k, sample.jacobian/const_factor

    def get_ch_wgt(self, k: Vector, channel: int, mc_exp: float):
        props = [
            k.squared() + self.m_psi**2,
            (k-self.q.spatial()).squared() + self.m_psi**2,
            (k+self.p.spatial()).squared() + self.m_psi**2
        ]

        return props[channel]**-mc_exp/(props[0]**-mc_exp + props[1]**-mc_exp + props[2]**-mc_exp)

    def integrand_xspace(self, xs: list[float], parameterization: str, integrand: str, multi_channeling: bool = False) -> float:
        try:
            if not multi_channeling:
                k, jac = self.parameterize(xs, parameterization)
                wgt = self.integrand(k, integrand)
                final_wgt = wgt * jac
            else:
                logger.info("reeeeeeeeeeeeee")
                final_wgt = 0.
                # exponent is of the propagator terms, not the energies -> factor 2
                mc_exp = self.alpha/2.

                k, jac = self.parameterize(xs, parameterization)
                wgt = self.integrand(k, integrand)
                final_wgt += self.get_ch_wgt(k, 0, mc_exp)*wgt*jac
                k, jac = self.parameterize(
                    xs, parameterization, origin=(-1)*self.q)
                wgt = self.integrand(k, integrand)
                final_wgt += self.get_ch_wgt(k, 1, mc_exp)*wgt*jac
                k, jac = self.parameterize(xs, parameterization, origin=self.p)
                wgt = self.integrand(k, integrand)
                final_wgt += self.get_ch_wgt(k, 2, mc_exp)*wgt*jac

            if math.isnan(final_wgt):
                logger.debug(f"Integrand evaluated to NaN at xs = [{Colour.BLUE}{
                             ', '.join(f'{xi:+.16e}' for xi in xs)}{Colour.END}]. Setting it to zero")
                final_wgt = 0.
        except ZeroDivisionError:
            logger.debug(f"Integrand divided by zero at xs = [{Colour.BLUE}{
                         ', '.join(f'{xi:+.16e}' for xi in xs)}{Colour.END}]. Setting it to zero")
            final_wgt = 0.

        return final_wgt

    def integrand(self, k: Vector, integrand: str) -> float:

        try:
            call_args = [self.m_psi, k.to_list(), self.q.to_list(),
                         self.p.to_list(), self.weight-self.mt_weight]
            match integrand:
                case 'ltd_triangle':
                    return triangle_f(*call_args)
                case 'const_f':
                    return const_f(*call_args)
                case _: raise TrianglerException(f'Integrand type {integrand} not implemented.')
        except ZeroDivisionError:
            logger.debug(f"Integrand divided by zero for k = [{Colour.BLUE}{', '.join(
                f'{ki:+.16e}' for ki in k.to_list())}{Colour.END}]. Setting it to zero")
            return 0.

    def integrate(self, integrator: str, parameterisation: str, integrand: str, target: float | None = None, **opts) -> IntegrationResult:

        match integrator:
            case 'naive': return self.naive_integrator(parameterisation, integrand, target, **opts)
            case 'vegas': return Triangle.vegas_integrator(self, parameterisation, integrand, target, **opts)
            case 'symbolica': return self.symbolica_integrator(parameterisation, integrand, target, **opts)
            case _: raise TrianglerException(f'Integrator {integrator} not implemented.')

    def naive_integrator(self, parameterisation: str, integrand: str, target, **opts) -> IntegrationResult:

        integration_result = IntegrationResult(0., 0.)

        function_call_args = [
            parameterisation, integrand, opts['multi_channeling']]

        def get_rx():
            if parameterisation == 'momtrop':
                return [random.random() for _ in range(self.mt_dim)]
            return [random.random(), random.random(), random.random()]

        for i_iter in range(opts['n_iterations']):
            logger.info(f'Naive integration: starting iteration {Colour.GREEN}{i_iter+1}/{opts["n_iterations"]}{
                        Colour.END} using {Colour.BLUE}{opts["points_per_iteration"]}{Colour.END} points ...')

            res = IntegrationResult(0., 0.)
            res.n_samples = opts['points_per_iteration']
            all_xs = [get_rx() for _ in range(opts['points_per_iteration'])]

            if opts['n_cores'] > 1:
                with multiprocessing.Pool(processes=opts['n_cores']) as pool:
                    all_args = [[xs, *function_call_args]
                                for xs in all_xs]  # I hate this
                    samples = list(pool.starmap(
                        self.integrand_xspace, all_args))
            else:
                samples = list(map(lambda xs: self.integrand_xspace(
                    xs, *function_call_args), all_xs))

            samples_abs = [abs(sample) for sample in samples]

            res.central_value = sum(samples)
            res.max_wgt = max(samples_abs)
            res.max_wgt_point = all_xs[samples_abs.index(res.max_wgt)]
            res.error = sum((sample**2 for sample in samples))
            integration_result.combine_with(res)

            # Normalize a copy for temporary printout
            processed_result = copy.deepcopy(integration_result)
            processed_result.normalize()
            logger.info(f'... result after this iteration:\n{
                        processed_result.str_report(target)}')

        # Normalize results
        integration_result.normalize()

        return integration_result

    @staticmethod
    def vegas_worker(triangle: Triangle, id: int, all_xs: list[list[float]], call_args: list[Any]) -> tuple[int, list[float], IntegrationResult]:
        res = IntegrationResult(0., 0.)
        t_start = time.time()
        all_weights = []
        for xs in all_xs:
            weight = triangle.integrand_xspace(xs, *call_args)
            all_weights.append(weight)
            if res.max_wgt is None or abs(weight) > abs(res.max_wgt):
                res.max_wgt = weight
                res.max_wgt_point = xs
            res.central_value += weight
            res.error += weight**2
            res.n_samples += 1
        res.elapsed_time += time.time() - t_start

        return (id, all_weights, res)

    @staticmethod
    def vegas_functor(triangle: Triangle, res: IntegrationResult, n_cores: int, call_args: list[Any]) -> Callable[[list[list[float]]], list[float]]:

        @vegas.batchintegrand
        def f(all_xs):
            all_weights = []
            if n_cores > 1:
                all_args = [(copy.deepcopy(triangle), i_chunk, all_xs_split, call_args)
                            for i_chunk, all_xs_split in enumerate(chunks(all_xs, len(all_xs)//n_cores+1))]
                with multiprocessing.Pool(processes=n_cores) as pool:
                    all_results = pool.starmap(Triangle.vegas_worker, all_args)
                for _id, wgts, this_result in sorted(all_results, key=lambda x: x[0]):
                    all_weights.extend(wgts)
                    res.combine_with(this_result)
                return all_weights
            else:
                _id, wgts, this_result = Triangle.vegas_worker(
                    triangle, 0, all_xs, call_args)
                all_weights.extend(wgts)
                res.combine_with(this_result)
            return all_weights

        return f

    def vegas_integrator(self, parameterisation: str, integrand: str, _target, **opts) -> IntegrationResult:

        integration_result = IntegrationResult(0., 0.)

        integrator = vegas.Integrator(3 * [[0, 1],])

        local_worker = Triangle.vegas_functor(self, integration_result, opts['n_cores'], [
            parameterisation, integrand, opts['multi_channeling']])
        # Adapt grid
        integrator(local_worker, nitn=opts['n_iterations'],
                   neval=opts['points_per_iteration'], analyzer=vegas.reporter())
        # Final result
        result = integrator(local_worker, nitn=opts['n_iterations'],
                            neval=opts['points_per_iteration'], analyzer=vegas.reporter())

        integration_result.central_value = result.mean
        integration_result.error = result.sdev
        return integration_result

    def symbolica_integrator(self, parameterisation: str, integrand: str, target, **opts) -> IntegrationResult:

        integration_result = IntegrationResult(0., 0.)

        if opts['multi_channeling']:
            integrator = NumericalIntegrator.discrete([
                NumericalIntegrator.continuous(3),
                NumericalIntegrator.continuous(3),
                NumericalIntegrator.continuous(3)
            ])
        else:
            integrator = NumericalIntegrator.continuous(3)

        for i_iter in range(opts['n_iterations']):
            logger.info(f'Symbolica integration: starting iteration {Colour.GREEN}{i_iter+1}/{opts["n_iterations"]}{
                        Colour.END} using {Colour.BLUE}{opts["points_per_iteration"]}{Colour.END} points ...')
            samples = integrator.sample(opts['points_per_iteration'])
            raise NotImplementedError(
                "Implement Symbolica integrator in function 'symbolica_integrator' (Ex. 2.12)")
            integrator.add_training_samples(samples, res)

            # Learning rate is 1.5
            avg, err, _chi_sq = integrator.update(1.5)
            integration_result.central_value = avg
            integration_result.error = err
            logger.info(f'... result after this iteration:\n{
                        integration_result.str_report(target)}')

        return integration_result

    def analytical_result(self) -> complex:
        if self.m_s > 2 * self.m_psi:
            logger.critical(
                'Analytical result not implemented for m_s > 2 * m_psi. Analytical result set to 0.')
            return complex(0., 0.)
        else:
            return math.asin(self.m_s/self.m_psi/2)**2/self.m_s**2/(8*math.pi**2)

    def plot(self, **opts):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcol
        from mpl_toolkits.mplot3d import Axes3D

        fixed_x = None
        for i_x in range(3):
            if i_x not in opts['xs']:
                fixed_x = i_x
                break
        if fixed_x is None:
            raise TrianglerException(
                'At least one x must be fixed (0,1 or 2).')
        n_bins = opts['mesh_size']
        # Create a grid of x and y values within the range [0., 1.]
        # Apply small offset to avoid divisions by zero
        offset = 1e-6
        x = np.linspace(opts['range'][0]+offset,
                        opts['range'][1]-offset, n_bins)
        y = np.linspace(opts['range'][0]+offset,
                        opts['range'][1]-offset, n_bins)
        X, Y = np.meshgrid(x, y)

        # Calculate the values of f(x, y) for each point in the grid
        Z = np.zeros((n_bins, n_bins))
        # Calculate the values of f(x, y) for each point in the grid using nested loops
        xs = [0.,]*3
        xs[fixed_x] = opts['fixed_x']
        for i in range(n_bins):
            for j in range(n_bins):
                xs[opts['xs'][0]] = X[i, j]
                xs[opts['xs'][1]] = Y[i, j]
                if opts['x_space']:
                    Z[i, j] = self.integrand_xspace(
                        xs, opts['parameterisation'], opts['integrand'], opts['multi_channeling'])
                else:
                    Z[i, j] = self.integrand(Vector(
                        xs[0], xs[1], xs[2]), opts['integrand'])

        # Take the logarithm of the function values, handling cases where the value is 0
        with np.errstate(divide='ignore'):
            log_Z = np.log10(np.abs(Z))
            # Replace -inf with 0 for visualization
            log_Z[log_Z == -np.inf] = 0

        if opts['x_space']:
            xs = ['x0', 'x1', 'x2']
        else:
            xs = ['kx', 'ky', 'kz']
        xs[fixed_x] = str(opts['fixed_x'])

        if not opts['3D']:
            # Create the heatmap using matplotlib
            fig = plt.figure(figsize=(8, 6))
            plt.imshow(log_Z, origin='lower', extent=[
                       opts['range'][0], opts['range'][1], opts['range'][0], opts['range'][1]], cmap='viridis')
            plt.colorbar(label=f"log10(I({','.join(xs)}))")
        else:
            # Create a 3D plot (i.e. shamelessly copy one of the documentation examples)
            fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={
                                   'projection': '3d'})
            # Plot the 3D surface
            surf = ax.plot_surface(
                X, Y, log_Z, edgecolor='royalblue', lw=0.8, rstride=8, cstride=8, cmap='viridis')
            ax.set(zlabel=f"log10(I({','.join(xs)}))")
            fig.colorbar(surf, label=f"log10(I({','.join(xs)}))")

        plt.xlabel(f"{xs[opts['xs'][0]]}")
        plt.ylabel(f"{xs[opts['xs'][1]]}")
        plt.title(f"log10(I({','.join(xs)}))")
        plt.show()


if __name__ == '__main__':

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='Triangler')

    # Add options common to all subcommands
    parser.add_argument('--verbosity', '-v', type=str, choices=[
                        'debug', 'info', 'critical'], default='info', help='Set verbosity level')
    parser.add_argument('--parameterisation', '-param', type=str,
                        choices=['cartesian', 'spherical', 'momtrop'],
                        default='spherical',
                        help='Parameterisation to employ.')
    parser.add_argument('--integrand', '-ig', type=str, default='ltd_triangle', choices=[
                        'ltd_triangle', 'const_f'], help='Integrand implementation selected. Default = %(default)s')
    parser.add_argument('--multi_channeling', '-mc', type=bool, default=False,
                        help='Consider a multi-channeled integrand.')
    parser.add_argument('--command', '-c', type=str, choices=[
                        'analytical_result', 'inspect', 'integrate', 'plot', 'compare_sampl'],
                        default='analytical_result', help='Set the command type')
    parser.add_argument('--m_s', type=float,
                        default=0.01,
                        help='Mass of the decaying scalar. Default = %(default)s GeV')
    parser.add_argument('--m_psi', type=float,
                        default=0.02,
                        help='Mass of the spinors. Default = %(default)s GeV')
    parser.add_argument('-p', type=float, nargs=4,
                        default=[0.005, 0.0, 0.0, 0.005],
                        help='Four-momentum of the first photon. Default = %(default)s GeV')
    parser.add_argument('-q', type=float, nargs=4,
                        default=[0.005, 0.0, 0.0, -0.005],
                        help='Four-momentum of the second photon. Default = %(default)s GeV')
    parser.add_argument('--alpha', '-a', type=float, default=1.,
                        help='Exponent alpha in the multichanneling expression')
    parser.add_argument('--weight', '-w', type=float,
                        default=0.5,
                        help='LTD propagator weight. Default = %(default)s, meaning OSE equivalent.')

    # Add subcommands and their options
    subparsers = parser.add_subparsers(
        title="commands", dest="command", help='Various commands available')

    # create the parser for the "inspect" command
    parser_inspect = subparsers.add_parser(
        'inspect', help='Inspect evaluation of a sample point of the integration space.')
    parser_inspect.add_argument(
        '--point', '-p', type=float, nargs=3, help='Sample point to inspect')
    parser_inspect.add_argument('--x_space', action='store_true', default=False,
                                help='Inspect a point given in x-space. Default = %(default)s')
    parser_inspect.add_argument('--full_integrand', action='store_true', default=False,
                                help='Inspect the complete integrand, incl. multi-channeling. Default = %(default)s')

    # create the parser for the "integrate" command
    parser_integrate = subparsers.add_parser(
        'integrate', help='Integrate the loop amplitude.')
    parser_integrate.add_argument('--n_iterations', '-n', type=int, default=10,
                                  help='Number of iterations to perform. Default = %(default)s')
    parser_integrate.add_argument('--points_per_iteration', '-ppi', type=int,
                                  default=100000, help='Number of points per iteration. Default = %(default)s')
    parser_integrate.add_argument('--integrator', '-it', type=str, default='naive', choices=[
                                  'naive', 'symbolica', 'vegas'], help='Integrator selected. Default = %(default)s')
    parser_integrate.add_argument('--n_cores', '-nc', type=int, default=1,
                                  help='Number of cores to run with. Default = %(default)s')
    parser_integrate.add_argument(
        '--seed', '-s', type=int, default=None, help='Specify random seed. Default = %(default)s')

    # Create the parser for the "plot" command
    parser_plot = subparsers.add_parser('plot', help='Plot the integrand.')
    parser_plot.add_argument('--xs', type=int, nargs=2, default=None,
                             help='Chosen 2-dimension projection of the integration space')
    parser_plot.add_argument('--fixed_x', type=float, default=0.75,
                             help='Value of x kept fixed: default = %(default)s')
    parser_plot.add_argument('--range', '-r', type=float, nargs=2,
                             default=[0., 1.], help='range to plot. default = %(default)s')
    parser_plot.add_argument('--x_space', action='store_true', default=False,
                             help='Plot integrand in x-space. Default = %(default)s')
    parser_plot.add_argument('--3D', '-3D', action='store_true', default=False,
                             help='Make a 3D plot. Default = %(default)s')
    parser_plot.add_argument('--mesh_size', '-ms', type=int, default=300,
                             help='Number of bins in meshing: default = %(default)s')

    # create the parser for the "analytical_result" command
    parser_analytical = subparsers.add_parser(
        'analytical_result', help='Compute the analytical result.')

    # create the parser for the "compare_params" command. Yes, some are necessary duplicates
    parser_compare = subparsers.add_parser(
        'compare_sampl', help='Compare all the different triangle samplers.')
    parser_compare.add_argument('--n_iterations', '-n', type=int, default=10,
                                help='Number of iterations to perform. Default = %(default)s')
    parser_compare.add_argument('--points_per_iteration', '-ppi', type=int,
                                default=10000, help='Number of points per iteration. Default = %(default)s')
    parser_compare.add_argument('--integrator', '-it', type=str, default='naive', choices=[
        'naive', 'symbolica', 'vegas'], help='Integrator selected. Default = %(default)s')
    parser_compare.add_argument('--n_cores', '-nc', type=int, default=1,
                                help='Number of cores to run with. Default = %(default)s')
    parser_compare.add_argument(
        '--seed', '-s', type=int, default=None, help='Specify random seed. Default = %(default)s')
    parser_compare.add_argument('--n_train', '-nt', type=int, nargs='*', default=[
                                10, 50], help='Training iterations for a variable number of momtrop + madnis samplers. Default = %(default)s')

    args = parser.parse_args()

    match args.verbosity:
        case 'debug': logger.setLevel(logging.DEBUG)
        case 'info': logger.setLevel(logging.INFO)
        case 'critical': logger.setLevel(logging.CRITICAL)

    q_vec = LorentzVector(args.q[0], args.q[1], args.q[2], args.q[3])
    p_vec = LorentzVector(args.p[0], args.p[1], args.p[2], args.p[3])
    triangle = Triangle(args.m_psi, args.m_s, p_vec,
                        q_vec, args.alpha, args.weight)

    match args.command:

        case 'analytical_result':
            res = triangle.analytical_result()
            logger.info(f'{Colour.GREEN}Analytical result:{Colour.END} {
                        res.real:+.16e} {res.imag:+.16e}j GeV^{{-2}}')

        case 'inspect':
            if args.full_integrand:
                res = triangle.integrand_xspace(
                    args.point, args.parameterisation, args.integrand, args.multi_channeling)
                logger.info(f"Full integrand evaluated at xs = [{Colour.BLUE}{', '.join(
                    f'{xi:+.16e}' for xi in args.point)}{Colour.END}] : {Colour.GREEN}{res:+.16e}{Colour.END}")
            else:
                if args.x_space:
                    k_to_inspect, jacobian = triangle.parameterize(
                        args.point, args.parameterisation)
                else:
                    k_to_inspect, jacobian = Vector(*args.point), 1.
                res = triangle.integrand(
                    k_to_inspect, args.integrand)
                report = f"Integrand evaluated at loop momentum k = [{Colour.BLUE}{', '.join(
                    f'{ki:+.16e}' for ki in k_to_inspect.to_list())}{Colour.END}] : {Colour.GREEN}{res:+.16e}{Colour.END}"
                if args.x_space:
                    report += f' (excl. jacobian = {jacobian:+.16e})'
                logger.info(report)

        case 'integrate':
            if args.seed is not None:
                random.seed(args.seed)
                logger.info(
                    "Note that setting the random seed only ensure reproducible results with the naive integrator and a single core.")

            if args.n_cores > multiprocessing.cpu_count():
                raise TrianglerException(f'Number of cores requested ({
                                         args.n_cores}) is larger than number of available cores ({multiprocessing.cpu_count()})')

            target = triangle.analytical_result()
            t_start = time.time()
            res = triangle.integrate(
                target=target.real, **vars(args)
            )
            integration_time = time.time() - t_start
            tabs = '\t'*5
            new_line = '\n'
            logger.info('-'*80)
            logger.info(f"Integration with settings below completed in {Colour.GREEN}{integration_time:.2f}s{Colour.END}:{new_line}"
                        f"{new_line.join(f'| {Colour.BLUE}{k:<30s}{Colour.END}: {Colour.GREEN}{
                                         pformat(v)}{Colour.END}' for k, v in vars(args).items())}"
                        f"{new_line}| {new_line}{res.str_report(target.real)}")
            logger.info('-'*80)

        case 'plot':
            triangle.plot(**vars(args))

        case 'compare_sampl':
            import numpy as np
            import matplotlib.pyplot as plt
            logger.setLevel(logging.CRITICAL)
            target = triangle.analytical_result()

            p_massive, p_massless = p_vec, LorentzVector(0., 0., 0., 5.)
            q_massive, q_massless = q_vec, LorentzVector(1., 4., 3., 2.)

            # Different sampling implementations
            samplers = ['naive', 'MC', 'vegas',
                        'vegas MC', 'momtrop']
            for nt in args.n_train:
                samplers.append('mtm ' + str(nt))
            n_sampl = len(samplers)
            n_mtm = len(args.n_train)
            n_samples = args.n_iterations*args.points_per_iteration

            # Triangler settings for the non-madnis samplers
            integrators = ['naive', 'naive', 'vegas', 'vegas', 'naive']
            params = ['spherical', 'spherical',
                      'spherical', 'spherical', 'momtrop']
            mc_settings = [False, True, False, True, False]

            # Settings for madnis+momtrop
            mtm_integrands = 2*[triangle_f, const_f]

            # Different integrands
            applications = ['triangle', 'f=1, w=0.7',
                            'triangle, m=0', 'f=1, w=0.7, m=0']
            n_appl = len(applications)
            mtpl = int(n_appl/2)

            # Triangler settings for the integrands
            integrands = 2*['ltd_triangle', 'const_f']
            weights = 2*[0.5, 0.7]
            masses_psi = mtpl*[args.m_psi] + mtpl*[0.]
            masses_s = mtpl*[args.m_s] + mtpl*[1.]
            p_vecs = mtpl*[p_massive] + mtpl*[p_massless]
            q_vecs = mtpl*[q_massive] + mtpl*[q_massless]

            results = [list(range(n_sampl))
                       for _ in range(n_appl)]

            # Loop over all the implementations we want to compare
            for i_sampl in range(n_sampl - n_mtm):
                args.integrator = integrators[i_sampl]
                args.parameterisation = params[i_sampl]
                args.multi_channeling = mc_settings[i_sampl]
                for i_appl in range(n_appl):
                    args.integrand = integrands[i_appl]
                    args.weight = weights[i_appl]
                    args.m_psi = masses_psi[i_appl]
                    args.m_s = masses_s[i_appl]

                    t_start = time.time()
                    triangle = Triangle(args.m_psi, args.m_s, p_vecs[i_appl],
                                        q_vecs[i_appl], args.alpha, args.weight)

                    res = triangle.integrate(
                        target=target.real, **vars(args))
                    res.elapsed_time = time.time() - t_start
                    result = error_fmter(res.central_value, res.error)
                    RSD = res.error/res.central_value * math.sqrt(n_samples)
                    results[i_appl][i_sampl] = f'{result}\n{RSD=:.2f}\nin {res.elapsed_time:.2f}s'
                    del triangle

                    print(
                        f'Finished {samplers[i_sampl]} ||| {applications[i_appl]}')

            # Compute all the integrands using the madnis implementations
            torch.set_default_dtype(torch.float64)
            for i_appl in range(n_appl):
                m_psi = masses_psi[i_appl]
                p_vec = p_vecs[i_appl].to_list()
                q_vec = q_vecs[i_appl].to_list()
                integrand = mtm_integrands[i_appl]
                weight = weights[i_appl]

                t_start = time.time()
                mtm_integrand = TriangleIntegrand(
                    m_psi=m_psi, p=p_vec, q=q_vec, integrand=integrand, weight=weight)
                mtm_integrator = TropicalIntegrator(mtm_integrand)

                nt_curr = 0
                for i_mtm, nt in enumerate(args.n_train):
                    i_sampl = n_sampl - n_mtm + i_mtm  # Current sampler index for results
                    # Required training iterations to match total  training iterations to nt
                    nt_next = nt - nt_curr
                    logging_interval = int(nt_next/5)

                    print(f'Running_training: starting at {nt_curr}/{nt}')
                    mtm_integrator.train(nt_next, logging_interval)
                    print(
                        f'Training done for {samplers[i_sampl]} ||| {applications[i_appl]}')
                    int_flow, err_flow = mtm_integrator.integrate(n_samples)
                    RSD = err_flow / int_flow * math.sqrt(n_samples)
                    result = error_fmter(int_flow, err_flow)
                    results[i_appl][i_sampl] = f'{result}\n{RSD=:.2f}\nin {time.time() - t_start:.2f}s'
                    print(
                        f'Finished {samplers[i_sampl]} ||| {applications[i_appl]}')
                    nt_curr = nt
                    t_start = time.time()

                del mtm_integrator

            # Create the figure. Setting a small pad on tight_layout
            # seems to better regulate white space. Sometimes experimenting
            # with an explicit figsize here can produce better outcome.
            plt.figure(tight_layout={'pad': 1},
                       figsize=(17, 8)
                       )
            # Add a table at the bottom of the axes
            the_table = plt.table(cellText=results,
                                  rowLabels=applications,
                                  rowLoc='right',
                                  colLabels=samplers,
                                  loc='center')
            # Scaling is the only influence we have over top and bottom cell padding.
            # Make the rows taller (i.e., make cell y scale larger).
            the_table.scale(0.8, 3.0)
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(11)
            # Hide axes and axes border
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.box(on=None)
            plt.suptitle(
                f'Comparison of sampler performances for n={args.points_per_iteration} x {args.n_iterations} iterations', size=25)
            # Add footer for the analytical result
            plt.figtext(0.95, 0.05, f'target for massive triangle: {target:5g}',
                        horizontalalignment='right', size=25, weight='light')
            # Force the figure to update, so the title centers on the figure, not the axes
            plt.draw()
            # Create image. plt.savefig ignores figure edge and face colors, so map them.
            fig = plt.gcf()
            plt.savefig('sampler_comparisons.png',
                        # bbox='tight',
                        dpi=150
                        )
            plt.show()

        case _:
            raise TrianglerException(
                f'Command {args.command} not implemented.')
