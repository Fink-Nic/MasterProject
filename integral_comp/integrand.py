import sys
import os
import random
import logging

import numpy as np
import torch

try:
    from gammaloop.interface.gammaloop_interface import *
except:
    gammaloop_path = os.path.abspath(
        os.path.join(os.getcwd(), os.path.pardir, 'python'))
    sys.path.insert(0, gammaloop_path)
    try:
        from gammaloop.interface.gammaloop_interface import *
        print('\n'.join(["",
                         "Note: gammaLoop could not be loaded from default PYTHONPATH.",
                         f"Path '{gammaloop_path}' was successfully used instead.", ""]))
    except:
        print('\n'.join([
            "ERROR: Could not import Python's gammaloop module.",
            "Add '<GAMMALOOP_INSTALLATION_DIRECTORY>/python' to your PYTHONPATH."]))
from gammaloop.misc.common import GL_CONSOLE_HANDLER

class Integrand:
    def __init__(self, sampling: str, verbosity=logging.CRITICAL, alpha=1.):
        GL_CONSOLE_HANDLER.setLevel(verbosity)

        self.gL_runner = GammaLoop()

        if sampling == 'tropical':
            sampling_str = "{'type':'discrete_graph_sampling','subtype':'tropical'}"
        elif sampling == 'spherical':
            sampling_str = "{'type':'default'}"
        elif sampling == 'multichanneling':
            sampling_str = "{'type':'multi_channeling','alpha':%f}" % alpha

        self.gL_runner.run(CommandList.from_string(
            """
            #You can run commands like if you were running a command card
            import_model scalars-full
            launch ../GL_MASSIVE_VACUUM_MERCEDES
            set sampling %(sampling)s
            set e_cm 1.
            set externals.momenta [[0.,0.,0.,0.],]
            """ % {'sampling': sampling_str}
        ))
        self.gL_runner.sync_worker_with_output()

    def __call__(self, r: torch.Tensor):
        return torch.tensor([abs(self.gL_runner.rust_worker.inspect_integrand(
            "massive_ltd_topology_f", t, [0], False, False, False
        )[1]) for t in r.detach().numpy()])
