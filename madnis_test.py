# type: ignore
from madnis.integrator import Integrator
integrator = Integrator(lambda x: (2*x).prod(dim=1), dims=4)
integrator.train(100)
result, error = integrator.integral()
print(f"Integration result: {result:.5f} +- {error:.5f}")
