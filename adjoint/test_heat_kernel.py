#%load_ext autoreload
#%autoreload 2
import heat_kernel as hk
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


def test_heat_kernel():
    n = 11
    # stencil radius
    r = 4
    a = np.ones(n)
    dt = np.float(0.1)
    x = np.linspace(0, 1, n)
    u = x ** 2
    h = x[1] - x[0]
    d_a = hk.array(a)
    d_u = hk.array(u)
    hk.heat_equation(d_u, d_a, dt, h, update=0.0)
    v = d_u.get()
    assert np.all(np.isclose(2.0 * dt + 0 * v[r:-r], v[r:-r]))

def test_periodic_bc():
    n = 11
    x = np.linspace(0, 2 * np.pi, n)
    u = np.cos(x)
    d_u = hk.array(u)
    hk.periodic_bc(d_u)
    v = d_u.get()
    assert np.all(np.isclose(u,v))
