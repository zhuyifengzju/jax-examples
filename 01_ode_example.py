import jax
import jax.numpy as jnp

import numpy as np
from jax.experimental.ode import odeint
from functools import partial
import matplotlib.pyplot as plt

from jax.experimental import stax
from jax.experimental import optimizers

def first_order_system(state, t=0):
    x = state
    xdot = - x
    return jnp.stack([xdot])

def van_der_pol(state, t=0, mu=1.0):
    """

    Formula:
       \frac{d^{2}x}{dt^{2}} - mu(1 - x^2)\frac{dx}{dt} + x = 0

    """
    
    x, xdot = state
    xddot = mu * (1 - x ** 2) * xdot - x
    return jnp.stack([xdot, xddot])


def create_ode_solver(f, **kwargs):
    @partial(jax.jit, backend="cpu")
    def solve_ode(init_condition, times):
        return odeint(f, init_condition, t=times, rtol=1e-10, atol=1e-10, **kwargs)
    return solve_ode

def main():

    N = 10
    # Canonical system
    xs = None
    solve_ode = create_ode_solver(first_order_system)
    for i in range(-10, 10):
        x0 = np.array([i], dtype=np.float32)
        t = np.linspace(0, N, 1000)
        x = jax.device_get(solve_ode(x0, t))
        if xs is None:
            xs = x
        else:
            xs = np.hstack((xs, x))
    # simple ode
    plt.plot(np.array(xs))
    plt.show()

    # Van der Pol oscillator
    solve_ode = create_ode_solver(van_der_pol)

    for i in [0.5]:
        for j in np.linspace(-2, 2, 5):
            x0 = np.array([i, j], dtype=np.float32)
            t = np.linspace(0, N, N*100)
            x = jax.device_get(solve_ode(x0, t))
            print(x.shape)
            x1, x2 = x[:, 0], x[:, 1]
            plt.plot(x1, x2)
    plt.show()
    


if __name__ == '__main__':
    main()
