import jax
import jax.numpy as jnp

import numpy as np
from jax.experimental.ode import odeint

import matplotlib.pyplot as plt
from functools import partial

from jax.experimental import stax
from jax.experimental import optimizers



def main():

    N = 10
    t = jnp.linspace(0, N, 1000)
    fun = lambda x: jnp.sin(x)

    y = jax.vmap(fun)(t)

    grad_y_analytical = jnp.cos(t)

    grad_y_numerical = jax.vmap(jax.grad(fun))(t)
    
    plt.plot(t, y)
    plt.plot(t, grad_y_analytical)
    plt.plot(t, grad_y_numerical)

    plt.legend(['y=sin(x)', 'analytical dy', 'numerical dy'])
    plt.show()


if __name__ == '__main__':
    main()
