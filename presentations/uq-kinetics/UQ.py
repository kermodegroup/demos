# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "arviz",
#     "diffrax",
#     "equinox",
#     "jax",
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "numpyro",
#     "optax",
#     "pandas",
#     "qrcode",
#     "scipy",
#     "seaborn",
#     "tinygp",
#     "tqdm",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium", layout_file="layouts/UQ.slides.json")


@app.cell(hide_code=True)
def _():
    import os
    import sys
    import numpy as np
    import scipy.stats as st
    import matplotlib.pyplot as plt
    import pandas as pd

    # Customize default plotting style
    import seaborn as sns
    sns.set_context('talk')

    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import NUTS, MCMC
    numpyro.set_host_device_count(min(2, os.cpu_count() or 1))

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from jax import config
    from jax.flatten_util import ravel_pytree
    from typing import Dict

    config.update("jax_enable_x64", True)     # use double rather than single precision floats
    config.update('jax_platform_name', 'cpu') # use CPU rather than GPU

    from pathlib import Path as _Path
    public_dir = str(_Path(__file__).parent / "public")
    return MCMC, NUTS, public_dir, jax, jnp, jr, np, pd, plt


@app.cell
def _(plt):
    plt.rcParams["figure.figsize"] = (10, 8)
    return


@app.cell(hide_code=True)
def _(mo, qr_base64):
    mo.md(rf"""
    ## Uncertainty in Models and Data

    **James Kermode** <br>
    School of Engineering <br>
    University of Warwick

    <div style="display: flex; gap: 2em; align-items: flex-start; margin-top: 0.5em;">
    <div style="flex: 1;">

    ### Objectives

    - Give a brief overview of predictive modelling and uncertainty quantification
    - Explain key ideas behind inverse problems and model calibration
    - Demonstrate how a simple catalysis inverse problems can be solved in three ways:
        - classical optimisation with a mechanistic ODE model
        - scientific machine learning with a hybrid mechanistic/ML model
        - in a fully Bayesian fashion using MCMC sampling

    </div>
    <div style="text-align: center; flex-shrink: 0; padding-top: 1em;">
    <img src="data:image/png;base64,{qr_base64}" alt="QR Code" style="width: 480px; height: 480px;" /><br>
    <a href="https://sciml.warwick.ac.uk/demos/uq-kinetics/" style="font-size: 0.9em;">sciml.warwick.ac.uk/demos/uq-kinetics</a><br>
    <a href="https://github.com/kermodegroup/demos" style="font-size: 0.9em;">github.com/kermodegroup/demos</a>
    </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(public_dir, mo):
    mo.md(rf"""
    ## Research Environment at Warwick

    {mo.center(mo.hstack([mo.vstack([mo.md("EPSRC Centre for Doctoral Training in     Modelling of Heterogeneous Systems (HetSys CDT)"),
                                     mo.image(f"{public_dir}/hetSys-schematic.png"),
                                     mo.image(f"{public_dir}/quantum-atomistic-continuum.png")],
                                     heights=[.5, 2, .5], justify="center", gap=0, align="center"),
                          mo.vstack([mo.md("Warwick Centre for Predictive Modelling (WCPM)"),
                                     mo.image(f"{public_dir}/wcpm-schematic.png")])
                         ], widths=[1.2, 1], align="center")
              )}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verification and Validation

    - **Verification** is the process of demonstrating that a simulation code solves the underlying mathematical model correctly. This requires answering questions such as:
      - Does the code solve the equations it claims to?
      - How big is the numerical error in the result?
      - How will this error change with mesh size, timestep, etc.

    - **Validation** addresses the question of whether the mathematical model is appropriate for the system of interest: are we solving the correct equations?

    - Verification is typically a mathematical/computational science problem, while validation is a scientific problem. Verification must be completed before validation can start!

    - Validation relates to the aphorism attributed to [George Box](https://en.wikipedia.org/wiki/All_models_are_wrong)
    > All models are wrong, but some are useful

    - i.e. we need to know whether the model is appropriate for the target domain of interest. This is typically done by comparing simulation outputs with experimental results.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Uncertainty Quantification

    - The goal of Uncertainty Quantification (UQ) is to put error bars on the outputs of a simulation
    - Sources of error include parametric uncertainty, numerical error in the simulation, and limitations in the mathematical model
    - UQ is rooted in statistics and probability, but is an interdisciplinary challenge because details of the physics/chemistry/biology/engineering problem are key

    - The main steps in UQ are:
        - Identifying quantities of interest (problem dependent)
        - Modelling the uncertainties in the input parameters with probability distributions
        - Selecting the most important parameters (e.g. using sensitivity analysis)
        - Propagation of uncertain inputs through the simulation
        - Determining how uncertainties affect predictions, and updating assumptions of input parameter distributions using data (inverse problems)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sources of Uncertainty

    - **Aleatoric Uncertainties** arise from inherent randomness in a system (*aleator* is Latin for dice player), and can be described by probability distributions.
    - **Epistemic Uncertainties** arise from our lack of knowledge, typically due to approximations made in the mathematical model of the system.
    """)
    return


@app.cell(hide_code=True)
def _(public_dir, mo):
    mo.md(rf"""
    ## Uncertainty Quantification Across the Scales

    In the HetSys CDT, we are particularly interested in multiscale modelling. The addition of reliable uncertainty quantification on model outputs is in its infancy, and has the potential to make models much more powerful.

    {mo.center(mo.image(f"{public_dir}/hetsys-multiscale-fig.png", width="60%"))}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calibration and Inverse Problems

    - In a *calibration* or *inverse* problem we have a computer model, with a (potentially large) number of adjustable parameters, that we think is adequete to predict a given QoI, for which we also have experimental measurements.
    - The goal is to adjust (calibrate) the parameters of the model to match the experiment
    - We classify the model inputs into *input parameters*  $\mathbf{x}$  common to simulation and experiment, and *calibration parameters* $\mathbf{t}$ that only exist in the simulation.
    - The noisy experimental values can be written as $\mathbf{y}(\mathbf{x}) = \mathbf{f}(\mathbf{x}, \mathbf{t}) + \boldsymbol\epsilon$
    where $\mathbf{f}(\mathbf{x}, \mathbf{t})$ are the model outputs and $\boldsymbol\epsilon$ represents random noise
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classical Approach to Inverse Problems

    - Inverse problems can be framed as optimisation problems by minimising the discrepancy between the model and the data by adjusting the calibration parameters $\mathbf{t}$, often using a least squares loss function

    $$
    L(\mathbf{t}) = \frac{1}{2} \| \mathbf{f}(\mathbf{x}, \mathbf{t}) - \mathbf{y}(\mathbf{x}) \|^2 = \frac{1}{2} \sum_{i=1}^M ( f(\mathbf{x}_i, \mathbf{t}_i) - y_i )^2
    $$

    - Under this approach the calibrated values of the parameters are given by

    $$
    \mathbf{t}^* = \min_{\mathbf{t}} L(\mathbf{t}) = \min_{\mathbf{t}} \frac12 \| \mathbf{f}(\mathbf{x}, \mathbf{t}) - \mathbf{y} \|^2
    $$

    - Solving this optimisation problem efficiently typically requires derivatives of the model $\mathbf{f}(\mathbf{x}, \mathbf{t})$ with respect to the calibration parameters $\mathbf{t}$, which can be be difficult or expensive to obtain (although automatic differentiation may be an option).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Drawbacks of classical approach to inverse problems

    - The classical approach to calibration often works well, but there are some key deficiencies of the classical approach:

        + In general, the problem is ill-posed: there may be no solutions, or more than one solution may exist.
        + There is no apparent way to quantify uncertainties.
        + There is no systematic way to account for prior knowledge.

    - We will next look at a Bayesian approach that addresses these limitations.
    """)
    return


@app.cell(hide_code=True)
def _(public_dir, mo):
    mo.md(rf"""
    ## Bayesian Approach to Inverse Problems

    - *Uncertainty Propagation* is equivalent to *forward uncertainty quantification*
    - We propagate *assumed* distributions of input parameters through a simulation code to determine distribution of outputs

    {mo.center(mo.image(f"{public_dir}/uq-forward.png", width="60%"))}
    """)
    return


@app.cell(hide_code=True)
def _(public_dir, mo):
    mo.md(rf"""
    ### Inverse Uncertainty Quantification

    - In *inverse uncertainty quantification*, we want to infer the distribution of the input (calibration) parameters from data, which could come either from experiment or high-level simulation (e.g. calibrating a reaction model with data from computational chemistry)

    {mo.center(mo.image(f"{public_dir}/uq-inverse-1.png", width="60%"))}
    """)
    return


@app.cell(hide_code=True)
def _(public_dir, mo):
    mo.md(rf"""
    ### Inverse + Forward Uncertainty Quantification

    - Putting everything together gives an improved estimated of the distribution for the output quantities of interest.

    {mo.center(mo.image(f"{public_dir}/uq-inverse-2.png", width="60%"))}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayesian Approach to Inverse Problems

    If we have priors for the calibration parameters and the measurement error we can use Bayes' rule to update our estimate for the calibration parameters $\mathbf{t}$ given a set of measurements $\mathbf{y}$ and non-random input parameters $\mathbf{x}$

    $$
    \mathbb{P}(\mathbf{t}, \boldsymbol\epsilon | \mathbf{y}, \mathbf{x}) = \frac{\mathbb{P}(\mathbf{y} | \mathbf{x}, \mathbf{t}, \boldsymbol\epsilon)\, \mathbb{P}(\mathbf{t})\, \mathbb{P}(\boldsymbol\epsilon)}
    {\int \mathrm{d}\mathbf{t} \int \mathrm{d}\boldsymbol\epsilon\, \mathbb{P}(\mathbf{y} | \mathbf{x}, \mathbf{t} \boldsymbol\epsilon)\, \mathbb{P}(\mathbf{t})\, \mathbb{P}(\boldsymbol\epsilon) }
    $$

    where the terms on the numerator are the likelihood of the data given the model and the noise, a prior for the parameters and a noise prior, respectively, and the denominator is a normalisation constant.

    The calibration problem reduces to sampling from the posterior probability distribution $\mathbb{P}(\mathbf{t}, \boldsymbol\epsilon | \mathbf{y}, \mathbf{x})$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayesian Calibration

    - Let's assume the measurement errors are independent and each is distributed as $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$, where $\sigma$ is a known standard deviation
    - The likelihood of the data can then be written as a product of Gaussians with means given by the model output $\mathbf{f}(\mathbf{x},\mathbf{t})$ and standard deviations $\sigma$

    $$
    \mathbb{P}(\mathbf{y} | \mathbf{x}, \mathbf{t}, \sigma) = \mathcal{N}(\mathbf{y}| f(\mathbf{x}, \mathbf{t}), \sigma^2 I) = \prod_{i=1}^N \mathcal{N}(y(\mathbf{x}_i) | f(\mathbf{x}_i, \mathbf{t}), \sigma^2)
    $$

    - The log likelihood is

    $$
    \log \mathbb{P}(\mathbf{y} | \mathbf{x}, \mathbf{t}, \sigma) = -\frac1{2\sigma^2} \sum_{i=1}^N \left(f(\mathbf{x}_i, \mathbf{t}) - y_i \right)^2 - N \log \sigma - \frac{N}2 \log 2\pi
    $$

    - For fixed $N$, maximising the (log) likelihood will lead to the same solution for the parameters $\mathbf{t}$ as minimising the least-squares cost function $L(\mathbf{t})$

    - Maximising the posterior instead of the likelihood incorporates prior information.

    - Sampling from the posterior allows uncertainties to be quantified.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example: Catalytic Conversion of Nitrate to Nitrogen

    ### Classical Approach to Inverse Problems

    - Here we consider the catalytic conversion of nitrate ($\text{NO}_3^-$) to nitrogen ($\text{N}_2$) and other
    by-products by electrochemical means. The example is adapted from Example 3.1 of [(Tsilifis, 2014)](http://arxiv.org/abs/1410.5522).

    - The mechanism that is followed is complex and not well understood.
    The experiment of [(Katsounaros, 2012)](http://www.sciencedirect.com/science/article/pii/S0013468612005208) confirmed the
    production of nitrogen ($\text{N}_2$), ammonia
    ($\text{NH}_3$), and nitrous oxide ($\text{N}_2\text{O}$) as final products
    of the reaction, as well as the intermediate production of nitrite ($\text{NO}_2^-$).

    - Experimental data is available. The time is measured in minutes and the concentrations are measured in $\text{mmol}\cdot\text{L}^{-1}$.
    """)
    return


@app.cell
def _(public_dir, pd):
    catalysis_data = pd.read_csv(f"{public_dir}/catalysis.csv")
    catalysis_data
    return (catalysis_data,)


@app.cell
def _(catalysis_data):
    catalysis_data.plot(style='s', x=0, figsize=(12, 6))
    return


@app.cell(hide_code=True)
def _(catalysis_data, mo):
    mo.vstack([
        mo.md(
        rf"""
        We would expect the total mass to be conserved during the catalysis process. However, if we sum along the columns in the data we find the total concentration is not constant.

        """),
        catalysis_data.sum(axis=1)
    ])
    return


@app.cell(hide_code=True)
def _(public_dir, mo):
    mo.md(rf"""
    ### Catalytic Reactions

    This imbalance is explained by the presence of an unknown precursor which we shall call X. The paper suggests the follow mechanism:

    {mo.center(mo.image(f"{public_dir}/process.png", width="50%"))}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This mechanism can be represented by a coupled system of ordinary differential equations (ODEs) as follows

    $$
    \begin{align*}
    \frac{d \left[\text{NO}_3^-\right]}{dt} &= -k_1\left[\text{NO}_3^-\right], \\
        \frac{d\left[\text{NO}_2^-\right]}{dt} &= k_1\left[\text{NO}_3^-\right] - (k_2 + k_4 +
        k_5)[\text{NO}_2^-], \\
        \frac{d \left[\text{X}\right]}{dt} &= k_2 \left[\text{NO}_2^-\right] - k_3 [X],\\
        \frac{d \left[\text{N}_2\right]}{dt} &= k_3 \left[\text{X}\right], \\
        \frac{d \left[\text{NH}_3\right]}{dt} &= k_4 \left[\text{NO}_2^-\right],\\
        \frac{d \left[\text{N}_2O\right]}{dt} &= k_5 \left[\text{NO}_2^-\right],
    \end{align*}
    $$

    where $[\cdot]$ denotes the concentration of a quantity, and
    $k_i > 0$, $i=1,...5$ are the *kinetic rate constants*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simplifying Notation

    Let's define:

    $$
    \begin{array}{ccc}
    z_1 &:=& \left[\text{NO}_3^-\right],\\
    z_2 &:=& \left[\text{NO}_2^-\right],\\
    z_3 &:=& \left[\text{X}\right],\\
    z_4 &:=& \left[\text{N}_2\right],\\
    z_5 &:=& \left[\text{NH}_3\right],\\
    z_6 &:=& \left[\text{N}_2O\right],
    \end{array}
    $$

    which we can combine into the vector

    $$
    \mathbf{z} = (z_1,z_2,z_3,z_4,z_5,z_6) \in \mathbb{R}^6
    $$

    and also define a matrix $A(k_1, \ldots k_5)$ which encapsulates the dynamics in the form

    $$
    \dot{\mathbf{z}} = A(k_1,\dots,k_5)\mathbf{z},
    $$

    with initial conditions

    $$
    \mathbf{z}(0) = z_0 = (500, 0, 0, 0, 0, 0)\in\mathbb{R}^6,
    $$

    from the experimental data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Forward solver

    Before we can address the inverse problem of calibraring the $k_i$ on the experimental data, we first need a forward solver for this system. We denote the solution of the system at time $t$ by:

    $$
    \mathbf{z}(k_1,\dots,k_5, t).
    $$

    We can incorporate chemical/physical intuitions and constraints in the construction of our solver:

    + The $k_i$ have units of inverse time. We can scale them by the total time to give dimensionless parameters $\hat{x}_i = 180k_i$.

    + $k_i$ is positive, therefore $\hat{x_i}$ must be positive. We can ensure this by working with the logarithm of $\hat{x_i}$:

    $$
    x_i = \log \hat{x_i} = \log 180k_i.
    $$

    + We define the parameter vector $\mathbf{x} = (x_1,\dots,x_5)\in \mathbb{R}^5.$

    - The dynamical system can be written $\dot{\mathbf{z}} = A(\mathbf{x})\,\mathbf{z}$.

    - The result of integrating the ODEs to predict concentrations at the known times $t_j$ with parameters $\mathbf{x}$ is $\mathbf{z}(\mathbf{x}, t_j)$. This can be compared with the experimental data $\mathbf{y}$.
    """)
    return


@app.cell
def _(catalysis_data, jnp):
    from jax.experimental.ode import odeint

    def A(x):
        """
        Dynamical system matrix with parameters x_i = log(180 k_i), i=1,...,5
        """
        k = jnp.exp(x) / 180.0
        _res = jnp.array([[-k[0], 0, 0, 0, 0, 0], 
                          [k[0], -(k[1] + k[3] + k[4]), 0, 0, 0, 0], 
                          [0, k[1], -k[2], 0, 0, 0], 
                          [0, 0, k[2], 0, 0, 0], 
                          [0, k[3], 0, 0, 0, 0], 
                          [0, k[4], 0, 0, 0, 0]])
        return _res

    def g(z, t, x):
        """
        Right hand side of the ODE system at (z, t) with parameters x
        """
        return A(x) @ z

    def Z(x, t):
        """
        The full solution of the dynamical system for parameters x at time t.

        Uses `jax.experimental.ode.odeint()` to integrate g(z, t, x)

        """
        return odeint(g, z0, t, x)    

    def F(x, t):
        """
        The matrix function F(x,t) - matches to Y

        This should call Z(x, t) and extract the results for all species except X
        """
        _res = Z(x, t)
        return jnp.hstack([_res[:, :2], _res[:, 3:]])

    def f(x, t):
        """
        The vector-valued function f(x,t) - matches to y
        """
        return F(x, t).flatten()

    z0 = jnp.array([500., 0., 0., 0., 0., 0.])
    t_exp = jnp.array([30. * j for j in range(0, 7)])

    # The experimental data as a matrix
    Y = catalysis_data.values[:, 1:]

    # The experimental data as a vector
    y = jnp.array(Y.flatten())
    return Y, Z, f, g, t_exp, y, z0


@app.cell
def _(mo):
    x1 = mo.ui.slider(-2.0, 2.0, 0.05, value=0, label="$x_1$")
    x2 = mo.ui.slider(-2.0, 2.0, 0.05, value=0, label="$x_2$")
    x3 = mo.ui.slider(-2.0, 2.0, 0.05, value=0, label="$x_3$")
    x4 = mo.ui.slider(-2.0, 2.0, 0.05, value=0, label="$x_4$")
    x5 = mo.ui.slider(-2.0, 2.0, 0.05, value=0, label="$x_5$")
    sliders = mo.vstack([mo.hstack([x1, x4]), 
                         mo.hstack([x2, x5]),
                         mo.hstack([x3])])
    return sliders, x1, x2, x3, x4, x5


@app.cell(hide_code=True)
def _(Z, catalysis_plot, jnp, mo, np, sliders, x1, x2, x3, x4, x5):
    _x = jnp.array([x1.value, x2.value, x3.value, x4.value, x5.value])
    _t = np.linspace(0, 180, 100)
    _Yp = Z(_x, _t)

    mo.vstack([
        mo.md(
        rf"""
        ## Manual Calibration

        **x** = {np.round(_x, 2)}
        """),
        sliders,
        catalysis_plot(_t, _Yp)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inverse Problem as an Optimisation Problem

    Given the solver $\mathbf{f}(\mathbf{x}, t)$ we can use the classical approach to inverse problems to reframe the problem of finding the parameter vector $\mathbf{x}$ which best matches the experimental results as an optimisation problem, where we seek the solution which minimise the loss function

    $$
    L(\mathbf{x}, \mathbf{t}, \mathbf{y}) = \frac{1}{2} \| \mathbf{f}(\mathbf{x}, \mathbf{t}) - \mathbf{y}) \|^2
    $$

    i.e.

    $$
    \mathbf{x}_\mathrm{opt} = \arg\min_{\mathbf{x}} L(\mathbf{x})
    $$

    We can just-in-time compile the function for speed using JAX. We also scale for numerical stability. We can compute the derivatives of the loss function with respect to the parameters using automatic differentiation, i.e. the gradient vector $\nabla_\mathbf{x} L$ - note we are differentiating through the ODE solver itself!

    ```python
    @jax.jit
    def L(x, t, y):
        scale = 500.0
        return 0.5 * jnp.sum((f(x, t) / scale - y / scale) ** 2)

    dL_dx = jax.grad(L)
    ```
    """)
    return


@app.cell
def _(mo):
    button = mo.ui.run_button(label="Optimize")
    return (button,)


@app.cell(hide_code=True)
def _(
    L,
    Z,
    button,
    catalysis_plot,
    dL_dx,
    jnp,
    mo,
    np,
    t_exp,
    x1,
    x2,
    x3,
    x4,
    x5,
    y,
):
    from scipy.optimize import minimize

    x_opt = jnp.array([x1.value, x2.value, x3.value, x4.value, x5.value])
    if button.value:
        _res = minimize(L, x_opt, jac=dL_dx, args=(t_exp, y))
        x_opt = _res.x

    _t = np.linspace(0, 180, 100)
    _Yp = Z(x_opt, _t)

    mo.vstack([
        mo.md(
        rf"""
        ## Optimisation-based Calibration

        **x** = {np.round(x_opt, 2)}
        """),
        button,
        catalysis_plot(_t, _Yp)
    ])
    return


@app.cell
def _(f, jax, jnp):
    @jax.jit
    def L(x, t, y):
        scale = 500.0
        return 0.5 * jnp.sum((f(x, t) / scale - y / scale) ** 2)

    dL_dx = jax.grad(L)
    return L, dL_dx


@app.cell
def _(catalysis_data, plt):
    def catalysis_plot(t, Y, Y_samples=None, Y_lower=None, Y_upper=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        catalysis_data.plot(ax=ax, style='s', x=0)
        colors = ['C0', 'C1', 'C5', 'C2', 'C3', 'C4']
        labels = ['Model NO$_3$', 'Model NO$_2$', 'Model X', 'Model N$_2$', 'Model NH$_3$', 'Model N$_2$0']
        for _i, (color, label) in enumerate(zip(colors, labels)):
            ax.plot(t, Y[:, _i], color=color, label=label)
        if Y_samples is not None:
            for Y in Y_samples:
                for _i, color in enumerate(colors):
                    ax.plot(t, Y[:, _i], color=color, alpha=0.2)
        if Y_lower is not None and Y_upper is not None:
            for _i, (color, label) in enumerate(zip(colors, labels)):
                ax.fill_between(t, Y_lower[:, _i], Y_upper[:, _i], color=color, alpha=0.5)
        plt.legend(loc='upper right')
        return ax

    return (catalysis_plot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scientific Machine Learning and Neural ODEs

    - Rather than using a standard optimiser, we could use stochastic gradient descent to choose the rate constants.

    - If we don't know the functional form of the mode or we only know it approximately we could expand the flexibility of the model using a neural network, for example by including a multi-input/single-output neural network $\mathcal{G}(\mathbf{z}; \mathbf{w}) : \mathbb{R}^6 \to \mathbb{R}$ with parameters $\mathbf{w}$ (optimised concurrently with $\mathbf{x}$)

    $$
    \dot{\mathbf{z}} = A(\mathbf{x}) \mathbf{z} + \nabla \mathcal{G}(\mathbf{z},\mathbf{w})
    $$

    - This is an example of a *Neural ODE* approach

    - In either case, the loss function we seek to minimise is the same as before (the discrepancy between the model and the experiment)
    """)
    return


@app.cell
def _(mo):
    run_sgd = mo.ui.run_button(label="Run SGD...")
    return (run_sgd,)


@app.cell
def _(Y, catalysis_plot, g, jax, jnp, mo, np, run_sgd, t_exp, z0):
    import equinox as eqx
    from optax import adam
    import diffrax

    def solve(params, ts, t0=0.0, t1=180.0):
        sol = diffrax.diffeqsolve(diffrax.ODETerm(lambda t, y, args: g(y, t, params)), # vector -> vector
                                  diffrax.Tsit5(), t0=t0, t1=t1, dt0=1.0, y0=z0,
                                  saveat=diffrax.SaveAt(ts=ts))                           # save y(t) at fixed times
        return sol.ys

    def loss(params, t, y):
        scale = 500.0
        res = solve(params, t)
        pred_y = jnp.hstack([res[:, :2], res[:, 3:]])
        return jnp.mean((pred_y / scale - y / scale)**2)

    def dataloader(arrays, batch_size, *, key):
        dataset_size = arrays[0].shape[0]
        assert all(array.shape[0] == dataset_size for array in arrays)
        indices = jnp.arange(dataset_size)
        while True:
            perm = jax.random.permutation(key, indices)
            (key,) = jax.random.split(key, 1)
            start = 0
            end = batch_size
            while end < dataset_size:
                batch_perm = perm[start:end]
                yield tuple(array[batch_perm] for array in arrays)
                start = end
                end = start + batch_size

    @eqx.filter_jit                                          # compile, filtering out non-array data
    def make_step(model, opt_state, t, y):
        grads = eqx.filter_grad(loss)(model, t, y)           # gradients wrt model, filtering out non-array data
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)            # use eqx.apply_update() as it understands non-array data
        return model, opt_state    

    if run_sgd.value:
        params = jax.random.normal(jax.random.PRNGKey(1), [5])

        optim = adam(learning_rate=1e-3)                         # learning rate: you can experiment with this
        # params, static = eqx.partition(mlp, eqx.is_array)        # separate model into variables and fixed data
        opt_state = optim.init(params)


        with mo.status.progress_bar(
            total=5000,
            title="Stochastic gradient descent",
            subtitle=f"Initial loss {loss(params, t_exp, Y):.4f}",
            show_eta=True,
            show_rate=True,
            remove_on_exit=True
        ) as bar:
            for step in range(5000):    #                     dataloader((t_exp, Y), batch_size=10,
                #                                key=jax.random.PRNGKey(1))):
                # order = _t.argsort()                                  # ensure batches are in increasing time order
                # _t, _y = _t[order], _y[order]
                _t, _y = t_exp, Y
                # if step < 500:                                      # avoid local minima by training on first 20%
                #     _t, _y = _t[:int(len(_t)*0.2)], _y[:int(len(_t)*0.2)]
                params, opt_state = make_step(params, opt_state, _t, _y)
                if step % 1000 == 0:
                    bar.update(subtitle=f'Step {step} loss {loss(params, t_exp, Y):.4f}')
                else:
                    bar.update()
                    # print(f"{step=} loss={loss(params, t_exp, Y)}")

        _t = np.linspace(0, 180, 200)
        _y = solve(params, _t)
        outputs = [ catalysis_plot(_t, _y) ]
    else:
        outputs = []

    mo.vstack([
        mo.md(
        rf"""
        ## Stochastic gradient descent Calibration

        """),
        run_sgd] + outputs)
    return adam, diffrax, eqx, optim, params


@app.cell
def _(mo):
    run_sciml = mo.ui.run_button(label="Run SciML...")
    return (run_sciml,)


@app.cell
def _(
    Y,
    adam,
    catalysis_plot,
    diffrax,
    eqx,
    g,
    jax,
    jnp,
    mo,
    np,
    optim,
    params,
    run_sciml,
    t_exp,
    z0,
):
    class ODEModel(eqx.Module):
        params: jax.Array
        mlp: eqx.nn.MLP

        def __init__(self, params, key):
            self.params = params
            self.mlp = mlp = eqx.nn.MLP(in_size=6, out_size="scalar", width_size=16, depth=3,  # vector -> scalar MLP with
                     activation=jax.nn.tanh, key=key)

        def __call__(self, ts, z0, t0=0.0, t1=180.0):
            sol = diffrax.diffeqsolve(diffrax.ODETerm(lambda t, z, args: g(z, t, self.params) + jax.grad(self.mlp)(z)), # vector -> vector
                                   diffrax.Tsit5(), t0=t0, t1=t1, dt0=1.0, y0=z0,
                                   saveat=diffrax.SaveAt(ts=ts))                           # save z(t) at fixed times
            return sol.ys

    def ode_loss(model, t, z, z0):
        scale = 500.0
        res = model(t, z0)
        pred_z = jnp.hstack([res[:, :2], res[:, 3:]])
        return jnp.mean((pred_z / scale - z / scale)**2)

    @eqx.filter_jit                                          # compile, filtering out non-array data
    def make_step2(model, opt_state, t, y):
        grads = eqx.filter_grad(ode_loss)(model, t, y, z0)           # gradients wrt model, filtering out non-array data
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)            # use eqx.apply_update() as it understands non-array data
        return model, opt_state

    if run_sciml.value:
        model = ODEModel(params=params, key=jax.random.PRNGKey(0))    
        optim2 = adam(learning_rate=3e-4)                         # lower learning rate: you can experiment with this
        params2, static = eqx.partition(model, eqx.is_array)        # separate model into variables and fixed data
        opt_state2 = optim2.init(params2)

        with mo.status.progress_bar(
            total=5000,
            title="SciML optimisation",
            subtitle=f"Initial loss {ode_loss(model, t_exp, Y, z0):.4f}",
            show_eta=True,
            show_rate=True,
            remove_on_exit=True
        ) as bar2:
            for _step in range(5000):    #                     dataloader((t_exp, Y), batch_size=10,
                #                                key=jax.random.PRNGKey(1))):
                # order = _t.argsort()                                  # ensure batches are in increasing time order
                # _t, _y = _t[order], _y[order]
                _t, _y = t_exp, Y
                if _step < 500:                                      # avoid local minima by training on first 20%
                    _t, _y = _t[:int(len(_t)*0.2)], _y[:int(len(_t)*0.2)]
                model, opt_state2 = make_step2(model, opt_state2, _t, _y)
                if _step % 1000 == 0:
                    bar2.update(subtitle=f'Step {_step} loss {ode_loss(model, t_exp, Y, z0):.4f}')
                else:
                    bar2.update()            
                # if _step % 1000 == 0:
                #     print(f"{_step=} loss={ode_loss(model, t_exp, Y, z0)}")        

        _t = np.linspace(0, 180, 200)
        _y = model(_t, z0)
        _outputs = [ catalysis_plot(_t, _y) ]
    else:
        _outputs = []

    mo.vstack([
        mo.md(
        rf"""
        ## SciML-based Calibration

        """),
        run_sciml] + _outputs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Discussion of Optimisation-based Approaches

    - Fit to data is pretty good, despite the optimisation problem being ill-posed
    - Automatic differentiation (AD) speeds up the optimisation
    - Flexibility of modelling can be enchanced with SciML ideas
    - Disadvantages include:
        - Problem may be ill-posed: could be too many solutions, or none!
        - No way to incorporate prior information
        - No way to handle uncertainty in inputs or outputs
    """)
    return


@app.cell(hide_code=True)
def _(axs2, mo):
    mo.vstack([mo.md(
        rf"""
    ## Markov Chain Monte Carlo

    - MCMC can be used to generate samples from the full posterior for the catalysis problem using MCMC
    - This takes a bit longer, so I won't demonstrate in real time, but allows correlations between parameters to be assessed.

    """
    ), axs2[-1, 1]])
    return


@app.cell
def _(f, jax, jnp):
    @jax.jit
    def log_post(x, sigma, gamma, t, y):
        likelihood = jnp.sum(((f(x, t) - y) / sigma) ** 2)
        prior = jnp.sum((x / gamma) ** 2)
        return -0.5 * (likelihood + prior)

    return (log_post,)


@app.cell
def _(mcmc):
    import arviz as az
    D = az.from_numpyro(mcmc)
    axs2 = az.plot_pair(D, marginals=True, scatter_kwargs={'s': 50}, figsize=(12, 6))
    return D, axs2


@app.cell
def _(MCMC, NUTS, jnp, jr, log_post, t_exp, y):
    _sigma = 10.0
    _gamma = 100.0

    mcmc = MCMC(NUTS(potential_fn=lambda x: -log_post(x, _sigma, _gamma, t_exp, y)),  num_warmup=500, num_samples=500, num_chains=1)
    key = jr.PRNGKey(0)
    mcmc.run(key, init_params=jnp.array([1.36, 1.66, 1.35, -1.05, -0.16]), progress_bar=False)
    return (mcmc,)


@app.cell(hide_code=True)
def _(D, Z, catalysis_plot, np):
    _t = np.linspace(0, 180, 200)

    X_rest = D["posterior"]['Param:0'][0, :, :]
    Y_rest = np.zeros((X_rest.shape[0], 200, 6))
    for _i in range(X_rest.shape[0]):
        Y_rest[_i, :, :] = Z(np.array(X_rest[_i, :]), _t)

    catalysis_plot(_t, np.percentile(Y_rest, 50, axis=0), Y_samples=Y_rest[np.random.choice(len(Y_rest), 10), :, :], Y_lower=np.percentile(Y_rest, 2.5, axis=0), Y_upper=np.percentile(Y_rest, 97.5, axis=0))
    return


@app.cell(hide_code=True)
def _(mo, qr_base64):
    mo.md(rf"""
    ## Conclusions

    <div style="display: flex; gap: 2em; align-items: flex-start;">
    <div style="flex: 1;">

    - Inverse problems can be reformulated as optimisation problems through the definition of a loss function that is minimimsed to give a good match between a model and data.

    - This classical approach to calibration often works well, but there may be no solutions, or more than one solution, and there no obvious way to quantify uncertainties or to account for prior knowledge.

    - Bayesian approaches address these limitations, but can be prohibitively expensive. The cost can be reduced with approximations such as Laplace approximation or variational inference.

    - Scientific Machine Learning (SciML) ideas such as Neural ODEs allow the limitations of mechanistic model to be overcome using a more flexible model such as a neural network.

    - This can be combined with Bayesian inference in a Bayesian neural network, allowing both epistemic and aleatoric uncertainty to be quantified, and reducing risks of overfitting.

    - Now collaborating with Linda Wanika and Mike Chappell to apply some of these ideas to pharmacokinetic models in context of ERAMET project.

    *The ERAMET project has received funding from the European Commission's Horizon Europe Programme under the grant agreement number 101137141*

    </div>
    <div style="text-align: center; flex-shrink: 0; padding-top: 1em;">
    <img src="data:image/png;base64,{qr_base64}" alt="QR Code" style="width: 480px; height: 480px;" /><br>
    <a href="https://sciml.warwick.ac.uk/demos/uq-kinetics/" style="font-size: 0.9em;">sciml.warwick.ac.uk/demos/uq-kinetics</a><br>
    <a href="https://github.com/kermodegroup/demos" style="font-size: 0.9em;">github.com/kermodegroup/demos</a>
    </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _():
    import qrcode
    import io
    import base64

    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://sciml.warwick.ac.uk/demos/uq-kinetics/')
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    qr_base64 = base64.b64encode(buffer.read()).decode()
    return (qr_base64,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
