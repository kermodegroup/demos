# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.2.5",
#     "qrcode[pil]==8.2",
#     "jax",
#     "jaxlib",
#     "tinygp",
#     "altair",
#     "pandas",
# ]
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.Html('''
    <style>
        body, .marimo-container {
            margin: 0 !important;
            padding: 0 !important;
            height: 100vh;
            overflow: hidden;
        }

        .app-header {
            padding: 8px 16px;
            border-bottom: 1px solid #dee2e6;
            background-color: #fff;
        }

        .app-layout {
            display: flex;
            height: calc(100vh - 80px);
            align-items: flex-start;
            justify-content: center;
            gap: 2em;
            padding: 1em 0.5em;
        }

        .app-plot {
            flex: 1;
            min-width: 0;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            align-items: stretch;
            z-index: 1;
            overflow: hidden;
        }

        .app-plot img,
        .app-plot svg {
            max-width: 100%;
            height: auto;
        }

        .app-sidebar {
            z-index: 10;
            position: relative;
            display: flex;
            flex-direction: column;
            gap: clamp(0.3em, 1vh, 1em);
            padding: clamp(0.5em, 1.5vh, 1.5em);
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            width: 30%;
            min-width: 280px;
            max-width: 400px;
            flex-shrink: 0;
            max-height: calc(100vh - 120px);
            overflow-y: auto;
        }

        @media (max-width: 768px) {
            .app-layout {
                flex-direction: column;
                height: auto;
                overflow-y: auto;
            }
            .app-sidebar {
                width: 100%;
                max-width: none;
                min-width: auto;
                max-height: none;
            }
        }

        .app-sidebar h4 {
            margin: clamp(0.3em, 1vh, 1em) 0 clamp(0.2em, 0.5vh, 0.5em) 0;
            font-size: 0.9em;
            color: #495057;
            border-bottom: 1px solid #dee2e6;
            padding-bottom: 0.2em;
        }

        .app-sidebar h4:first-child {
            margin-top: 0;
        }

        .app-sidebar .marimo-ui-element {
            margin-bottom: clamp(0.1em, 0.3vh, 0.3em);
        }
    </style>
    ''')
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import tinygp

    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'cpu')

    return alt, jax, jnp, jr, mo, np, pd, tinygp


@app.cell(hide_code=True)
def _():
    import qrcode
    import io
    import base64

    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://sciml.warwick.ac.uk/live/bayesian-optimization-demo/')
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    qr_base64 = base64.b64encode(buffer.read()).decode()
    return (qr_base64,)


@app.cell(hide_code=True)
def _(mo, qr_base64):
    header = mo.Html(f'''
    <div class="app-header">
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 0; padding: 0;">
            <div>
                <p style='font-size: 24px; margin: 0; padding: 0; line-height: 1.3;'><b>Bayesian Optimization Demo</b>
                <br><span style="font-size: 16px;"><i>Live demos:</i>
                <a href="https://sciml.warwick.ac.uk/" target="_blank" style="color: #0066cc; text-decoration: none;">sciml.warwick.ac.uk</a>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <i>Code:</i>
                <a href="https://github.com/kermodegroup/demos" target="_blank" style="color: #0066cc; text-decoration: none;">github.com/kermodegroup/demos</a>
                </span></p>
            </div>
            <img src="data:image/png;base64,{qr_base64}" alt="QR Code" style="width: 100px; height: 100px; flex-shrink: 0;" />
        </div>
    </div>
    ''')
    return (header,)


@app.cell(hide_code=True)
def _(mo):
    # State for custom code
    get_custom_code, set_custom_code = mo.state(
        "# Define y = f(x) where x is a JAX array\n"
        "# Available: jnp, x\n"
        "y = jnp.sin(3 * x) * jnp.exp(-0.5 * x**2)"
    )

    # Objective function selection
    objective_dropdown = mo.ui.dropdown(
        options={
            'Ackley': 'ackley',
            'Forrester': 'forrester',
            'Gramacy-Lee': 'gramacy_lee',
            'Custom': 'custom',
        },
        value='Ackley',
        label='Objective Function'
    )

    # Domain range
    domain_slider = mo.ui.range_slider(
        start=-4, stop=4, step=0.5,
        value=[-2.0, 2.0],
        label='Domain range'
    )

    # Kernel type selection
    kernel_dropdown = mo.ui.dropdown(
        options={
            'RBF (Squared Exponential)': 'rbf',
            'Matern 3/2': 'matern32',
            'Matern 5/2': 'matern52',
        },
        value='Matern 3/2',
        label='Kernel Type'
    )

    # Kernel hyperparameters
    variance_slider = mo.ui.slider(0.1, 5.0, 0.1, 1.0, label='Variance $v$')
    lengthscale_slider = mo.ui.slider(0.05, 2.0, 0.05, 0.3, label='Lengthscale $\\ell$')
    noise_slider = mo.ui.slider(0.001, 0.1, 0.001, 0.01, label='Noise $\\sigma_n$')

    # Acquisition function selection
    acquisition_dropdown = mo.ui.dropdown(
        options={
            'Thompson Sampling': 'thompson',
            'Expected Improvement (EI)': 'ei',
            'Lower Confidence Bound (LCB)': 'lcb',
        },
        value='Thompson Sampling',
        label='Acquisition Function'
    )

    # Acquisition function parameters
    beta_slider = mo.ui.slider(0.1, 5.0, 0.1, 2.0, label='LCB $\\beta$')
    n_thompson_slider = mo.ui.slider(1, 10, 1, 3, label='Thompson samples')

    # Optimization controls
    n_initial_slider = mo.ui.slider(1, 10, 1, 3, label='Initial points')
    seed_slider = mo.ui.slider(0, 100, 1, 42, label='Random seed')

    # Custom function code editor and accordion
    custom_code_editor = mo.ui.code_editor(
        value=get_custom_code(),
        language="python",
        min_height=120,
        on_change=set_custom_code,
    )

    custom_function_accordion = mo.accordion({
        "Custom Function Code": mo.vstack([
            mo.md("Define `y = f(x)`. Available: `jnp`, `x` (JAX array)."),
            custom_code_editor,
        ])
    }, lazy=True)

    return (
        objective_dropdown,
        domain_slider,
        kernel_dropdown,
        variance_slider,
        lengthscale_slider,
        noise_slider,
        acquisition_dropdown,
        beta_slider,
        n_thompson_slider,
        n_initial_slider,
        seed_slider,
        get_custom_code,
        set_custom_code,
        custom_code_editor,
        custom_function_accordion,
    )


@app.cell(hide_code=True)
def _(jnp):
    # Objective functions
    def ackley_1d(x):
        """Ackley function, minimum at x=0, f(0)=0."""
        a, b, c = 20.0, 0.2, 2 * jnp.pi
        return -a * jnp.exp(-b * jnp.abs(x)) - jnp.exp(jnp.cos(c * x)) + a + jnp.e

    def forrester_1d(x):
        """Forrester function on [0,1], rescaled to domain."""
        # Map x to [0,1] assuming x is roughly in [-2,2]
        x_scaled = (x + 2) / 4
        return (6 * x_scaled - 2)**2 * jnp.sin(12 * x_scaled - 4)

    def gramacy_lee_1d(x):
        """Gramacy-Lee function, rescaled."""
        # Map x to [0.5, 2.5] assuming x is roughly in [-2,2]
        x_scaled = (x + 2) / 2 + 0.5
        return jnp.sin(10 * jnp.pi * x_scaled) / (2 * x_scaled) + (x_scaled - 1)**4

    def get_objective_func(func_type, custom_code=None):
        """Return objective function and its true minimum location."""
        if func_type == 'ackley':
            return ackley_1d, 0.0  # minimum at x=0
        elif func_type == 'forrester':
            return forrester_1d, -1.04  # approximate minimum location
        elif func_type == 'gramacy_lee':
            return gramacy_lee_1d, -1.4  # approximate minimum location
        elif func_type == 'custom' and custom_code:
            def custom_func(x):
                try:
                    namespace = {'jnp': jnp, 'x': x}
                    exec(custom_code, {"__builtins__": {}}, namespace)
                    if 'y' not in namespace:
                        return jnp.full_like(x, jnp.nan)
                    return namespace['y']
                except Exception:
                    return jnp.full_like(x, jnp.nan)
            return custom_func, 0.0  # default minimum location
        return ackley_1d, 0.0

    return ackley_1d, forrester_1d, gramacy_lee_1d, get_objective_func


@app.cell(hide_code=True)
def _(jnp, tinygp):
    # GP utilities
    def build_gp(X, y, kernel_type, variance, lengthscale, noise):
        """Build tinygp GaussianProcess and condition on data."""
        if kernel_type == 'rbf':
            kernel = variance * tinygp.kernels.ExpSquared(scale=lengthscale)
        elif kernel_type == 'matern32':
            kernel = variance * tinygp.kernels.Matern32(scale=lengthscale)
        else:  # matern52
            kernel = variance * tinygp.kernels.Matern52(scale=lengthscale)

        gp = tinygp.GaussianProcess(kernel, X, diag=noise**2)
        return gp

    def gp_posterior(gp, y, X_test, noise):
        """Compute posterior mean and std at test points."""
        _, cond_gp = gp.condition(y, X_test, diag=noise**2)
        mean = cond_gp.mean
        std = jnp.sqrt(cond_gp.variance - noise**2)  # epistemic uncertainty
        return mean, std, cond_gp

    return build_gp, gp_posterior


@app.cell(hide_code=True)
def _(jnp):
    # Acquisition functions
    def thompson_sample(cond_gp, X_grid, key):
        """Thompson sampling: sample from posterior, return argmin."""
        sample = cond_gp.sample(key=key, shape=(1,)).flatten()
        idx = jnp.argmin(sample)
        return X_grid[idx], sample

    def expected_improvement(mean, std, y_best, xi=0.01):
        """Expected Improvement for minimization."""
        # EI = (y_best - mu - xi) * Phi(Z) + sigma * phi(Z)
        # where Z = (y_best - mu - xi) / sigma
        improvement = y_best - mean - xi
        Z = improvement / (std + 1e-9)

        # Standard normal PDF and CDF approximations
        phi = jnp.exp(-0.5 * Z**2) / jnp.sqrt(2 * jnp.pi)
        Phi = 0.5 * (1 + jnp.tanh(Z * 0.7978845608))  # approximation to erf

        ei = improvement * Phi + std * phi
        return jnp.maximum(ei, 0)

    def lower_confidence_bound(mean, std, beta=2.0):
        """Lower Confidence Bound for minimization."""
        return mean - beta * std

    return thompson_sample, expected_improvement, lower_confidence_bound


@app.cell(hide_code=True)
def _(mo, jnp, jr):
    # State for optimization
    get_X_obs, set_X_obs = mo.state(jnp.array([]))
    get_y_obs, set_y_obs = mo.state(jnp.array([]))
    get_iteration, set_iteration = mo.state(0)
    get_best_x, set_best_x = mo.state(None)
    get_best_y, set_best_y = mo.state(None)
    get_rng_key, set_rng_key = mo.state(jr.PRNGKey(42))
    get_history, set_history = mo.state([])
    get_next_x, set_next_x = mo.state(None)
    get_thompson_samples, set_thompson_samples = mo.state([])

    return (
        get_X_obs, set_X_obs,
        get_y_obs, set_y_obs,
        get_iteration, set_iteration,
        get_best_x, set_best_x,
        get_best_y, set_best_y,
        get_rng_key, set_rng_key,
        get_history, set_history,
        get_next_x, set_next_x,
        get_thompson_samples, set_thompson_samples,
    )


@app.cell(hide_code=True)
def _(
    mo, jnp, jr, np,
    objective_dropdown, domain_slider, kernel_dropdown,
    variance_slider, lengthscale_slider, noise_slider,
    acquisition_dropdown, beta_slider, n_thompson_slider,
    n_initial_slider, seed_slider,
    get_objective_func, build_gp, gp_posterior,
    thompson_sample, expected_improvement, lower_confidence_bound,
    get_X_obs, set_X_obs, get_y_obs, set_y_obs,
    get_iteration, set_iteration,
    get_best_x, set_best_x, get_best_y, set_best_y,
    get_rng_key, set_rng_key,
    get_history, set_history,
    get_next_x, set_next_x,
    get_thompson_samples, set_thompson_samples,
    get_custom_code,
):
    # Get parameters
    func_type = objective_dropdown.value
    domain_min, domain_max = domain_slider.value
    kernel_type = kernel_dropdown.value
    variance = variance_slider.value
    lengthscale = lengthscale_slider.value
    noise = noise_slider.value
    acq_type = acquisition_dropdown.value
    beta = beta_slider.value
    n_thompson = n_thompson_slider.value
    n_initial = n_initial_slider.value
    seed = seed_slider.value
    custom_code = get_custom_code() if func_type == 'custom' else None

    # Get objective function
    objective_func, true_min_x = get_objective_func(func_type, custom_code)

    # Grid for evaluation
    X_grid = jnp.linspace(domain_min, domain_max, 200)

    def reset_optimization(_):
        """Reset to initial random points."""
        key = jr.PRNGKey(seed)
        key, subkey = jr.split(key)

        # Generate initial random points
        X_init = jr.uniform(subkey, (n_initial,), minval=domain_min, maxval=domain_max)
        y_init = jnp.array([objective_func(x) for x in X_init])

        # Find best initial point
        best_idx = jnp.argmin(y_init)
        best_x = X_init[best_idx]
        best_y = y_init[best_idx]

        set_X_obs(X_init)
        set_y_obs(y_init)
        set_iteration(n_initial)
        set_best_x(float(best_x))
        set_best_y(float(best_y))
        set_rng_key(key)
        set_history([{'iter': i+1, 'best_y': float(jnp.min(y_init[:i+1])),
                     'regret': float(jnp.min(y_init[:i+1]) - objective_func(jnp.array(true_min_x)))}
                    for i in range(n_initial)])
        set_next_x(None)
        set_thompson_samples([])

    def step_optimization(_):
        """Perform one BO iteration."""
        X = get_X_obs()
        y = get_y_obs()
        key = get_rng_key()

        if len(X) == 0:
            reset_optimization(None)
            return

        # Build GP and get posterior
        gp = build_gp(X, y, kernel_type, variance, lengthscale, noise)
        mean, std, cond_gp = gp_posterior(gp, y, X_grid, noise)

        # Select next point based on acquisition function
        key, subkey = jr.split(key)
        thompson_samples_list = []

        if acq_type == 'thompson':
            # Thompson sampling: draw multiple samples, pick minimum of best sample
            best_sample_min = jnp.inf
            x_next = X_grid[0]
            for i in range(n_thompson):
                key, sample_key = jr.split(key)
                x_candidate, sample = thompson_sample(cond_gp, X_grid, sample_key)
                thompson_samples_list.append(np.array(sample))
                if jnp.min(sample) < best_sample_min:
                    best_sample_min = jnp.min(sample)
                    x_next = x_candidate
        elif acq_type == 'ei':
            # Expected Improvement
            ei = expected_improvement(mean, std, jnp.min(y))
            x_next = X_grid[jnp.argmax(ei)]
        else:  # lcb
            # Lower Confidence Bound
            lcb = lower_confidence_bound(mean, std, beta)
            x_next = X_grid[jnp.argmin(lcb)]

        # Evaluate objective at next point
        y_next = objective_func(x_next)

        # Update observations
        X_new = jnp.append(X, x_next)
        y_new = jnp.append(y, y_next)
        set_X_obs(X_new)
        set_y_obs(y_new)
        set_rng_key(key)
        set_thompson_samples(thompson_samples_list)

        # Update best
        if y_next < get_best_y():
            set_best_x(float(x_next))
            set_best_y(float(y_next))

        # Update iteration and history
        new_iter = get_iteration() + 1
        set_iteration(new_iter)
        history = get_history()
        true_min_val = float(objective_func(jnp.array(true_min_x)))
        history.append({
            'iter': new_iter,
            'best_y': float(jnp.min(y_new)),
            'regret': float(jnp.min(y_new) - true_min_val)
        })
        set_history(history)
        set_next_x(float(x_next))

    # Action buttons
    step_button = mo.ui.button(label='Step', on_click=step_optimization)
    reset_button = mo.ui.button(label='Reset', on_click=reset_optimization)

    # Run N iterations
    def run_n_iterations(n):
        def handler(_):
            for _ in range(n):
                step_optimization(None)
        return handler

    run_5_button = mo.ui.button(label='Run 5', on_click=run_n_iterations(5))
    run_10_button = mo.ui.button(label='Run 10', on_click=run_n_iterations(10))

    return (
        step_button, reset_button, run_5_button, run_10_button,
        objective_func, true_min_x, X_grid,
        func_type, domain_min, domain_max, kernel_type, variance, lengthscale, noise,
        acq_type, beta, n_thompson, n_initial, seed,
    )


@app.cell(hide_code=True)
def _(
    alt, pd, mo, jnp, np,
    objective_func, true_min_x, X_grid,
    domain_min, domain_max, kernel_type, variance, lengthscale, noise,
    acq_type, beta,
    build_gp, gp_posterior, expected_improvement, lower_confidence_bound,
    get_X_obs, get_y_obs, get_best_x, get_best_y, get_next_x, get_thompson_samples,
):
    # Get current state
    X_obs = get_X_obs()
    y_obs = get_y_obs()
    best_x = get_best_x()
    best_y = get_best_y()
    next_x = get_next_x()
    thompson_samples = get_thompson_samples()

    # Evaluate ground truth
    y_true = np.array([float(objective_func(x)) for x in X_grid])
    X_grid_np = np.array(X_grid)

    # Find true minimum in domain
    true_min_idx = np.argmin(y_true)
    true_min_x_domain = X_grid_np[true_min_idx]
    true_min_y = y_true[true_min_idx]

    # Prepare data
    gt_df = pd.DataFrame({'x': X_grid_np, 'y': y_true})

    # Compute GP posterior if we have observations
    if len(X_obs) > 0:
        gp = build_gp(X_obs, y_obs, kernel_type, variance, lengthscale, noise)
        mean, std, cond_gp = gp_posterior(gp, y_obs, X_grid, noise)
        mean_np = np.array(mean)
        std_np = np.array(std)

        # GP data
        gp_df = pd.DataFrame({
            'x': X_grid_np,
            'mean': mean_np,
            'lower': mean_np - 2 * std_np,
            'upper': mean_np + 2 * std_np,
        })

        # Observations data
        obs_df = pd.DataFrame({
            'x': np.array(X_obs),
            'y': np.array(y_obs),
        })

        # Acquisition function data
        if acq_type == 'ei':
            acq_values = np.array(expected_improvement(mean, std, jnp.min(y_obs)))
            acq_label = 'Expected Improvement'
            acq_df = pd.DataFrame({'x': X_grid_np, 'value': acq_values})
        elif acq_type == 'lcb':
            acq_values = np.array(lower_confidence_bound(mean, std, beta))
            acq_label = 'Lower Confidence Bound'
            acq_df = pd.DataFrame({'x': X_grid_np, 'value': acq_values})
        else:
            acq_values = None
            acq_label = None
            acq_df = pd.DataFrame(columns=['x', 'value'])
    else:
        gp_df = pd.DataFrame(columns=['x', 'mean', 'lower', 'upper'])
        obs_df = pd.DataFrame(columns=['x', 'y'])
        acq_values = None
        acq_df = pd.DataFrame(columns=['x', 'value'])
        acq_label = None

    # Define scales
    x_scale = alt.Scale(domain=[domain_min, domain_max])
    y_min_plot = min(y_true.min(), -1) - 0.5
    y_max_plot = max(y_true.max(), 5) + 0.5
    y_scale = alt.Scale(domain=[y_min_plot, y_max_plot])

    # Build main chart layers
    layers = []

    # Ground truth (black dashed)
    gt_line = alt.Chart(gt_df).mark_line(
        color='black', strokeWidth=2, strokeDash=[5, 5], opacity=0.7
    ).encode(
        x=alt.X('x:Q', scale=x_scale, title='x'),
        y=alt.Y('y:Q', scale=y_scale, title='f(x)'),
    )
    layers.append(gt_line)

    # GP uncertainty band (blue shaded)
    if len(gp_df) > 0:
        band = alt.Chart(gp_df).mark_area(
            opacity=0.3, color='#1f77b4'
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('lower:Q', scale=y_scale),
            y2='upper:Q',
        )
        layers.append(band)

        # GP mean (blue solid)
        mean_line = alt.Chart(gp_df).mark_line(
            color='#1f77b4', strokeWidth=2
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('mean:Q', scale=y_scale),
        )
        layers.append(mean_line)

    # Thompson samples (colored lines)
    if thompson_samples and len(thompson_samples) > 0:
        sample_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8']
        for i, sample in enumerate(thompson_samples):
            sample_df = pd.DataFrame({
                'x': X_grid_np,
                'y': sample,
                'sample': f'Sample {i+1}'
            })
            sample_line = alt.Chart(sample_df).mark_line(
                color=sample_colors[i % len(sample_colors)],
                strokeWidth=1.5,
                opacity=0.5
            ).encode(
                x=alt.X('x:Q', scale=x_scale),
                y=alt.Y('y:Q', scale=y_scale),
            )
            layers.append(sample_line)

    # Observations (blue circles)
    if len(obs_df) > 0:
        obs_points = alt.Chart(obs_df).mark_circle(
            color='#1f77b4', size=100, opacity=0.8
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale),
        )
        layers.append(obs_points)

    # Current best (green diamond)
    if best_x is not None and best_y is not None:
        best_df = pd.DataFrame({'x': [best_x], 'y': [best_y]})
        best_point = alt.Chart(best_df).mark_point(
            color='#2ca02c', size=200, shape='diamond', filled=True
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale),
        )
        layers.append(best_point)

    # Next query point (red star)
    if next_x is not None:
        next_y = float(objective_func(jnp.array(next_x)))
        next_df = pd.DataFrame({'x': [next_x], 'y': [next_y]})
        next_point = alt.Chart(next_df).mark_point(
            color='#d62728', size=250, shape='cross', filled=True, strokeWidth=3
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale),
        )
        layers.append(next_point)

    # True optimum (vertical dashed line)
    true_opt_df = pd.DataFrame({'x': [true_min_x_domain]})
    true_opt_line = alt.Chart(true_opt_df).mark_rule(
        color='green', strokeWidth=2, strokeDash=[3, 3], opacity=0.7
    ).encode(x=alt.X('x:Q', scale=x_scale))
    layers.append(true_opt_line)

    # Manual legend
    legend_data = pd.DataFrame({
        'label': ['Ground Truth', 'GP Mean', 'GP ±2σ', 'Observations', 'Best Found', 'Next Query', 'True Optimum'],
        'type': ['line', 'line', 'area', 'point', 'point', 'point', 'line'],
        'color': ['black', '#1f77b4', '#1f77b4', '#1f77b4', '#2ca02c', '#d62728', 'green'],
        'dash': ['dashed', 'solid', 'solid', 'solid', 'solid', 'solid', 'dashed'],
        'col': [0, 1, 0, 1, 0, 1, 0],
        'row': [0, 0, 1, 1, 2, 2, 3],
    })

    # Legend positioning
    legend_right = domain_max - 0.1
    legend_top = y_max_plot - 0.5
    row_spacing = (y_max_plot - y_min_plot) * 0.08
    col_width = (domain_max - domain_min) * 0.15
    col1_symbol_x = legend_right - 2 * col_width + 0.1
    col2_symbol_x = legend_right - col_width + 0.1

    legend_items_df = legend_data.copy()
    legend_items_df['x'] = legend_items_df['col'].apply(lambda c: col1_symbol_x if c == 0 else col2_symbol_x)
    legend_items_df['y'] = legend_items_df['row'].apply(lambda r: legend_top - r * row_spacing)
    legend_items_df['x_text'] = legend_items_df['x'] + (domain_max - domain_min) * 0.03

    n_rows = 4
    # Legend background
    legend_bg = alt.Chart(pd.DataFrame({
        'x': [col1_symbol_x - 0.15], 'x2': [legend_right + 0.1],
        'y': [legend_top + row_spacing * 0.5], 'y2': [legend_top - (n_rows - 1) * row_spacing - row_spacing * 0.5]
    })).mark_rect(fill='white', stroke='#ccc', strokeWidth=1, cornerRadius=4, opacity=0.95).encode(
        x=alt.X('x:Q', scale=x_scale), x2='x2:Q',
        y=alt.Y('y:Q', scale=y_scale), y2='y2:Q'
    )
    layers.append(legend_bg)

    # Legend symbols and text
    for _, row in legend_items_df.iterrows():
        if row['type'] == 'line':
            dash_pattern = [5, 5] if row['dash'] == 'dashed' else []
            line_seg = alt.Chart(pd.DataFrame({
                'x': [row['x'] - 0.08, row['x'] + 0.08],
                'y': [row['y'], row['y']]
            })).mark_line(
                color=row['color'], strokeWidth=2.5, strokeDash=dash_pattern,
                opacity=0.8 if row['dash'] == 'dashed' else 1.0
            ).encode(x=alt.X('x:Q', scale=x_scale), y=alt.Y('y:Q', scale=y_scale))
            layers.append(line_seg)
        elif row['type'] == 'area':
            area_sym = alt.Chart(pd.DataFrame({
                'x': [row['x'] - 0.06], 'x2': [row['x'] + 0.06],
                'y': [row['y'] - row_spacing * 0.3], 'y2': [row['y'] + row_spacing * 0.3]
            })).mark_rect(fill=row['color'], opacity=0.3, stroke=row['color'], strokeWidth=1).encode(
                x=alt.X('x:Q', scale=x_scale), x2='x2:Q',
                y=alt.Y('y:Q', scale=y_scale), y2='y2:Q'
            )
            layers.append(area_sym)
        elif row['type'] == 'point':
            shape = 'circle'
            if row['label'] == 'Best Found':
                shape = 'diamond'
            elif row['label'] == 'Next Query':
                shape = 'cross'
            point_sym = alt.Chart(pd.DataFrame({'x': [row['x']], 'y': [row['y']]})).mark_point(
                color=row['color'], size=80, shape=shape, filled=True
            ).encode(x=alt.X('x:Q', scale=x_scale), y=alt.Y('y:Q', scale=y_scale))
            layers.append(point_sym)

    # Legend text
    legend_text = alt.Chart(legend_items_df).mark_text(
        align='left', fontSize=10, font='sans-serif'
    ).encode(
        x=alt.X('x_text:Q', scale=x_scale),
        y=alt.Y('y:Q', scale=y_scale),
        text='label:N'
    )
    layers.append(legend_text)

    # Combine main chart
    main_chart = alt.layer(*layers).properties(
        width='container', height=300,
        title='GP Posterior and Observations'
    )

    # Acquisition function chart (if applicable)
    if acq_values is not None and acq_label is not None:
        acq_chart = alt.Chart(acq_df).mark_line(
            color='#ff7f0e', strokeWidth=2
        ).encode(
            x=alt.X('x:Q', scale=x_scale, title='x'),
            y=alt.Y('value:Q', title=acq_label),
        ).properties(
            width='container', height=120,
            title=f'Acquisition Function: {acq_label}'
        )

        # Mark maximum/minimum of acquisition
        if acq_type == 'ei':
            acq_opt_idx = np.argmax(acq_values)
        else:
            acq_opt_idx = np.argmin(acq_values)
        acq_opt_df = pd.DataFrame({'x': [X_grid_np[acq_opt_idx]], 'value': [acq_values[acq_opt_idx]]})
        acq_opt_point = alt.Chart(acq_opt_df).mark_point(
            color='#d62728', size=150, shape='cross', filled=True
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('value:Q'),
        )
        acq_chart = alt.layer(acq_chart, acq_opt_point)

        combined_chart = alt.vconcat(main_chart, acq_chart).configure_axis(
            grid=True, gridOpacity=0.3,
            labelFontSize=12, titleFontSize=14
        ).configure_title(fontSize=16)
    else:
        combined_chart = main_chart.configure_axis(
            grid=True, gridOpacity=0.3,
            labelFontSize=12, titleFontSize=14
        ).configure_title(fontSize=16)

    chart_display = mo.ui.altair_chart(combined_chart)

    return (chart_display, true_min_x_domain, true_min_y)


@app.cell(hide_code=True)
def _(chart_display):
    # Display chart
    chart_output = chart_display
    return (chart_output,)


@app.cell(hide_code=True)
def _(mo, np, get_best_x, get_best_y, get_iteration, true_min_x_domain, true_min_y):
    # Metrics table
    _best_x = get_best_x()
    _best_y = get_best_y()
    _iteration = get_iteration()

    if _best_x is not None and _best_y is not None:
        _simple_regret = _best_y - true_min_y
        _distance_to_opt = abs(_best_x - true_min_x_domain)
    else:
        _simple_regret = np.nan
        _distance_to_opt = np.nan

    def fmt(v, precision=4):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return '—'
        return f'{v:.{precision}f}'

    metrics_table = mo.Html(f'''
    <table style="width: 100%; border-collapse: collapse; font-size: 13px; max-width: 500px; margin: 0 auto;">
        <thead>
            <tr style="border-bottom: 2px solid #dee2e6;">
                <th style="text-align: left; padding: 6px 8px;">Metric</th>
                <th style="text-align: right; padding: 6px 8px;">Value</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 6px 8px;">Iteration</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{_iteration}</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 6px 8px;">Best x found</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(_best_x)}</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 6px 8px;">Best f(x) found</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(_best_y)}</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 6px 8px;">True optimum f(x*)</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(true_min_y)}</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 6px 8px;">Simple Regret (f - f*)</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(_simple_regret)}</td>
            </tr>
            <tr>
                <td style="padding: 6px 8px;">Distance to optimum |x - x*|</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(_distance_to_opt)}</td>
            </tr>
        </tbody>
    </table>
    ''')
    return (metrics_table,)


@app.cell(hide_code=True)
def _(
    mo,
    objective_dropdown, domain_slider, kernel_dropdown,
    variance_slider, lengthscale_slider, noise_slider,
    acquisition_dropdown, beta_slider, n_thompson_slider,
    n_initial_slider, seed_slider,
    step_button, reset_button, run_5_button, run_10_button,
    custom_function_accordion, acq_type,
):
    # Conditional display of acquisition parameters
    if acq_type == 'lcb':
        acq_params = mo.vstack([beta_slider], gap="0.3em")
    elif acq_type == 'thompson':
        acq_params = mo.vstack([n_thompson_slider], gap="0.3em")
    else:
        acq_params = mo.Html("")

    sidebar = mo.Html(f'''
    <div class="app-sidebar">
        <h4>Objective Function</h4>
        {objective_dropdown}
        {custom_function_accordion}
        {domain_slider}

        <h4>GP Model</h4>
        {kernel_dropdown}
        {variance_slider}
        {lengthscale_slider}
        {noise_slider}

        <h4>Acquisition Function</h4>
        {acquisition_dropdown}
        {mo.as_html(acq_params)}

        <h4>Optimization</h4>
        {n_initial_slider}
        {seed_slider}

        <div style="display: flex; gap: 0.5em; margin-top: 1em; flex-wrap: wrap;">
            {step_button}
            {run_5_button}
            {run_10_button}
            {reset_button}
        </div>

        <p style="font-size: 0.85em; color: #666; margin-top: 1em;">
            <b>Tip:</b> Click Reset to initialize, then Step or Run to iterate.
            Thompson sampling shows sampled functions; EI/LCB show acquisition plots.
        </p>
    </div>
    ''')
    return (sidebar,)


@app.cell(hide_code=True)
def _(mo, header, chart_output, sidebar, metrics_table):
    mo.vstack([
        header,
        mo.Html(f'''
        <div class="app-layout">
            <div class="app-plot">
                {mo.as_html(chart_output)}
                <div style="margin-top: 1em;">
                    {mo.as_html(metrics_table)}
                </div>
            </div>
            {sidebar}
        </div>
        ''')
    ])
    return


if __name__ == "__main__":
    app.run()
