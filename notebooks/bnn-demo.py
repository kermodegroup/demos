# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.2.5",
#     "qrcode[pil]==8.2",
#     "jax",
#     "jaxlib",
#     "equinox",
#     "numpyro",
#     "altair",
#     "pandas",
#     "pyarrow",
# ]
# ///

import marimo

__generated_with = "0.18.1"
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
    import equinox as eqx

    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS, Predictive
    from numpyro.infer.autoguide import AutoNormal

    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'cpu')
    numpyro.set_host_device_count(1)

    return alt, dist, eqx, jax, jnp, mo, np, pd, numpyro, SVI, Trace_ELBO, MCMC, NUTS, Predictive, AutoNormal


@app.cell(hide_code=True)
def _():
    import qrcode
    import io
    import base64

    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://sciml.warwick.ac.uk/live/bnn-demo/')
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
                <p style='font-size: 24px; margin: 0; padding: 0; line-height: 1.3;'><b>BNN Regression Demo</b>
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
        "# Define y = f(X) where X is a numpy array\n"
        "# Available: np, math, X\n"
        "y = np.sin(np.pi * X)"
    )

    # Data controls
    n_points_slider = mo.ui.slider(10, 100, 5, 30, label='Training points $N$')
    noise_slider = mo.ui.slider(0.0, 0.5, 0.05, 0.1, label='Noise level $\\sigma$')
    function_dropdown = mo.ui.dropdown(
        options={'Sine': 'sin', 'Step': 'step', 'Runge': 'runge', 'Witch of Agnesi': 'witch', 'Custom': 'custom'},
        value='Sine',
        label='Target function'
    )
    seed_slider = mo.ui.slider(0, 10, 1, 0, label='Random seed')
    train_range_slider = mo.ui.range_slider(
        start=-2, stop=2, step=0.1,
        value=[-1.0, 1.0],
        label='Training range'
    )

    # Network architecture controls
    width_slider = mo.ui.slider(4, 128, 4, 8, label='Width (units per layer)')
    depth_slider = mo.ui.slider(1, 5, 1, 2, label='Depth (hidden layers)')
    activation_dropdown = mo.ui.dropdown(
        options={'ReLU': 'relu', 'Tanh': 'tanh', 'Sigmoid': 'sigmoid'},
        value='Tanh',
        label='Activation function'
    )

    # Noise model controls
    noise_type_dropdown = mo.ui.dropdown(
        options={'Homoscedastic': 'homo', 'Heteroscedastic': 'hetero'},
        value='Homoscedastic',
        label='Noise type'
    )
    hetero_scale_slider = mo.ui.slider(0.0, 2.0, 0.1, 0.5, label='Heteroscedastic scale')

    # Inference method controls
    inference_dropdown = mo.ui.dropdown(
        options={'Variational Inference (VI)': 'vi', 'MCMC (NUTS)': 'mcmc'},
        value='Variational Inference (VI)',
        label='Inference method'
    )

    # VI-specific controls
    svi_steps_slider = mo.ui.slider(500, 10000, 500, 5000, label='SVI steps')
    vi_lr_slider = mo.ui.slider(-4, -1, 0.5, -3, label='Learning rate (10^x)')

    # MCMC-specific controls
    mcmc_samples_slider = mo.ui.slider(100, 2000, 100, 1000, label='MCMC samples')
    mcmc_warmup_slider = mo.ui.slider(100, 2000, 100, 1000, label='Warmup steps')

    # Common inference controls
    prior_scale_slider = mo.ui.slider(0.1, 10.0, 0.1, 1.0, label='Prior scale $\\sigma_w$')
    pred_samples_slider = mo.ui.slider(50, 500, 50, 200, label='Posterior samples')
    show_samples_checkbox = mo.ui.checkbox(value=False, label='Show sample curves')

    # Action buttons
    train_button = mo.ui.run_button(label='Train')
    reset_button = mo.ui.run_button(label='Reset')
    store_button = mo.ui.run_button(label='Store Fit')
    clear_stored_button = mo.ui.run_button(label='Clear Stored')

    # Custom function code editor and accordion
    custom_code_editor = mo.ui.code_editor(
        value=get_custom_code(),
        language="python",
        min_height=120,
        on_change=set_custom_code,
    )

    custom_function_accordion = mo.accordion({
        "Custom Function Code": mo.vstack([
            mo.md("Define `y = f(X)`. Available: `np`, `math`, `X` (numpy array)."),
            custom_code_editor,
        ])
    }, lazy=True)

    return (
        n_points_slider, noise_slider, function_dropdown, seed_slider, train_range_slider,
        width_slider, depth_slider, activation_dropdown,
        noise_type_dropdown, hetero_scale_slider,
        inference_dropdown, svi_steps_slider, vi_lr_slider,
        mcmc_samples_slider, mcmc_warmup_slider,
        prior_scale_slider, pred_samples_slider, show_samples_checkbox,
        train_button, reset_button, store_button, clear_stored_button,
        get_custom_code, set_custom_code, custom_code_editor, custom_function_accordion,
    )


@app.cell(hide_code=True)
def _(np):
    def _eval_function(X, function_type, custom_code=None):
        """Evaluate target function."""
        if function_type == 'sin':
            return np.sin(np.pi * X)
        elif function_type == 'step':
            return np.where(X < 0, -0.5, 0.5)
        elif function_type == 'runge':
            return 1.0 / (1.0 + 25.0 * X**2)
        elif function_type == 'witch':
            return 1.0 / (1.0 + X**2)
        elif function_type == 'custom' and custom_code:
            import math
            safe_builtins = {
                'range': range, 'len': len, 'sum': sum, 'min': min, 'max': max,
                'abs': abs, 'round': round, 'int': int, 'float': float,
                'True': True, 'False': False, 'None': None,
            }
            try:
                namespace = {'np': np, 'math': math, 'X': X}
                exec(custom_code, {"__builtins__": safe_builtins}, namespace)
                if 'y' not in namespace:
                    raise ValueError("Code must define 'y'")
                y = np.atleast_1d(np.asarray(namespace['y'], dtype=float))
                if y.shape != X.shape:
                    y = np.broadcast_to(y, X.shape).copy()
                return y
            except Exception:
                return np.full_like(X, np.nan, dtype=float)
        return np.sin(np.pi * X)

    def generate_data(n_points, noise_std, function_type, seed, custom_code=None,
                      heteroscedastic=False, hetero_scale=0.5):
        """Generate training data with optional heteroscedastic noise."""
        rng = np.random.default_rng(seed)
        X = rng.uniform(-2, 2, n_points)
        X = np.sort(X)

        y_true = _eval_function(X, function_type, custom_code)
        if heteroscedastic:
            sigma = noise_std * (1.0 + hetero_scale * np.abs(X))
            y = y_true + rng.normal(0, 1, n_points) * sigma
        else:
            sigma = np.full(n_points, noise_std)
            y = y_true + rng.normal(0, noise_std, n_points)
        return X, y, y_true, sigma

    def get_ground_truth(X, function_type, custom_code=None):
        """Get ground truth values for plotting."""
        return _eval_function(X, function_type, custom_code)

    return generate_data, get_ground_truth


@app.cell(hide_code=True)
def _(jax, jnp, dist, numpyro):
    def get_activation(name):
        """Get JAX activation function by name."""
        if name == 'relu':
            return jax.nn.relu
        elif name == 'tanh':
            return jnp.tanh
        elif name == 'sigmoid':
            return jax.nn.sigmoid
        else:
            return jax.nn.relu

    def bnn_model(X, Y=None, width=32, depth=2, activation_fn=jnp.tanh,
                  prior_scale=1.0, heteroscedastic=False):
        """BNN model function for use with numpyro inference."""
        h = X.reshape(-1, 1)
        in_dim = 1

        # Hidden layers
        for i in range(depth):
            w = numpyro.sample(f"w{i}", dist.Normal(
                jnp.zeros((in_dim, width)),
                prior_scale * jnp.ones((in_dim, width))
            ))
            b = numpyro.sample(f"b{i}", dist.Normal(
                jnp.zeros(width),
                prior_scale * jnp.ones(width)
            ))
            h = activation_fn(h @ w + b)
            in_dim = width

        # Mean output head
        w_out = numpyro.sample("w_out", dist.Normal(
            jnp.zeros((width, 1)),
            prior_scale * jnp.ones((width, 1))
        ))
        b_out = numpyro.sample("b_out", dist.Normal(
            jnp.zeros(1),
            prior_scale * jnp.ones(1)
        ))
        mean = (h @ w_out + b_out).squeeze(-1)

        if heteroscedastic:
            # Second output head for input-dependent log-variance
            w_var = numpyro.sample("w_var", dist.Normal(
                jnp.zeros((width, 1)),
                prior_scale * jnp.ones((width, 1))
            ))
            b_var = numpyro.sample("b_var", dist.Normal(
                jnp.zeros(1),
                prior_scale * jnp.ones(1)
            ))
            log_var = (h @ w_var + b_var).squeeze(-1)
            sigma = jnp.sqrt(jax.nn.softplus(log_var) + 1e-6)
        else:
            prec = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
            sigma = 1.0 / jnp.sqrt(prec)

        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("obs", dist.Normal(mean, sigma), obs=Y)

        return mean

    return get_activation, bnn_model


@app.cell(hide_code=True)
def _(mo):
    # State for inference results
    get_result, set_result = mo.state(None)
    get_losses, set_losses = mo.state([])
    get_trained, set_trained = mo.state(False)
    get_train_params, set_train_params = mo.state(None)
    get_pred_stats, set_pred_stats = mo.state(None)
    get_sample_traces, set_sample_traces = mo.state(None)
    get_stored_fits, set_stored_fits = mo.state([])
    get_diag_info, set_diag_info = mo.state(None)
    return (
        get_result, set_result, get_losses, set_losses,
        get_trained, set_trained, get_train_params, set_train_params,
        get_pred_stats, set_pred_stats, get_sample_traces, set_sample_traces,
        get_stored_fits, set_stored_fits, get_diag_info, set_diag_info,
    )


@app.cell(hide_code=True)
def _(
    mo, jax, jnp, np, numpyro,
    SVI, Trace_ELBO, MCMC, NUTS, Predictive, AutoNormal,
    n_points_slider, noise_slider, function_dropdown, seed_slider, train_range_slider,
    width_slider, depth_slider, activation_dropdown,
    noise_type_dropdown, hetero_scale_slider,
    inference_dropdown, svi_steps_slider, vi_lr_slider,
    mcmc_samples_slider, mcmc_warmup_slider,
    prior_scale_slider, pred_samples_slider, show_samples_checkbox,
    train_button, reset_button, store_button, clear_stored_button,
    generate_data, get_activation, bnn_model,
    get_result, set_result, get_losses, set_losses,
    get_trained, set_trained, get_train_params, set_train_params,
    get_pred_stats, set_pred_stats, get_sample_traces, set_sample_traces,
    get_stored_fits, set_stored_fits, get_diag_info, set_diag_info,
    get_custom_code,
):
    # Get parameters
    custom_code = get_custom_code() if function_dropdown.value == 'custom' else None
    _heteroscedastic = noise_type_dropdown.value == 'hetero'

    # Generate training data
    X_full, y_full, _, sigma_full = generate_data(
        n_points_slider.value,
        noise_slider.value,
        function_dropdown.value,
        seed_slider.value,
        custom_code,
        heteroscedastic=_heteroscedastic,
        hetero_scale=hetero_scale_slider.value,
    )

    # Filter to training range
    train_min, train_max = train_range_slider.value
    mask = (X_full >= train_min) & (X_full <= train_max)
    X_train = X_full[mask]
    y_train = y_full[mask]
    sigma_train = sigma_full[mask]

    X_jax = jnp.array(X_train)
    y_jax = jnp.array(y_train)

    # Network params
    width = width_slider.value
    depth = depth_slider.value
    activation_name = activation_dropdown.value
    activation = get_activation(activation_name)
    prior_scale = prior_scale_slider.value
    inference_method = inference_dropdown.value

    # Current parameters for staleness tracking
    current_params = {
        'n_points': n_points_slider.value,
        'noise': noise_slider.value,
        'function': function_dropdown.value,
        'custom_code': custom_code,
        'seed': seed_slider.value,
        'width': width,
        'depth': depth,
        'activation': activation_name,
        'prior_scale': prior_scale,
        'noise_type': noise_type_dropdown.value,
        'hetero_scale': hetero_scale_slider.value,
        'inference': inference_method,
        'train_min': train_min,
        'train_max': train_max,
    }

    # Handle reset button
    if reset_button.value:
        set_result(None)
        set_losses([])
        set_trained(False)
        set_train_params(None)
        set_pred_stats(None)
        set_sample_traces(None)
        set_diag_info(None)

    # Handle train button
    if train_button.value:
        _key = jax.random.PRNGKey(seed_slider.value + 42)

        # Build model closure with current architecture params
        def _model_fn(X, Y=None):
            return bnn_model(X, Y, width=width, depth=depth,
                           activation_fn=activation, prior_scale=prior_scale,
                           heteroscedastic=_heteroscedastic)

        # Helper to show centered popup progress
        def _show_popup(title, subtitle, pct=None):
            if pct is not None:
                _bar_html = (
                    '<div style="background:#eee;border-radius:6px;height:10px;'
                    'overflow:hidden;">'
                    f'<div style="background:linear-gradient(90deg,#e34a33,#ff6b4a);'
                    f'height:100%;border-radius:6px;width:{pct:.0f}%;"></div>'
                    '</div>'
                )
            else:
                _bar_html = (
                    '<div style="background:#eee;border-radius:6px;height:10px;'
                    'overflow:hidden;">'
                    '<div style="background:linear-gradient(90deg,#e34a33,#ff6b4a);'
                    'height:100%;border-radius:6px;width:100%;'
                    'animation:pulse 1.5s ease-in-out infinite alternate;"></div>'
                    '</div>'
                    '<style>@keyframes pulse{from{opacity:.4}to{opacity:1}}</style>'
                )
            mo.output.replace(mo.Html(
                '<div style="position:fixed;top:50%;left:50%;'
                'transform:translate(-50%,-50%);background:white;'
                'padding:24px 32px;border-radius:12px;'
                'box-shadow:0 8px 32px rgba(0,0,0,0.15);z-index:10000;'
                'font-family:system-ui;text-align:center;min-width:300px;">'
                f'<div style="font-size:15px;color:#333;margin-bottom:12px;">'
                f'{title}</div>'
                f'{_bar_html}'
                f'<div style="font-size:13px;color:#666;margin-top:8px;">'
                f'{subtitle}</div></div>'
            ))

        if inference_method == 'vi':
            # --- Variational Inference ---
            _guide = AutoNormal(_model_fn)
            _optimizer = numpyro.optim.Adam(10 ** vi_lr_slider.value)
            _svi = SVI(_model_fn, _guide, _optimizer, loss=Trace_ELBO())

            _n_steps = svi_steps_slider.value

            # Run SVI in JIT-compiled chunks for speed + progress updates
            # Slider step is 500 so _n_steps is always a multiple of _chunk_size
            _chunk_size = min(500, _n_steps)
            _n_chunks = _n_steps // _chunk_size

            # JIT-compiled function to run a chunk of SVI steps via lax.scan
            @jax.jit
            def _run_chunk(_state):
                def _body(carry, _unused):
                    s, _ = carry
                    s, loss = _svi.stable_update(s, X_jax, y_jax)
                    return (s, loss), loss
                (new_state, _), losses = jax.lax.scan(
                    _body, (_state, 0.0), None, length=_chunk_size
                )
                return new_state, losses

            _show_popup('Variational Inference', 'Compiling...')
            _svi_state = _svi.init(_key, X_jax, y_jax)
            _all_losses = []
            _steps_done = 0

            for _chunk_i in range(_n_chunks):
                _svi_state, _chunk_losses = _run_chunk(_svi_state)
                _all_losses.extend([float(v) for v in _chunk_losses])
                _steps_done += _chunk_size
                _pct = 100 * _steps_done / _n_steps
                _current_loss = float(_chunk_losses[-1])
                _show_popup(
                    f'Variational Inference &mdash; step {_steps_done}/{_n_steps}',
                    f'-ELBO: {_current_loss:.1f}',
                    pct=_pct,
                )

            _svi_params = _svi.get_params(_svi_state)
            set_losses(_all_losses)
            set_result({'type': 'vi', 'params': _svi_params, 'guide': _guide})

            # Posterior predictive samples
            _show_popup('Variational Inference', 'Sampling posterior predictive...')
            _n_pred = pred_samples_slider.value
            _pred_key = jax.random.PRNGKey(seed_slider.value + 100)
            _X_plot = jnp.linspace(-2, 2, 200)
            _predictive = Predictive(_model_fn, guide=_guide,
                                     params=_svi_params, num_samples=_n_pred)
            _pred_samples = _predictive(_pred_key, _X_plot)
            _obs_samples = _pred_samples['obs']  # shape: (n_pred, 200)

            # Compute summary stats
            _mean = np.array(jnp.mean(_obs_samples, axis=0))
            _std = np.array(jnp.std(_obs_samples, axis=0))
            _q05 = np.array(jnp.percentile(_obs_samples, 5, axis=0))
            _q25 = np.array(jnp.percentile(_obs_samples, 25, axis=0))
            _q75 = np.array(jnp.percentile(_obs_samples, 75, axis=0))
            _q95 = np.array(jnp.percentile(_obs_samples, 95, axis=0))

            set_pred_stats({
                'mean': _mean, 'std': _std,
                'q05': _q05, 'q25': _q25, 'q75': _q75, 'q95': _q95,
                'X_plot': np.array(_X_plot),
            })

            # Sample traces for optional display
            _n_show = min(20, _n_pred)
            set_sample_traces(np.array(_obs_samples[:_n_show]))

            set_diag_info({
                'type': 'vi',
                'final_elbo': float(_all_losses[-1]),
                'n_steps': _n_steps,
            })

        else:
            # --- MCMC (NUTS) ---
            _kernel = NUTS(_model_fn)
            _n_samples = mcmc_samples_slider.value
            _n_warmup = mcmc_warmup_slider.value
            _mcmc = MCMC(_kernel, num_warmup=_n_warmup, num_samples=_n_samples,
                        progress_bar=False)
            _show_popup(
                f'MCMC (NUTS) &mdash; {_n_warmup} warmup + {_n_samples} samples',
                'Running...',
            )
            _mcmc.run(_key, X_jax, y_jax)
            _mcmc_samples = _mcmc.get_samples()

            # Log-likelihood trace for diagnostics
            if 'prec_obs' in _mcmc_samples:
                _trace_vals = [float(v) for v in _mcmc_samples['prec_obs']]
            else:
                # Use w0 first element as trace diagnostic
                _trace_vals = [float(v) for v in _mcmc_samples['w0'][:, 0, 0]]
            set_losses(_trace_vals)
            set_result({'type': 'mcmc', 'samples': _mcmc_samples})

            # Posterior predictive
            _show_popup('MCMC (NUTS)', 'Sampling posterior predictive...')
            _X_plot = jnp.linspace(-2, 2, 200)
            _pred_key = jax.random.PRNGKey(seed_slider.value + 100)
            _predictive = Predictive(_model_fn, posterior_samples=_mcmc_samples)
            _pred_samples = _predictive(_pred_key, _X_plot)
            _obs_samples = _pred_samples['obs']

            _mean = np.array(jnp.mean(_obs_samples, axis=0))
            _std = np.array(jnp.std(_obs_samples, axis=0))
            _q05 = np.array(jnp.percentile(_obs_samples, 5, axis=0))
            _q25 = np.array(jnp.percentile(_obs_samples, 25, axis=0))
            _q75 = np.array(jnp.percentile(_obs_samples, 75, axis=0))
            _q95 = np.array(jnp.percentile(_obs_samples, 95, axis=0))

            set_pred_stats({
                'mean': _mean, 'std': _std,
                'q05': _q05, 'q25': _q25, 'q75': _q75, 'q95': _q95,
                'X_plot': np.array(_X_plot),
            })

            _n_show = min(20, _n_samples)
            set_sample_traces(np.array(_obs_samples[:_n_show]))

            # Diagnostics: ESS from numpyro
            from numpyro.diagnostics import effective_sample_size
            _ess_vals = []
            for _pname, _pval in _mcmc_samples.items():
                # Add chain dimension for ESS computation (single chain)
                _chain_vals = _pval[jnp.newaxis, ...]
                _ess = effective_sample_size(_chain_vals)
                _ess_vals.extend(np.ravel(np.array(_ess)).tolist())

            set_diag_info({
                'type': 'mcmc',
                'n_samples': _n_samples,
                'n_warmup': _n_warmup,
                'min_ess': min(_ess_vals) if _ess_vals else float('nan'),
                'max_rhat': float('nan'),  # single chain, R-hat not meaningful
            })

        # Dismiss popup
        mo.output.replace(mo.Html(''))

        set_trained(True)
        set_train_params(current_params.copy())

    # Handle store button
    if store_button.value:
        _pred_stats = get_pred_stats()
        if _pred_stats is not None:
            _stored = get_stored_fits()
            _label = f"Fit {len(_stored) + 1}: {inference_method.upper()}, W={width}, D={depth}, {activation_name}"
            _new_fit = {
                'mean': _pred_stats['mean'].copy(),
                'q05': _pred_stats['q05'].copy(),
                'q95': _pred_stats['q95'].copy(),
                'X_plot': _pred_stats['X_plot'].copy(),
                'label': _label,
            }
            set_stored_fits(_stored + [_new_fit])

    # Handle clear stored button
    if clear_stored_button.value:
        set_stored_fits([])

    return X_train, y_train, sigma_train, width, depth, train_min, train_max, current_params


@app.cell(hide_code=True)
def _(
    alt, pd, mo, np,
    X_train, y_train, sigma_train, function_dropdown, current_params, train_min, train_max,
    get_ground_truth, get_trained, get_train_params, get_pred_stats,
    get_sample_traces, get_stored_fits, show_samples_checkbox,
    get_custom_code,
):
    # Prepare data
    X_plot = np.linspace(-2, 2, 200)
    _custom_code = get_custom_code() if function_dropdown.value == 'custom' else None
    y_gt = get_ground_truth(X_plot, function_dropdown.value, _custom_code)

    gt_df = pd.DataFrame({'x': X_plot, 'y': y_gt, 'label': 'Ground truth'})
    train_df = pd.DataFrame({
        'x': X_train, 'y': y_train,
        'y_lo': y_train - 2 * sigma_train,
        'y_hi': y_train + 2 * sigma_train,
        'label': 'Training data',
    })
    train_region_df = pd.DataFrame({
        'x': [train_min], 'x2': [train_max],
        'y': [-1.5], 'y2': [1.5]
    })

    # Check staleness
    _train_params = get_train_params()
    _is_stale = _train_params is not None and _train_params != current_params

    # Scales
    x_scale = alt.Scale(domain=[-2.1, 2.1])
    y_scale = alt.Scale(domain=[-1.5, 1.5])

    # Legend color/dash scales for all series
    _legend_domain = ['Ground truth', 'Training data', 'Mean prediction',
                      '50% credible interval', '90% credible interval']
    _legend_colors = ['black', '#1f77b4', '#ff7f0e', '#ff7f0e', '#ff7f0e']

    # Training region
    region_layer = alt.Chart(train_region_df).mark_rect(
        color='#1f77b4', opacity=0.1
    ).encode(
        x=alt.X('x:Q', scale=x_scale), x2='x2:Q',
        y=alt.Y('y:Q', scale=y_scale), y2='y2:Q'
    )

    # Ground truth
    gt_layer = alt.Chart(gt_df).mark_line(
        strokeWidth=3, opacity=0.7
    ).encode(
        x=alt.X('x:Q', scale=x_scale, title='x'),
        y=alt.Y('y:Q', scale=y_scale, title='y'),
        color=alt.Color('label:N',
                        scale=alt.Scale(domain=_legend_domain, range=_legend_colors),
                        legend=alt.Legend(title=None)),
    )

    # Training points with error bars
    train_errorbars = alt.Chart(train_df).mark_rule(
        color='#1f77b4', opacity=0.4, strokeWidth=1.5
    ).encode(
        x=alt.X('x:Q', scale=x_scale),
        y=alt.Y('y_lo:Q', scale=y_scale),
        y2='y_hi:Q',
    )
    train_layer = alt.Chart(train_df).mark_circle(
        size=100, opacity=0.7
    ).encode(
        x=alt.X('x:Q', scale=x_scale),
        y=alt.Y('y:Q', scale=y_scale),
        color=alt.Color('label:N',
                        scale=alt.Scale(domain=_legend_domain, range=_legend_colors),
                        legend=alt.Legend(title=None)),
    )

    layers = [region_layer, gt_layer, train_errorbars, train_layer]

    # Stored fits
    _stored_fits = get_stored_fits()
    _stored_colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    if _stored_fits:
        stored_fit_rows = []
        stored_band_rows = []
        for _i, _fit in enumerate(_stored_fits):
            for _x, _m, _lo, _hi in zip(_fit['X_plot'], _fit['mean'], _fit['q05'], _fit['q95']):
                stored_fit_rows.append({'x': _x, 'y': _m, 'label': _fit['label']})
                stored_band_rows.append({'x': _x, 'y_lo': _lo, 'y_hi': _hi, 'label': _fit['label']})
        stored_fits_df = pd.DataFrame(stored_fit_rows)
        stored_bands_df = pd.DataFrame(stored_band_rows)
        stored_line = alt.Chart(stored_fits_df).mark_line(
            strokeWidth=2.5, opacity=0.8
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale),
            color=alt.Color('label:N', scale=alt.Scale(range=_stored_colors), legend=alt.Legend(title='Stored Fits'))
        )
        stored_band = alt.Chart(stored_bands_df).mark_area(
            opacity=0.1
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y_lo:Q', scale=y_scale),
            y2='y_hi:Q',
            color=alt.Color('label:N', scale=alt.Scale(range=_stored_colors), legend=None)
        )
        layers.append(stored_band)
        layers.append(stored_line)

    # Current prediction bands and mean
    _pred_stats = get_pred_stats()
    if _pred_stats is not None and get_trained() and not _is_stale:
        _ps = _pred_stats

        # 90% credible band
        _band90_df = pd.DataFrame({
            'x': _ps['X_plot'], 'y': _ps['q05'], 'y2': _ps['q95'],
            'label': '90% credible interval',
        })
        band90 = alt.Chart(_band90_df).mark_area(
            opacity=0.2
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale),
            y2='y2:Q',
            color=alt.Color('label:N',
                            scale=alt.Scale(domain=_legend_domain, range=_legend_colors),
                            legend=alt.Legend(title=None)),
        )
        layers.append(band90)

        # 50% credible band
        _band50_df = pd.DataFrame({
            'x': _ps['X_plot'], 'y': _ps['q25'], 'y2': _ps['q75'],
            'label': '50% credible interval',
        })
        band50 = alt.Chart(_band50_df).mark_area(
            opacity=0.35
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale),
            y2='y2:Q',
            color=alt.Color('label:N',
                            scale=alt.Scale(domain=_legend_domain, range=_legend_colors),
                            legend=alt.Legend(title=None)),
        )
        layers.append(band50)

        # Mean line
        _mean_df = pd.DataFrame({
            'x': _ps['X_plot'], 'y': _ps['mean'],
            'label': 'Mean prediction',
        })
        mean_layer = alt.Chart(_mean_df).mark_line(
            strokeWidth=3
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale),
            color=alt.Color('label:N',
                            scale=alt.Scale(domain=_legend_domain, range=_legend_colors),
                            legend=alt.Legend(title=None)),
        )
        layers.append(mean_layer)

        # Optional sample traces
        _sample_traces = get_sample_traces()
        if show_samples_checkbox.value and _sample_traces is not None:
            trace_rows = []
            _n_show = min(20, len(_sample_traces))
            for _s in range(_n_show):
                for _j, _x in enumerate(_ps['X_plot']):
                    trace_rows.append({'x': _x, 'y': float(_sample_traces[_s, _j]),
                                       'sample': str(_s)})
            _trace_df = pd.DataFrame(trace_rows)
            trace_layer = alt.Chart(_trace_df).mark_line(
                strokeWidth=0.8, opacity=0.3, color='#ff7f0e'
            ).encode(
                x=alt.X('x:Q', scale=x_scale),
                y=alt.Y('y:Q', scale=y_scale),
                detail='sample:N',
            )
            layers.append(trace_layer)

    elif _pred_stats is not None and _is_stale:
        # Stale prediction shown in grey
        _ps = _pred_stats
        stale_df = pd.DataFrame({'x': _ps['X_plot'], 'y': _ps['mean']})
        stale_layer = alt.Chart(stale_df).mark_line(
            color='lightgrey', strokeWidth=3
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale),
        )
        layers.append(stale_layer)

    data_chart = alt.layer(*layers).properties(
        width='container', height=300, title='BNN Regression with Uncertainty'
    ).configure_axis(
        grid=True, gridOpacity=0.3,
        labelFontSize=14, titleFontSize=16
    ).configure_title(fontSize=18)

    data_chart_output = mo.as_html(data_chart)

    return (data_chart_output,)


@app.cell(hide_code=True)
def _(
    alt, pd, mo, np,
    get_losses, get_trained, get_train_params, current_params,
    inference_dropdown,
):
    # Loss / diagnostic chart
    _losses = get_losses()
    _is_stale = get_train_params() is not None and get_train_params() != current_params
    _inf_method = inference_dropdown.value

    if _losses and get_trained():
        if _inf_method == 'vi' and not _is_stale:
            loss_df = pd.DataFrame({
                'step': np.arange(1, len(_losses) + 1),
                'loss': _losses,
            })
            _color = 'lightgrey' if _is_stale else '#ff7f0e'
            loss_chart = alt.Chart(loss_df).mark_line(
                color=_color, strokeWidth=2.5
            ).encode(
                x=alt.X('step:Q', title='SVI Step'),
                y=alt.Y('loss:Q', title='-ELBO'),
            ).properties(
                width='container', height=180, title='ELBO Convergence'
            ).configure_axis(
                grid=True, gridOpacity=0.3,
                labelFontSize=14, titleFontSize=16
            ).configure_title(fontSize=18)
        else:
            # MCMC trace plot
            _trace_df = pd.DataFrame({
                'sample': np.arange(1, len(_losses) + 1),
                'value': _losses,
            })
            _color = 'lightgrey' if _is_stale else '#ff7f0e'
            _y_title = 'prec_obs' if _inf_method == 'mcmc' else 'trace'
            loss_chart = alt.Chart(_trace_df).mark_line(
                color=_color, strokeWidth=1.5
            ).encode(
                x=alt.X('sample:Q', title='Sample Index'),
                y=alt.Y('value:Q', title=_y_title),
            ).properties(
                width='container', height=180, title='MCMC Trace Plot'
            ).configure_axis(
                grid=True, gridOpacity=0.3,
                labelFontSize=14, titleFontSize=16
            ).configure_title(fontSize=18)
    else:
        # Empty placeholder
        _placeholder = pd.DataFrame({'x': [0], 'y': [0]})
        loss_chart = alt.Chart(_placeholder).mark_point(opacity=0).encode(
            x=alt.X('x:Q', title='Step'),
            y=alt.Y('y:Q', title='Loss'),
        ).properties(
            width='container', height=180, title='Training Diagnostics'
        ).configure_axis(
            grid=True, gridOpacity=0.3,
            labelFontSize=14, titleFontSize=16
        ).configure_title(fontSize=18)

    loss_chart_output = mo.as_html(loss_chart)
    return (loss_chart_output,)


@app.cell(hide_code=True)
def _(
    mo,
    n_points_slider, noise_slider, function_dropdown, seed_slider, train_range_slider,
    width_slider, depth_slider, activation_dropdown,
    noise_type_dropdown, hetero_scale_slider,
    inference_dropdown, svi_steps_slider, vi_lr_slider,
    mcmc_samples_slider, mcmc_warmup_slider,
    prior_scale_slider, pred_samples_slider, show_samples_checkbox,
    train_button, reset_button, store_button, clear_stored_button,
    custom_function_accordion,
):
    # Conditional visibility for noise type
    _hetero_controls = mo.Html(f'{hetero_scale_slider}') if noise_type_dropdown.value == 'hetero' else mo.Html('')

    # Conditional visibility for inference method
    if inference_dropdown.value == 'vi':
        _inf_controls = mo.Html(f'''
            {svi_steps_slider}
            {vi_lr_slider}
            {pred_samples_slider}
        ''')
    else:
        _inf_controls = mo.Html(f'''
            {mcmc_samples_slider}
            {mcmc_warmup_slider}
        ''')

    sidebar = mo.Html(f'''
    <div class="app-sidebar">
        <h4>Data</h4>
        {n_points_slider}
        {noise_slider}
        {function_dropdown}
        {custom_function_accordion}
        {seed_slider}
        {train_range_slider}

        <h4>Network Architecture</h4>
        {width_slider}
        {depth_slider}
        {activation_dropdown}

        <h4>Noise Model</h4>
        {noise_type_dropdown}
        {_hetero_controls}

        <h4>Inference</h4>
        {inference_dropdown}
        {prior_scale_slider}
        {_inf_controls}
        {show_samples_checkbox}

        <div style="display: flex; gap: 0.5em; margin-top: 1em; flex-wrap: wrap;">
            {train_button}
            {reset_button}
            {store_button}
            {clear_stored_button}
        </div>
    </div>
    ''')
    return (sidebar,)


@app.cell(hide_code=True)
def _(mo, width, depth, X_train, get_diag_info, noise_type_dropdown):
    import math as _math

    # Compute parameter counts
    # First hidden: 1*width + width = 2*width
    # Subsequent: width*width + width = width*(width+1) each
    # Output head: width + 1
    _n_params = 2 * width + (depth - 1) * width * (width + 1) + width + 1
    if noise_type_dropdown.value == 'hetero':
        _n_params += width + 1  # extra output head
    else:
        _n_params += 1  # prec_obs

    _n_train = len(X_train)
    _ratio = _n_train / _n_params if _n_params > 0 else 0

    _diag = get_diag_info()
    if _diag is not None and _diag['type'] == 'vi':
        _n_var = 2 * _n_params  # AutoNormal: mean + scale per param
        _diag_str = f" | V={_n_var:,} | ELBO={_diag['final_elbo']:.1f}"
    elif _diag is not None and _diag['type'] == 'mcmc':
        _min_ess = _diag.get('min_ess', float('nan'))
        _max_rhat = _diag.get('max_rhat', float('nan'))
        _ess_str = f"{_min_ess:.0f}" if not _math.isnan(_min_ess) else "N/A"
        _rhat_str = f"{_max_rhat:.2f}" if not _math.isnan(_max_rhat) else "N/A"
        _diag_str = f" | ESS(min)={_ess_str} | R\u0302={_rhat_str}"
    else:
        _diag_str = ""

    param_info = mo.Html(f'''
    <div style="text-align: center; padding: 0.5em; color: #666; font-size: 14px;">
        N={_n_train} | P={_n_params:,} | N/P={_ratio:.2f}{_diag_str}
    </div>
    ''')
    return (param_info,)


@app.cell(hide_code=True)
def _(mo, header, data_chart_output, loss_chart_output, sidebar, param_info):
    mo.vstack([
        header,
        mo.Html(f'''
        <div class="app-layout">
            <div class="app-plot">
                {mo.as_html(data_chart_output)}
                {mo.as_html(loss_chart_output)}
                {param_info}
            </div>
            {sidebar}
        </div>
        ''')
    ])
    return


if __name__ == "__main__":
    app.run()
