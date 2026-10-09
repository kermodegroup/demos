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
#     "pyarrow",
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

        .app-sidebar-container {
            z-index: 10;
            position: relative;
            flex-shrink: 0;
            width: 300px;
        }

        .app-sidebar {
            display: flex;
            flex-direction: column;
            gap: 1.5em;
            padding: 1.5em;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            width: 100%;
        }

        @media (max-width: 768px) {
            .app-layout {
                flex-direction: column;
                height: auto;
                overflow-y: auto;
            }
            .app-plot {
                max-width: 100%;
                width: 100%;
            }
            .app-sidebar-container {
                width: 100%;
            }
            .app-sidebar {
                width: 100%;
            }
        }

        .app-sidebar h4 {
            margin: 1em 0 0.5em 0;
            font-size: 0.9em;
            color: #495057;
            border-bottom: 1px solid #dee2e6;
            padding-bottom: 0.3em;
        }

        .app-sidebar h4:first-child {
            margin-top: 0;
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
    from jax import grad

    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'cpu')

    return alt, grad, jax, jnp, mo, np, pd


@app.cell(hide_code=True)
def _():
    import qrcode
    import io
    import base64

    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://sciml.warwick.ac.uk/live/gp-derivative-demo/')
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
                <p style='font-size: 24px; margin: 0; padding: 0; line-height: 1.3;'><b>GP with Derivative Observations</b>
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
    # Target function selection
    target_dropdown = mo.ui.dropdown(
        options={
            'Sine': 'sin',
            'Step': 'step',
            'Runge': 'runge',
            'Witch of Agnesi': 'witch',
        },
        value='Sine',
        label='Target Function'
    )

    # Kernel hyperparameters
    variance_slider = mo.ui.slider(0.1, 5.0, 0.1, 1.0, label='Variance $v$')
    lengthscale_slider = mo.ui.slider(0.05, 1.0, 0.05, 0.3, label='Lengthscale $\\ell$')

    # Noise parameters (separate for function and derivative observations)
    noise_f_slider = mo.ui.slider(0.01, 0.5, 0.01, 0.1, label='Function noise $\\sigma_f$')
    noise_df_slider = mo.ui.slider(0.01, 0.5, 0.01, 0.1, label='Derivative noise $\\sigma_{f\'}$')

    # Sampling controls
    n_samples_slider = mo.ui.slider(0, 20, 1, 5, label='$N$ samples')

    # Show derivative plot toggle
    show_derivative_plot = mo.ui.checkbox(value=True, label='Show f\'(x) plot')

    return (
        target_dropdown,
        variance_slider,
        lengthscale_slider,
        noise_f_slider,
        noise_df_slider,
        n_samples_slider,
        show_derivative_plot,
    )


@app.cell(hide_code=True)
def _(mo):
    # State for clicked observations
    # Each entry: (x, y, obs_type) where obs_type is 'function' or 'derivative'
    get_observations, set_observations = mo.state([])

    return get_observations, set_observations


@app.cell(hide_code=True)
def _(jnp, grad):
    def target_function(x, func_type):
        """Generate target function values (JAX-compatible)."""
        if func_type == 'sin':
            return 0.5 + jnp.sin(2 * jnp.pi * x)
        elif func_type == 'step':
            return jnp.where(x < 0, -0.5, 0.5)
        elif func_type == 'runge':
            return 1.0 / (1.0 + 25.0 * x**2)
        elif func_type == 'witch':
            return 1.0 / (1.0 + x**2)
        return jnp.sin(x)

    def target_derivative(x, func_type):
        """Compute derivative of target function using JAX autodiff."""
        return grad(lambda t: target_function(t, func_type))(x)

    # Base RBF kernel (scalar inputs)
    def rbf_kernel_scalar(x1, x2, variance, lengthscale):
        """RBF kernel for scalar inputs."""
        sqdist = (x1 - x2) ** 2
        return variance * jnp.exp(-0.5 * sqdist / lengthscale**2)

    return target_function, target_derivative, rbf_kernel_scalar


@app.cell(hide_code=True)
def _(jax, jnp):
    # Vectorized derivative kernel using closed-form RBF derivatives
    # Much faster than using autodiff in loops

    @jax.jit
    def _build_kernel_matrix(X1, D1, X2, D2, variance, lengthscale):
        """
        Build kernel matrix for mixed function/derivative observations.
        Uses closed-form RBF kernel derivatives for speed.
        """
        diff = X1[:, None] - X2[None, :]
        sqdist = diff ** 2
        l2 = lengthscale ** 2
        K_base = variance * jnp.exp(-0.5 * sqdist / l2)
        K_dx1 = -K_base * diff / l2
        K_dx2 = K_base * diff / l2
        K_dx1dx2 = K_base * (1/l2 - sqdist / (l2 * l2))
        d1 = D1[:, None]
        d2 = D2[None, :]
        K = jnp.where(d1,
            jnp.where(d2, K_dx1dx2, K_dx1),
            jnp.where(d2, K_dx2, K_base))
        return K

    def gp_posterior(X_train, D_train, y_train, noise_f, noise_df,
                     X_test, D_test, variance, lengthscale):
        """
        Compute GP posterior with mixed function and derivative observations.
        """
        N_train = len(X_train)
        N_test = len(X_test)

        # Prior covariance at test points
        K_star = _build_kernel_matrix(X_test, D_test, X_test, D_test, variance, lengthscale)

        if N_train == 0:
            return jnp.zeros(N_test), K_star

        # Training kernel matrix with noise
        K = _build_kernel_matrix(X_train, D_train, X_train, D_train, variance, lengthscale)
        noise_diag = jnp.where(D_train, noise_df**2, noise_f**2)
        K_noisy = K + jnp.diag(noise_diag) + 1e-6 * jnp.eye(N_train)

        # Cross-kernel matrix
        K_cross = _build_kernel_matrix(X_train, D_train, X_test, D_test, variance, lengthscale)

        # Cholesky solve
        L = jnp.linalg.cholesky(K_noisy)
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y_train))
        f_mean = K_cross.T @ alpha

        v = jnp.linalg.solve(L, K_cross)
        f_cov = K_star - v.T @ v

        return f_mean, f_cov

    return (gp_posterior,)


@app.cell(hide_code=True)
def _(np, pd):
    # Static click grid for function plot (1:1 aspect ratio)
    _gx = np.linspace(-2, 2, 80)
    _gy_f = np.linspace(-2, 2, 80)
    click_grid_f_df = pd.DataFrame(
        [(x, y) for x in _gx for y in _gy_f],
        columns=['x', 'y']
    )

    # Static click grid for derivative plot (y range: -10 to 10 for slopes)
    _gy_df = np.linspace(-10, 10, 80)
    click_grid_df_df = pd.DataFrame(
        [(x, y) for x in _gx for y in _gy_df],
        columns=['x', 'y']
    )

    return click_grid_f_df, click_grid_df_df


@app.cell(hide_code=True)
def _(mo, set_observations, get_observations):
    def clear_observations(_):
        set_observations([])

    clear_button = mo.ui.button(label="Clear All", on_click=clear_observations)

    def undo_last(_):
        obs = get_observations() or []
        if obs:
            set_observations(obs[:-1])

    undo_button = mo.ui.button(label="Undo", on_click=undo_last)

    return clear_button, undo_button


@app.cell(hide_code=True)
def _(
    alt, np, pd, jax, jnp, mo,
    target_function, target_derivative,
    gp_posterior,
    target_dropdown, variance_slider, lengthscale_slider,
    noise_f_slider, noise_df_slider, n_samples_slider,
    show_derivative_plot,
    get_observations,
    click_grid_f_df, click_grid_df_df,
):
    # Get parameter values
    func_type = target_dropdown.value
    variance = variance_slider.value
    lengthscale = lengthscale_slider.value
    noise_f = noise_f_slider.value
    noise_df = noise_df_slider.value
    n_samples = n_samples_slider.value
    show_deriv = show_derivative_plot.value

    # Fixed axis limits (1:1 aspect ratio for correct tangent visualization)
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2

    # Get observations
    observations = get_observations() or []

    # Separate function and derivative observations
    func_obs = [(x, y) for x, y, t in observations if t == 'function']
    deriv_obs = [(x, y) for x, y, t in observations if t == 'derivative']

    # Build training data
    if observations:
        X_train = jnp.array([x for x, y, t in observations])
        y_train = jnp.array([y for x, y, t in observations])
        D_train = jnp.array([1.0 if t == 'derivative' else 0.0 for x, y, t in observations])
    else:
        X_train = jnp.array([])
        y_train = jnp.array([])
        D_train = jnp.array([])

    # Test points for function predictions
    X_test_f = jnp.linspace(x_min, x_max, 150)
    D_test_f = jnp.zeros_like(X_test_f)

    # Test points for derivative predictions
    X_test_df = jnp.linspace(x_min, x_max, 150)
    D_test_df = jnp.ones_like(X_test_df)

    # Compute posteriors
    f_mean, f_cov = gp_posterior(
        X_train, D_train, y_train, noise_f, noise_df,
        X_test_f, D_test_f, variance, lengthscale
    )

    df_mean, df_cov = gp_posterior(
        X_train, D_train, y_train, noise_f, noise_df,
        X_test_df, D_test_df, variance, lengthscale
    )

    f_std = jnp.sqrt(jnp.diag(f_cov))
    df_std = jnp.sqrt(jnp.diag(df_cov))

    # Ground truth
    y_true = jax.vmap(lambda x: target_function(x, func_type))(X_test_f)
    dy_true = jax.vmap(lambda x: target_derivative(x, func_type))(X_test_df)

    # Generate samples from function posterior
    f_samples_data = []
    df_samples_data = []
    if n_samples > 0:
        f_cov_jitter = f_cov + 1e-6 * jnp.eye(len(X_test_f))
        df_cov_jitter = df_cov + 1e-6 * jnp.eye(len(X_test_df))

        L_f = jnp.linalg.cholesky(f_cov_jitter)
        L_df = jnp.linalg.cholesky(df_cov_jitter)

        np.random.seed(42)
        for i in range(n_samples):
            z_f = np.random.randn(len(X_test_f))
            z_df = np.random.randn(len(X_test_df))
            f_sample = f_mean + L_f @ z_f
            df_sample = df_mean + L_df @ z_df
            for x_val, y_val in zip(np.array(X_test_f), np.array(f_sample)):
                f_samples_data.append({'x': x_val, 'y': y_val, 'sample': f'Sample {i+1}'})
            for x_val, y_val in zip(np.array(X_test_df), np.array(df_sample)):
                df_samples_data.append({'x': x_val, 'y': y_val, 'sample': f'Sample {i+1}'})

    # Build DataFrames for function plot
    gt_df = pd.DataFrame({'x': np.array(X_test_f), 'y': np.array(y_true)})
    mean_df = pd.DataFrame({'x': np.array(X_test_f), 'y': np.array(f_mean)})
    band_df = pd.DataFrame({
        'x': np.array(X_test_f),
        'y_mean': np.array(f_mean),
        'y_lower': np.array(f_mean - 2 * f_std),
        'y_upper': np.array(f_mean + 2 * f_std),
    })
    f_samples_df = pd.DataFrame(f_samples_data) if f_samples_data else pd.DataFrame(columns=['x', 'y', 'sample'])

    # Function observations DataFrame
    if func_obs:
        func_obs_df = pd.DataFrame(func_obs, columns=['x', 'y'])
    else:
        func_obs_df = pd.DataFrame(columns=['x', 'y'])

    # Derivative observations DataFrame with tangent lines
    deriv_obs_data = []
    tangent_data = []
    tangent_length = 0.1  # Half-length of tangent line
    for x, slope in deriv_obs:
        deriv_obs_data.append({'x': x, 'y': slope, 'slope': slope})
        # Tangent line endpoints
        tangent_data.append({
            'x1': x - tangent_length, 'y1': -tangent_length * slope,
            'x2': x + tangent_length, 'y2': tangent_length * slope,
            'x_center': x, 'slope': slope
        })
    deriv_obs_df = pd.DataFrame(deriv_obs_data) if deriv_obs_data else pd.DataFrame(columns=['x', 'y', 'slope'])
    tangent_df = pd.DataFrame(tangent_data) if tangent_data else pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2', 'x_center', 'slope'])

    # Derivative plot DataFrames
    dgt_df = pd.DataFrame({'x': np.array(X_test_df), 'y': np.array(dy_true)})
    dmean_df = pd.DataFrame({'x': np.array(X_test_df), 'y': np.array(df_mean)})
    dband_df = pd.DataFrame({
        'x': np.array(X_test_df),
        'y_mean': np.array(df_mean),
        'y_lower': np.array(df_mean - 2 * df_std),
        'y_upper': np.array(df_mean + 2 * df_std),
    })
    df_samples_df = pd.DataFrame(df_samples_data) if df_samples_data else pd.DataFrame(columns=['x', 'y', 'sample'])

    # Define scales
    x_scale = alt.Scale(domain=[x_min, x_max])
    y_scale_f = alt.Scale(domain=[y_min, y_max])
    y_scale_df = alt.Scale(domain=[-10, 10])

    # Click selection
    click_select = alt.selection_point(on='click', nearest=True, fields=['x', 'y'], name='click_select')

    # === Function Plot ===
    # Uncertainty band
    f_band = alt.Chart(band_df).mark_area(
        opacity=0.3, color='#1f77b4'
    ).encode(
        x=alt.X('x:Q', scale=x_scale, title='x'),
        y=alt.Y('y_lower:Q', scale=y_scale_f, title='f(x)'),
        y2='y_upper:Q'
    )

    # Posterior mean
    f_mean_line = alt.Chart(mean_df).mark_line(
        color='#1f77b4', strokeWidth=3
    ).encode(
        x=alt.X('x:Q', scale=x_scale),
        y=alt.Y('y:Q', scale=y_scale_f)
    )

    # Ground truth
    f_gt_line = alt.Chart(gt_df).mark_line(
        color='black', strokeWidth=3, strokeDash=[5, 5], opacity=0.7
    ).encode(
        x=alt.X('x:Q', scale=x_scale),
        y=alt.Y('y:Q', scale=y_scale_f)
    )

    f_layers = [f_band]

    # Samples
    if len(f_samples_df) > 0:
        f_samples_layer = alt.Chart(f_samples_df).mark_line(
            strokeWidth=1.5, opacity=0.4
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale_f),
            color=alt.Color('sample:N', legend=None)
        )
        f_layers.append(f_samples_layer)

    f_layers.extend([f_mean_line, f_gt_line])

    # Function observations (blue circles)
    if len(func_obs_df) > 0:
        f_obs_layer = alt.Chart(func_obs_df).mark_circle(
            color='#1f77b4', size=150, opacity=0.9
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale_f)
        )
        f_layers.append(f_obs_layer)

    # Derivative observations shown as tangent lines on function plot
    if len(tangent_df) > 0:
        # For function plot, we need to show tangent at the posterior mean location
        # First get the posterior mean at each derivative observation location
        deriv_x = jnp.array([x for x, _ in deriv_obs])
        deriv_slopes = [s for _, s in deriv_obs]

        # Query posterior at derivative locations (function values)
        if len(deriv_x) > 0:
            D_query = jnp.zeros_like(deriv_x)
            f_at_deriv, _ = gp_posterior(
                X_train, D_train, y_train, noise_f, noise_df,
                deriv_x, D_query, variance, lengthscale
            )

            # Build tangent data centered at posterior mean
            # With 1:1 aspect ratio, tangent lines display correctly
            tangent_f_data = []
            for i, (x, slope) in enumerate(deriv_obs):
                y_center = float(f_at_deriv[i])
                tangent_f_data.append({
                    'x1': x - tangent_length, 'y1': y_center - tangent_length * slope,
                    'x2': x + tangent_length, 'y2': y_center + tangent_length * slope,
                })
            tangent_f_df = pd.DataFrame(tangent_f_data)

            tangent_layer = alt.Chart(tangent_f_df).mark_rule(
                color='#d62728', strokeWidth=3, opacity=0.9
            ).encode(
                x=alt.X('x1:Q', scale=x_scale),
                y=alt.Y('y1:Q', scale=y_scale_f),
                x2='x2:Q',
                y2='y2:Q'
            )
            f_layers.append(tangent_layer)

    # Click layer for function plot
    f_click_layer = alt.Chart(click_grid_f_df).mark_point(
        opacity=0, size=100
    ).encode(
        x=alt.X('x:Q', scale=x_scale),
        y=alt.Y('y:Q', scale=y_scale_f)
    ).add_params(click_select)
    f_layers.append(f_click_layer)

    f_chart = alt.layer(*f_layers).properties(
        width='container', height=280 if show_deriv else 400,
        title='Function f(x)'
    )

    # === Derivative Plot ===
    if show_deriv:
        # Uncertainty band
        df_band = alt.Chart(dband_df).mark_area(
            opacity=0.3, color='#ff7f0e'
        ).encode(
            x=alt.X('x:Q', scale=x_scale, title='x'),
            y=alt.Y('y_lower:Q', scale=y_scale_df, title="f'(x)"),
            y2='y_upper:Q'
        )

        # Posterior mean
        df_mean_line = alt.Chart(dmean_df).mark_line(
            color='#ff7f0e', strokeWidth=3
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale_df)
        )

        # Ground truth
        df_gt_line = alt.Chart(dgt_df).mark_line(
            color='black', strokeWidth=3, strokeDash=[5, 5], opacity=0.7
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale_df)
        )

        df_layers = [df_band]

        # Samples
        if len(df_samples_df) > 0:
            df_samples_layer = alt.Chart(df_samples_df).mark_line(
                strokeWidth=1.5, opacity=0.4
            ).encode(
                x=alt.X('x:Q', scale=x_scale),
                y=alt.Y('y:Q', scale=y_scale_df),
                color=alt.Color('sample:N', legend=None)
            )
            df_layers.append(df_samples_layer)

        df_layers.extend([df_mean_line, df_gt_line])

        # Derivative observations (red circles)
        if len(deriv_obs_df) > 0:
            df_obs_layer = alt.Chart(deriv_obs_df).mark_circle(
                color='#d62728', size=150, opacity=0.9
            ).encode(
                x=alt.X('x:Q', scale=x_scale),
                y=alt.Y('y:Q', scale=y_scale_df)
            )
            df_layers.append(df_obs_layer)

        # Click layer for derivative plot
        df_click_select = alt.selection_point(on='click', nearest=True, fields=['x', 'y'], name='df_click_select')
        df_click_layer = alt.Chart(click_grid_df_df).mark_point(
            opacity=0, size=100
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale_df)
        ).add_params(df_click_select)
        df_layers.append(df_click_layer)

        df_chart = alt.layer(*df_layers).properties(
            width='container', height=200,
            title="Derivative f'(x)"
        )

        combined_chart = alt.vconcat(
            f_chart, df_chart
        ).configure_axis(
            grid=True, gridOpacity=0.3,
            labelFontSize=14, titleFontSize=16
        ).configure_title(
            fontSize=18
        )
    else:
        combined_chart = f_chart.configure_axis(
            grid=True, gridOpacity=0.3,
            labelFontSize=14, titleFontSize=16
        ).configure_title(
            fontSize=18
        )

    interactive_chart = mo.ui.altair_chart(combined_chart)

    # Store counts for display
    n_func_obs = len(func_obs)
    n_deriv_obs = len(deriv_obs)

    return interactive_chart, n_func_obs, n_deriv_obs


@app.cell(hide_code=True)
def _(interactive_chart):
    chart_display = interactive_chart
    return (chart_display,)


@app.cell(hide_code=True)
def _(interactive_chart, click_grid_f_df, click_grid_df_df, get_observations, set_observations, pd):
    # Click handler - detect clicks on either plot
    _current = get_observations() or []

    # Check for click on function plot
    _filtered_f = interactive_chart.apply_selection(click_grid_f_df)
    _clicked_f = (_filtered_f is not None and isinstance(_filtered_f, pd.DataFrame)
                  and len(_filtered_f) > 0 and len(_filtered_f) < len(click_grid_f_df))

    # Check for click on derivative plot
    _filtered_df = interactive_chart.apply_selection(click_grid_df_df)
    _clicked_df = (_filtered_df is not None and isinstance(_filtered_df, pd.DataFrame)
                   and len(_filtered_df) > 0 and len(_filtered_df) < len(click_grid_df_df))

    if _clicked_f:
        # Click on function plot -> add function observation
        _new_x = float(_filtered_f['x'].iloc[0])
        _new_y = float(_filtered_f['y'].iloc[0])
        _new_obs = (_new_x, _new_y, 'function')
        _existing = [(x, t) for x, y, t in _current]
        if (_new_x, 'function') not in _existing:
            set_observations(_current + [_new_obs])

    elif _clicked_df:
        # Click on derivative plot -> add derivative observation (y = slope)
        _new_x = float(_filtered_df['x'].iloc[0])
        _new_slope = float(_filtered_df['y'].iloc[0])
        _new_obs = (_new_x, _new_slope, 'derivative')
        _existing = [(x, t) for x, y, t in _current]
        if (_new_x, 'derivative') not in _existing:
            set_observations(_current + [_new_obs])

    return ()


@app.cell(hide_code=True)
def _(
    mo,
    target_dropdown,
    variance_slider, lengthscale_slider,
    noise_f_slider, noise_df_slider,
    n_samples_slider, show_derivative_plot,
    clear_button, undo_button,
    n_func_obs, n_deriv_obs,
):
    # Function section
    func_section = mo.vstack([
        mo.Html("<h4>Function</h4>"),
        target_dropdown,
    ], gap="0.3em")

    # Kernel section
    kernel_section = mo.vstack([
        mo.Html("<h4>Kernel</h4>"),
        variance_slider,
        lengthscale_slider,
    ], gap="0.3em")

    # Noise section
    noise_section = mo.vstack([
        mo.Html("<h4>Noise</h4>"),
        noise_f_slider,
        noise_df_slider,
    ], gap="0.3em")

    # Observations section
    obs_section = mo.vstack([
        mo.Html("<h4>Observations</h4>"),
        mo.hstack([clear_button, undo_button], gap="0.5em"),
        mo.Html(f"<small>f(x): {n_func_obs} | f'(x): {n_deriv_obs}</small>"),
    ], gap="0.3em")

    # Sampling section
    sampling_section = mo.vstack([
        mo.Html("<h4>Display</h4>"),
        n_samples_slider,
        show_derivative_plot,
    ], gap="0.3em")

    sidebar = mo.vstack([
        func_section, kernel_section, noise_section, obs_section, sampling_section
    ], gap="1em")

    sidebar_html = mo.Html(f'''
    <div class="app-sidebar">
        {sidebar}
        <p style="font-size: 0.85em; color: #666; margin-top: 1em;">
            <b>Tip:</b> Click on top plot to add f(x) observations.
            Click on bottom plot to add f'(x) observations (y-value = slope).
            Tangent lines show derivative constraints.
        </p>
    </div>
    ''')
    return (sidebar_html,)


@app.cell(hide_code=True)
def _(mo, header, chart_display, sidebar_html):
    mo.Html(f'''
    {header}
    <div class="app-layout">
        <div class="app-plot">{mo.as_html(chart_display)}</div>
        <div class="app-sidebar-container">
            {sidebar_html}
        </div>
    </div>
    ''')
    return


if __name__ == "__main__":
    app.run()
