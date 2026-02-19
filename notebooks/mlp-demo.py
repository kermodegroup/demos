# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.2.5",
#     "qrcode==8.2",
#     "jax",
#     "jaxlib",
#     "equinox",
#     "optax",
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
    import optax

    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'cpu')

    return alt, eqx, jax, jnp, mo, np, optax, pd


@app.cell(hide_code=True)
def _():
    import qrcode
    import io
    import base64

    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://sciml.warwick.ac.uk/live/mlp-demo/')
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
                <p style='font-size: 24px; margin: 0; padding: 0; line-height: 1.3;'><b>MLP Regression Demo</b>
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

    # Network controls
    width_slider = mo.ui.slider(8, 128, 8, 32, label='Width (units per layer)')
    depth_slider = mo.ui.slider(1, 5, 1, 2, label='Depth (hidden layers)')
    activation_dropdown = mo.ui.dropdown(
        options={'ReLU': 'relu', 'Tanh': 'tanh', 'Sigmoid': 'sigmoid'},
        value='ReLU',
        label='Activation function'
    )

    # Training controls
    lr_slider = mo.ui.slider(-4, -1, 0.5, -2, label='Learning rate (10^x)')
    optimizer_dropdown = mo.ui.dropdown(
        options={'Adam': 'adam', 'SGD': 'sgd'},
        value='Adam',
        label='Optimizer'
    )
    epochs_slider = mo.ui.slider(100, 5000, 100, 1000, label='Number of epochs')
    l1_slider = mo.ui.slider(-6, -1, 0.5, -6, label='L1 reg (10^x)')
    l2_slider = mo.ui.slider(-6, -1, 0.5, -6, label='L2 reg (10^x)')
    train_range_slider = mo.ui.range_slider(
        start=-2, stop=2, step=0.1,
        value=[-1.0, 1.0],
        label='Training range'
    )

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
        n_points_slider,
        noise_slider,
        function_dropdown,
        seed_slider,
        width_slider,
        depth_slider,
        activation_dropdown,
        lr_slider,
        optimizer_dropdown,
        epochs_slider,
        l1_slider,
        l2_slider,
        train_range_slider,
        train_button,
        reset_button,
        store_button,
        clear_stored_button,
        get_custom_code,
        set_custom_code,
        custom_code_editor,
        custom_function_accordion,
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

    def generate_data(n_points, noise_std, function_type, seed, custom_code=None):
        """Generate training data from target function with noise."""
        rng = np.random.default_rng(seed)
        X = rng.uniform(-2, 2, n_points)
        X = np.sort(X)

        y_true = _eval_function(X, function_type, custom_code)
        y = y_true + rng.normal(0, noise_std, n_points)
        return X, y, y_true

    def get_ground_truth(X, function_type, custom_code=None):
        """Get ground truth values for plotting."""
        return _eval_function(X, function_type, custom_code)

    return generate_data, get_ground_truth


@app.cell(hide_code=True)
def _(eqx, jax, jnp):
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

    class MLP(eqx.Module):
        """Multi-layer perceptron for 1D regression."""
        layers: list

        def __init__(self, key, width, depth, activation):
            keys = jax.random.split(key, depth + 1)
            layers = []

            # Input layer: scalar -> width
            layers.append(eqx.nn.Linear('scalar', width, key=keys[0]))
            layers.append(activation)

            # Hidden layers
            for i in range(depth - 1):
                layers.append(eqx.nn.Linear(width, width, key=keys[i + 1]))
                layers.append(activation)

            # Output layer: width -> scalar
            layers.append(eqx.nn.Linear(width, 'scalar', key=keys[-1]))

            self.layers = layers

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return get_activation, MLP


@app.cell(hide_code=True)
def _(eqx, jax, jnp):
    def make_train_step(l1_reg, l2_reg):
        """Create a train step function with specified regularization."""
        @eqx.filter_jit
        def train_step(model, opt_state, optimizer, x, y):
            """Single training step with gradient update."""
            def loss_fn(model):
                pred = jax.vmap(model)(x)
                mse_loss = jnp.mean((pred - y) ** 2)

                # Get all weight arrays for regularization
                params = eqx.filter(model, eqx.is_array)
                leaves = jax.tree_util.tree_leaves(params)

                # L1 regularization (sum of absolute values)
                l1_term = sum(jnp.sum(jnp.abs(w)) for w in leaves)

                # L2 regularization (sum of squared values)
                l2_term = sum(jnp.sum(w ** 2) for w in leaves)

                return mse_loss + l1_reg * l1_term + l2_reg * l2_term

            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss
        return train_step

    def train_model(model, optimizer, x, y, n_epochs, l1_reg=0.0, l2_reg=0.0):
        """Train model for n_epochs and return loss history and snapshots."""
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        losses = []
        snapshots = []  # List of (epoch, model) tuples
        snapshot_interval = max(1, n_epochs // 100)  # Save every 1%
        train_step = make_train_step(l1_reg, l2_reg)

        for i in range(n_epochs):
            model, opt_state, loss = train_step(model, opt_state, optimizer, x, y)
            losses.append(float(loss))

            # Save snapshot at intervals and at the final epoch
            if i % snapshot_interval == 0 or i == n_epochs - 1:
                snapshots.append((i + 1, model))  # 1-indexed epoch

        return model, losses, snapshots

    return make_train_step, train_model


@app.cell(hide_code=True)
def _(mo):
    # State for model and training results
    get_model, set_model = mo.state(None)
    get_losses, set_losses = mo.state([])
    get_trained, set_trained = mo.state(False)
    # State for tracking if fit is stale (parameters changed since last train)
    get_train_params, set_train_params = mo.state(None)  # dict of params used during training
    get_last_pred, set_last_pred = mo.state(None)  # last prediction array
    # State for stored fits (list of dicts with 'pred', 'losses', 'label')
    get_stored_fits, set_stored_fits = mo.state([])
    # State for model snapshots during training
    get_snapshots, set_snapshots = mo.state([])
    # State for selected epoch from loss chart brush
    get_selected_epoch, set_selected_epoch = mo.state(None)
    return (get_model, set_model, get_losses, set_losses, get_trained, set_trained,
            get_snapshots, set_snapshots,
            get_train_params, set_train_params, get_last_pred, set_last_pred,
            get_stored_fits, set_stored_fits,
            get_selected_epoch, set_selected_epoch)




@app.cell(hide_code=True)
def _(
    jax, jnp, np, optax,
    n_points_slider, noise_slider, function_dropdown, seed_slider,
    width_slider, depth_slider, activation_dropdown,
    lr_slider, optimizer_dropdown, epochs_slider, l1_slider, l2_slider, train_range_slider,
    train_button, reset_button, store_button, clear_stored_button,
    generate_data, get_activation, MLP, train_model,
    get_model, set_model, get_losses, set_losses, get_trained, set_trained,
    get_train_params, set_train_params, get_last_pred, set_last_pred,
    get_snapshots, set_snapshots,
    get_stored_fits, set_stored_fits,
    get_custom_code,
):
    # Get custom code if using custom function
    custom_code = get_custom_code() if function_dropdown.value == 'custom' else None

    # Generate training data
    X_full, y_full, _ = generate_data(
        n_points_slider.value,
        noise_slider.value,
        function_dropdown.value,
        seed_slider.value,
        custom_code
    )

    # Filter to training range
    train_min, train_max = train_range_slider.value
    mask = (X_full >= train_min) & (X_full <= train_max)
    X_train = X_full[mask]
    y_train = y_full[mask]

    X_jax = jnp.array(X_train)
    y_jax = jnp.array(y_train)

    # Get network parameters
    width = width_slider.value
    depth = depth_slider.value
    activation_name = activation_dropdown.value
    activation = get_activation(activation_name)
    learning_rate = 10 ** lr_slider.value
    n_epochs = epochs_slider.value
    l1_reg = 10 ** l1_slider.value
    l2_reg = 10 ** l2_slider.value

    # Current parameters dict for staleness tracking
    current_params = {
        'n_points': n_points_slider.value,
        'noise': noise_slider.value,
        'function': function_dropdown.value,
        'custom_code': custom_code,
        'seed': seed_slider.value,
        'width': width,
        'depth': depth,
        'activation': activation_name,
        'lr': lr_slider.value,
        'optimizer': optimizer_dropdown.value,
        'epochs': n_epochs,
        'l1_reg': l1_slider.value,
        'l2_reg': l2_slider.value,
        'train_min': train_min,
        'train_max': train_max,
    }

    # Create optimizer
    if optimizer_dropdown.value == 'adam':
        optimizer = optax.adam(learning_rate)
    else:
        optimizer = optax.sgd(learning_rate)

    # Initialize or reset model
    key = jax.random.PRNGKey(seed_slider.value + 42)

    # Handle reset button
    if reset_button.value:
        _model = MLP(key, width, depth, activation)
        set_model(_model)
        set_losses([])
        set_trained(False)
        set_train_params(None)
        set_last_pred(None)
        set_snapshots([])

    # Handle train button
    if train_button.value:
        # Check if parameters changed (fit is stale) - if so, reinitialize model
        _train_params = get_train_params()
        _is_stale = _train_params is not None and _train_params != current_params

        if _is_stale:
            # Parameters changed - start fresh with new model
            _current_model = MLP(key, width, depth, activation)
            _prev_losses = []
        else:
            # Continue training existing model
            _current_model = get_model()
            if _current_model is None:
                _current_model = MLP(key, width, depth, activation)
            _prev_losses = get_losses() or []

        # Train model
        _trained_model, _new_losses, _new_snapshots = train_model(
            _current_model, optimizer, X_jax, y_jax, n_epochs,
            l1_reg=l1_reg, l2_reg=l2_reg
        )
        set_model(_trained_model)

        # Accumulate losses (or start fresh if stale)
        _all_losses = _prev_losses + _new_losses
        set_losses(_all_losses)
        set_trained(True)

        # Adjust snapshot epochs to account for previous training and save
        _epoch_offset = len(_prev_losses)
        _adjusted_snapshots = [(epoch + _epoch_offset, model) for epoch, model in _new_snapshots]
        if _epoch_offset > 0:
            # Prepend existing snapshots when continuing training
            _prev_snapshots = get_snapshots() or []
            set_snapshots(_prev_snapshots + _adjusted_snapshots)
        else:
            set_snapshots(_adjusted_snapshots)

        # Store training parameters and compute prediction for staleness tracking
        set_train_params(current_params.copy())
        _X_plot = jnp.linspace(-2, 2, 200)
        _y_pred = np.array(jax.vmap(_trained_model)(_X_plot))
        set_last_pred(_y_pred)

    # Handle store button - save current fit for comparison
    if store_button.value:
        _last_pred = get_last_pred()
        _losses = get_losses()
        if _last_pred is not None and _losses:
            _stored = get_stored_fits()
            _l1_str = f"L1={l1_slider.value}" if l1_slider.value > -6 else ""
            _l2_str = f"L2={l2_slider.value}" if l2_slider.value > -6 else ""
            _reg_str = ", ".join(filter(None, [_l1_str, _l2_str]))
            _label = f"Fit {len(_stored) + 1}: W={width}, D={depth}, {activation_name}"
            if _reg_str:
                _label += f", {_reg_str}"
            _new_fit = {
                'pred': _last_pred.copy(),
                'losses': list(_losses),
                'label': _label,
            }
            set_stored_fits(_stored + [_new_fit])

    # Handle clear stored button
    if clear_stored_button.value:
        set_stored_fits([])

    # Initialize model if needed
    if get_model() is None:
        set_model(MLP(key, width, depth, activation))

    return X_train, y_train, X_jax, y_jax, width, depth, activation, learning_rate, n_epochs, l1_reg, l2_reg, optimizer, key, current_params, train_min, train_max




@app.cell(hide_code=True)
def _(
    alt, pd, mo, jax, jnp, eqx, np,
    X_train, y_train, function_dropdown, current_params, train_min, train_max,
    get_ground_truth, get_model, get_losses, get_trained,
    get_train_params, get_last_pred, get_stored_fits,
    get_snapshots, get_selected_epoch,
    get_custom_code,
):
    # Prepare data for Altair plots
    X_plot = np.linspace(-2, 2, 200)
    _custom_code = get_custom_code() if function_dropdown.value == 'custom' else None
    y_gt = get_ground_truth(X_plot, function_dropdown.value, _custom_code)

    # Ground truth DataFrame
    gt_df = pd.DataFrame({'x': X_plot, 'y': y_gt, 'type': 'Ground truth'})

    # Training data DataFrame
    train_df = pd.DataFrame({'x': X_train, 'y': y_train})

    # Training region rectangle data
    train_region_df = pd.DataFrame({
        'x': [train_min], 'x2': [train_max],
        'y': [-1.5], 'y2': [1.5]
    })

    # Check staleness
    _train_params = get_train_params()
    _last_pred = get_last_pred()
    _is_stale = _train_params is not None and _train_params != current_params

    # Get model prediction at selected epoch
    _model = get_model()
    _snapshots = get_snapshots() or []
    _selected_epoch = get_selected_epoch()

    # Use max epoch if no selection
    _losses = get_losses()
    if _selected_epoch is None:
        _selected_epoch = len(_losses) if _losses else 1

    pred_df = pd.DataFrame(columns=['x', 'y', 'label'])
    stale_pred_df = pd.DataFrame(columns=['x', 'y'])

    if _model is not None and get_trained():
        if _is_stale and _last_pred is not None:
            stale_pred_df = pd.DataFrame({'x': X_plot, 'y': _last_pred})
        elif _snapshots:
            _nearest_snapshot = min(_snapshots, key=lambda s: abs(s[0] - _selected_epoch))
            _snapshot_epoch, _snapshot_model = _nearest_snapshot
            X_jax_plot = jnp.array(X_plot)
            _y_pred = np.array(jax.vmap(_snapshot_model)(X_jax_plot))
            pred_df = pd.DataFrame({
                'x': X_plot,
                'y': _y_pred,
                'label': f'MLP (epoch {_snapshot_epoch})'
            })
        else:
            X_jax_plot = jnp.array(X_plot)
            _y_pred = np.array(jax.vmap(_model)(X_jax_plot))
            pred_df = pd.DataFrame({
                'x': X_plot,
                'y': _y_pred,
                'label': 'Current fit'
            })

    # Stored fits data
    stored_fits_data = []
    _stored_fits = get_stored_fits()
    _stored_colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for _i, _fit in enumerate(_stored_fits):
        _color = _stored_colors[_i % len(_stored_colors)]
        for _x, _y in zip(X_plot, _fit['pred']):
            stored_fits_data.append({'x': _x, 'y': _y, 'label': _fit['label'], 'color': _color})
    stored_fits_df = pd.DataFrame(stored_fits_data) if stored_fits_data else pd.DataFrame(columns=['x', 'y', 'label', 'color'])

    # Weight norms data
    norm_data = []
    if _snapshots and not _is_stale:
        for _snap_epoch, _snap_model in _snapshots:
            _params = eqx.filter(_snap_model, eqx.is_array)
            _leaves = jax.tree_util.tree_leaves(_params)
            _l1 = float(sum(jnp.sum(jnp.abs(w)) for w in _leaves))
            _l2 = float(jnp.sqrt(sum(jnp.sum(w ** 2) for w in _leaves)))
            norm_data.append({'epoch': _snap_epoch, 'l1_norm': _l1, 'l2_norm': _l2})
    norm_df = pd.DataFrame(norm_data) if norm_data else pd.DataFrame(columns=['epoch', 'l1_norm', 'l2_norm'])

    # === Build Data Plot ===
    # Define shared scales for consistent axes
    x_scale = alt.Scale(domain=[-2.1, 2.1])
    y_scale = alt.Scale(domain=[-1.5, 1.5])

    # Training region shading
    region_layer = alt.Chart(train_region_df).mark_rect(
        color='#1f77b4', opacity=0.1
    ).encode(
        x=alt.X('x:Q', scale=x_scale), x2='x2:Q',
        y=alt.Y('y:Q', scale=y_scale), y2='y2:Q'
    )

    # Ground truth line
    gt_layer = alt.Chart(gt_df).mark_line(
        color='black', strokeWidth=3, opacity=0.7
    ).encode(
        x=alt.X('x:Q', scale=x_scale, title='x'),
        y=alt.Y('y:Q', scale=y_scale, title='y')
    )

    # Generated training data points
    train_layer = alt.Chart(train_df).mark_circle(
        color='#1f77b4', size=100, opacity=0.7
    ).encode(
        x=alt.X('x:Q', scale=x_scale),
        y=alt.Y('y:Q', scale=y_scale)
    )

    # Build layers list conditionally to avoid empty DataFrame issues
    layers = [region_layer, gt_layer, train_layer]

    # Stored fits
    if len(stored_fits_df) > 0:
        stored_layer = alt.Chart(stored_fits_df).mark_line(
            strokeWidth=2.5, opacity=0.8
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale),
            color=alt.Color('label:N', scale=alt.Scale(range=_stored_colors), legend=alt.Legend(title='Stored Fits'))
        )
        layers.append(stored_layer)

    # Stale prediction (grey)
    if len(stale_pred_df) > 0:
        stale_layer = alt.Chart(stale_pred_df).mark_line(
            color='lightgrey', strokeWidth=3
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale)
        )
        layers.append(stale_layer)

    # Current prediction (orange) - use color encoding to show epoch in legend
    if len(pred_df) > 0:
        pred_layer = alt.Chart(pred_df).mark_line(
            strokeWidth=3
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale),
            color=alt.Color('label:N', scale=alt.Scale(range=['#ff7f0e']), legend=alt.Legend(title='Current Fit'))
        )
        layers.append(pred_layer)

    # Combine all layers
    data_chart_spec = alt.layer(*layers).properties(
        width='container', height=250, title='Data and Model Fit'
    ).configure_axis(
        grid=True, gridOpacity=0.3,
        labelFontSize=14, titleFontSize=16
    ).configure_title(
        fontSize=18
    )

    interactive_data_chart = mo.ui.altair_chart(data_chart_spec)

    # Store computed values for other cells
    is_stale = _is_stale

    return (interactive_data_chart, gt_df, train_df, pred_df, stale_pred_df, stored_fits_df, norm_df, is_stale)


@app.cell(hide_code=True)
def _(interactive_data_chart):
    # Display the interactive data chart
    # Selection handling is done separately to avoid circular dependencies
    data_chart_output = interactive_data_chart
    return (data_chart_output,)


@app.cell(hide_code=True)
def _(
    alt, pd, mo, np,
    get_losses, get_stored_fits, get_snapshots, get_selected_epoch,
    norm_df, is_stale,
):
    # === Build Loss Plot with brush selection for epoch ===
    _losses = get_losses()
    _stored_fits = get_stored_fits()
    _stored_colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Current loss data
    if _losses:
        loss_df = pd.DataFrame({
            'epoch': np.arange(1, len(_losses) + 1),
            'loss': _losses
        })
    else:
        loss_df = pd.DataFrame(columns=['epoch', 'loss'])

    # Stored losses data
    stored_loss_data = []
    for _i, _fit in enumerate(_stored_fits):
        _color = _stored_colors[_i % len(_stored_colors)]
        for _epoch, _loss_val in enumerate(_fit['losses'], 1):
            stored_loss_data.append({'epoch': _epoch, 'loss': _loss_val, 'label': _fit['label'], 'color': _color})
    stored_loss_df = pd.DataFrame(stored_loss_data) if stored_loss_data else pd.DataFrame(columns=['epoch', 'loss', 'label', 'color'])

    # Epoch click selection - click on loss curve to select epoch
    epoch_click = alt.selection_point(on='click', nearest=True, fields=['epoch'], name='epoch_click')

    # Build loss chart layers conditionally
    loss_layers = []

    # Current loss line with click selection
    if len(loss_df) > 0:
        _loss_color = 'lightgrey' if is_stale else '#ff7f0e'
        # Line for display
        loss_line = alt.Chart(loss_df).mark_line(
            color=_loss_color, strokeWidth=2.5
        ).encode(
            x=alt.X('epoch:Q', title='Epoch'),
            y=alt.Y('loss:Q', title='MSE Loss', scale=alt.Scale(type='log'))
        )
        # Invisible points for click interaction
        loss_points = alt.Chart(loss_df).mark_point(
            size=100, opacity=0
        ).encode(
            x='epoch:Q',
            y='loss:Q'
        ).add_params(epoch_click)
        loss_layers.append(loss_line)
        loss_layers.append(loss_points)
    else:
        # Placeholder chart when no data
        placeholder_df = pd.DataFrame({'epoch': [1], 'loss': [1]})
        loss_line = alt.Chart(placeholder_df).mark_line(opacity=0).encode(
            x=alt.X('epoch:Q', title='Epoch'),
            y=alt.Y('loss:Q', title='MSE Loss', scale=alt.Scale(type='log'))
        ).add_params(epoch_click)
        loss_layers.append(loss_line)

    # Stored loss lines
    if len(stored_loss_df) > 0:
        stored_loss_layer = alt.Chart(stored_loss_df).mark_line(
            strokeWidth=2.5, opacity=0.8
        ).encode(
            x='epoch:Q', y='loss:Q',
            color=alt.Color('label:N', scale=alt.Scale(range=_stored_colors), legend=alt.Legend(title='Stored Fits'))
        )
        loss_layers.append(stored_loss_layer)

    # Build base chart
    combined_chart = alt.layer(*loss_layers)

    # Weight norm lines (independent y-axis via layering)
    if len(norm_df) > 0 and not is_stale:
        # Reshape for long format
        norm_long = pd.melt(norm_df, id_vars=['epoch'], value_vars=['l1_norm', 'l2_norm'],
                           var_name='norm_type', value_name='value')
        norm_lines = alt.Chart(norm_long).mark_line(
            strokeWidth=2, strokeDash=[2, 2]
        ).encode(
            x='epoch:Q',
            y=alt.Y('value:Q', scale=alt.Scale(type='log'), title='Weight Norm'),
            color=alt.Color('norm_type:N',
                           scale=alt.Scale(domain=['l1_norm', 'l2_norm'], range=['#2ca02c', '#9467bd']),
                           legend=alt.Legend(title='Norms'))
        )
        # Layer with independent y scales
        combined_chart = alt.layer(
            combined_chart,
            norm_lines
        ).resolve_scale(y='independent')

    # Add selected epoch marker if we have snapshots
    _snapshots = get_snapshots() or []
    _selected_epoch = get_selected_epoch()
    if _snapshots and not is_stale and _selected_epoch is not None:
        epoch_marker_df = pd.DataFrame({'epoch': [_selected_epoch]})
        epoch_marker = alt.Chart(epoch_marker_df).mark_rule(
            color='black', strokeWidth=2, strokeDash=[4, 4]
        ).encode(x='epoch:Q')
        combined_chart = alt.layer(combined_chart, epoch_marker)

    combined_chart = combined_chart.properties(
        width='container', height=200, title='Training Loss and Weight Norms'
    ).configure_axis(
        grid=True, gridOpacity=0.3,
        labelFontSize=14, titleFontSize=16
    ).configure_title(
        fontSize=18
    )

    interactive_loss_chart = mo.ui.altair_chart(combined_chart)

    return (interactive_loss_chart, loss_df, stored_loss_df)


@app.cell(hide_code=True)
def _(interactive_loss_chart, loss_df, get_losses, get_selected_epoch, set_selected_epoch):
    # Process click selection from loss chart to update selected epoch
    _losses = get_losses()
    _current_epoch = get_selected_epoch()

    # For click selection: apply_selection returns the clicked point(s)
    if len(loss_df) > 0:
        _filtered_df = interactive_loss_chart.apply_selection(loss_df)
        if len(_filtered_df) > 0 and len(_filtered_df) < len(loss_df):
            # User clicked on a point - get the epoch
            _new_epoch = int(_filtered_df['epoch'].iloc[0])
            if _new_epoch != _current_epoch:
                set_selected_epoch(_new_epoch)
        elif _current_epoch is None and _losses:
            # Initialize to max epoch only if not yet set
            set_selected_epoch(len(_losses))
    elif _current_epoch is None and _losses:
        # Initialize to max epoch only if not yet set
        set_selected_epoch(len(_losses))

    loss_chart_output = interactive_loss_chart
    return (loss_chart_output,)


@app.cell(hide_code=True)
def _(
    mo,
    n_points_slider, noise_slider, function_dropdown, seed_slider,
    width_slider, depth_slider, activation_dropdown,
    lr_slider, optimizer_dropdown, epochs_slider, l1_slider, l2_slider, train_range_slider,
    train_button, reset_button, store_button, clear_stored_button,
    custom_function_accordion,
):
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

        <h4>Training</h4>
        {lr_slider}
        {optimizer_dropdown}
        {epochs_slider}
        {l1_slider}
        {l2_slider}

        <div style="display: flex; gap: 0.5em; margin-top: 1em; flex-wrap: wrap;">
            {train_button}
            {reset_button}
            {store_button}
            {clear_stored_button}
        </div>

        <p style="font-size: 0.85em; color: #666; margin-top: 1em;">
            <b>Tip:</b> Click on loss plot to select epoch for visualization.
        </p>
    </div>
    ''')
    return (sidebar,)


@app.cell(hide_code=True)
def _(mo, jax, jnp, eqx, width, depth, X_train, get_snapshots, get_selected_epoch, get_losses):
    # Compute number of parameters:
    # - Input layer: Linear('scalar', width) = 1*width + width (bias) = 2*width
    # - Hidden layers: (depth-1) * Linear(width, width) = (depth-1) * width*(width+1)
    # - Output layer: Linear(width, 'scalar') = width + 1 (with bias)
    n_params = 2 * width + (depth - 1) * width * (width + 1) + width + 1
    n_train = len(X_train)
    ratio = n_train / n_params

    # Compute weight norms from model at selected epoch
    _snapshots = get_snapshots() or []
    _selected_epoch = get_selected_epoch()
    _losses = get_losses()
    if _selected_epoch is None:
        _selected_epoch = len(_losses) if _losses else 1

    if _snapshots:
        _nearest_snapshot = min(_snapshots, key=lambda s: abs(s[0] - _selected_epoch))
        _, _snapshot_model = _nearest_snapshot
        _params = eqx.filter(_snapshot_model, eqx.is_array)
        _leaves = jax.tree_util.tree_leaves(_params)
        _l1_norm = float(sum(jnp.sum(jnp.abs(w)) for w in _leaves))
        _l2_norm = float(jnp.sqrt(sum(jnp.sum(w ** 2) for w in _leaves)))
        _norm_str = f" | ‖w‖₁={_l1_norm:.2f} | ‖w‖₂={_l2_norm:.2f}"
    else:
        _norm_str = ""

    param_info = mo.Html(f'''
    <div style="text-align: center; padding: 0.5em; color: #666; font-size: 14px;">
        Training points N={n_train} | Parameters P={n_params:,} | N/P={ratio:.2f}{_norm_str}
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
