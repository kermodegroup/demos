# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "numpy==2.2.5",
#     "seaborn==0.13.2",
#     "qrcode[pil]==8.2",
#     "jax",
#     "jaxlib",
#     "equinox",
#     "optax",
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
            justify-content: center;
            align-items: flex-start;
        }

        .app-plot img,
        .app-plot svg {
            max-width: 100%;
            height: auto;
        }

        .app-sidebar {
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
    import matplotlib.pyplot as plt

    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax

    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'cpu')

    import seaborn as sns
    sns.set_context('talk')
    return eqx, jax, jnp, mo, np, optax, plt, sns


@app.cell(hide_code=True)
def _():
    import qrcode
    import io
    import base64

    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://sciml.warwick.ac.uk/live/neural-ode-demo/')
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
                <p style='font-size: 24px; margin: 0; padding: 0; line-height: 1.3;'><b>Neural ODE Demo</b>
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
    # ODE selection
    ode_dropdown = mo.ui.dropdown(
        options={
            'Exponential Decay': 'exp_decay',
            'Damped Oscillator': 'damped_osc',
            'Linear Growth': 'linear_growth',
        },
        value='Exponential Decay',
        label='ODE'
    )

    # ODE parameters
    decay_rate_slider = mo.ui.slider(0.1, 3.0, 0.1, 1.0, label='Decay rate (k)')
    growth_rate_slider = mo.ui.slider(0.1, 2.0, 0.1, 1.0, label='Growth rate (k)')
    frequency_slider = mo.ui.slider(1, 20, 1, 10, label='Frequency (ω)')
    damping_slider = mo.ui.slider(0.0, 1.0, 0.05, 0.4, label='Damping (γ)')

    # Network controls
    width_slider = mo.ui.slider(10, 100, 10, 50, label='Network width')
    depth_slider = mo.ui.slider(1, 5, 1, 2, label='Depth (hidden layers)')
    activation_dropdown = mo.ui.dropdown(
        options={'Sigmoid': 'sigmoid', 'Tanh': 'tanh', 'ReLU': 'relu'},
        value='Sigmoid',
        label='Activation function'
    )

    # Training controls
    lr_slider = mo.ui.slider(-4, -1, 0.5, -2, label='Learning rate (10^x)')
    collocation_slider = mo.ui.slider(10, 200, 10, 50, label='Training points')
    epochs_slider = mo.ui.slider(100, 10000, 100, 2000, label='Number of epochs')
    x_max_slider = mo.ui.slider(1, 5, 0.5, 2, label='Domain end (x_max)')
    train_range_slider = mo.ui.range_slider(
        start=0, stop=5, step=0.1,
        value=[0.0, 1.0],
        label='Training range'
    )
    resample_checkbox = mo.ui.checkbox(value=False, label='Resample points each epoch')

    # Action buttons
    train_button = mo.ui.run_button(label='Train')
    reset_button = mo.ui.run_button(label='Reset')
    store_button = mo.ui.run_button(label='Store Fit')
    clear_stored_button = mo.ui.run_button(label='Clear Stored')

    return (
        ode_dropdown,
        decay_rate_slider,
        growth_rate_slider,
        frequency_slider,
        damping_slider,
        width_slider,
        depth_slider,
        activation_dropdown,
        lr_slider,
        collocation_slider,
        epochs_slider,
        x_max_slider,
        train_range_slider,
        resample_checkbox,
        train_button,
        reset_button,
        store_button,
        clear_stored_button,
    )


@app.cell(hide_code=True)
def _(jnp):
    # ODE definitions: (f_ode, y0, analytic_solution, ode_label)
    def get_ode_system(ode_type, decay_rate=1.0, growth_rate=1.0, frequency=10.0, damping=0.4):
        """Return ODE function, initial condition, analytic solution, and label.

        Parameters:
        -----------
        ode_type : str
            Type of ODE ('exp_decay', 'damped_osc', 'linear_growth')
        decay_rate : float
            Decay rate k for exponential decay: dy/dx = -k*y
        growth_rate : float
            Growth rate k for linear growth: dy/dx = k*y
        frequency : float
            Angular frequency ω for damped oscillator
        damping : float
            Damping constant γ for damped oscillator
        """
        if ode_type == 'exp_decay':
            # dy/dx = -k*y, y(0) = 1, solution: e^(-k*x)
            k = decay_rate
            def f_ode(x, y):
                return -k * y
            y0 = 1.0
            def analytic(x):
                return jnp.exp(-k * x)
            label = f"dy/dx = -{k:.1f}y"
        elif ode_type == 'damped_osc':
            # Damped oscillator: y = e^(-γx) * sin(ωx)
            # Derivative: dy/dx = e^(-γx) * (ω*cos(ωx) - γ*sin(ωx))
            #                   = ω*e^(-γx)*cos(ωx) - γ*y
            # So f(x, y) = ω*e^(-γx)*cos(ωx) - γ*y
            omega = frequency
            gamma = damping
            def f_ode(x, y):
                return omega * jnp.exp(-gamma * x) * jnp.cos(omega * x) - gamma * y
            y0 = 0.0
            def analytic(x):
                return jnp.exp(-gamma * x) * jnp.sin(omega * x)
            label = f"dy/dx = {omega:.1f}e^(-{gamma:.1f}x)cos({omega:.1f}x) - {gamma:.1f}y"
        elif ode_type == 'linear_growth':
            # dy/dx = k*y, y(0) = 1, solution: e^(k*x)
            k = growth_rate
            def f_ode(x, y):
                return k * y
            y0 = 1.0
            def analytic(x):
                return jnp.exp(k * x)
            label = f"dy/dx = {k:.1f}y"
        else:
            # Default to exponential decay
            k = decay_rate
            def f_ode(x, y):
                return -k * y
            y0 = 1.0
            def analytic(x):
                return jnp.exp(-k * x)
            label = f"dy/dx = -{k:.1f}y"

        return f_ode, y0, analytic, label

    return (get_ode_system,)


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
            return jax.nn.sigmoid

    class NeuralODE(eqx.Module):
        """Neural network for solving ODEs using trial solution method.

        Trial solution: y_t(x) = y0 + x * g(x)
        This automatically satisfies y_t(0) = y0 (initial condition).
        """
        layers: list
        activation: callable
        y0: float = eqx.field(static=True)  # Static so it won't be updated during training

        def __init__(self, key, width, depth, activation, y0):
            keys = jax.random.split(key, depth + 1)
            layers = []

            # Input layer: scalar -> width
            layers.append(eqx.nn.Linear('scalar', width, key=keys[0]))

            # Hidden layers
            for i in range(depth - 1):
                layers.append(eqx.nn.Linear(width, width, key=keys[i + 1]))

            # Output layer: width -> scalar (no bias per Lagaris et al.)
            layers.append(eqx.nn.Linear(width, 'scalar', use_bias=False, key=keys[-1]))

            self.layers = layers
            self.activation = activation
            self.y0 = float(y0)  # Store as Python float (static field)

        def network(self, x):
            """Neural network g(x) - the learned shape function."""
            h = x
            for layer in self.layers[:-1]:
                h = self.activation(layer(h))
            return self.layers[-1](h)

        def __call__(self, x):
            """Trial solution: y_t(x) = y0 + x * g(x)"""
            g = self.network(x)
            return self.y0 + x * g

    return get_activation, NeuralODE


@app.cell(hide_code=True)
def _(eqx, jax, jnp):
    def make_loss_fn(f_ode):
        """Create physics-based loss function for ODE."""
        def loss_fn(model, x_colloc):
            def residual(x):
                # Trial solution: y_t(x) = y0 + x * g(x)
                # Derivative: dy_t/dx = g(x) + x * dg/dx
                g = model.network(x)
                dg_dx = jax.grad(lambda t: model.network(t))(x)
                dy_dx = g + x * dg_dx

                # Compute y for the ODE right-hand side
                y = model.y0 + x * g

                # Physics residual: dy/dx should equal f(x, y)
                return dy_dx - f_ode(x, y)

            residuals = jax.vmap(residual)(x_colloc)
            return jnp.mean(residuals ** 2)
        return loss_fn

    @eqx.filter_jit
    def train_step(model, opt_state, optimizer, loss_fn, x_colloc):
        """Single training step with gradient update."""
        def batch_loss(model):
            return loss_fn(model, x_colloc)

        loss, grads = eqx.filter_value_and_grad(batch_loss)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    def train_model(model, optimizer, loss_fn, x_colloc, n_epochs, resample=False, x_min=None, x_max=None, n_points=None, key=None):
        """Train model for n_epochs and return loss history and snapshots.

        If resample=True, generate new random training points each epoch.
        Returns snapshots at every 1% of training for visualization.
        """
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        losses = []
        snapshots = []  # List of (epoch, model) tuples
        snapshot_interval = max(1, n_epochs // 100)  # Save every 1%

        for i in range(n_epochs):
            if resample and x_min is not None and x_max is not None and n_points is not None and key is not None:
                # Generate new random points each epoch
                key, subkey = jax.random.split(key)
                # Use uniform random sampling, but ensure we don't include x=0
                x_train = jax.random.uniform(subkey, (n_points,), minval=max(1e-6, x_min), maxval=x_max)
            else:
                x_train = x_colloc

            model, opt_state, loss = train_step(model, opt_state, optimizer, loss_fn, x_train)
            losses.append(float(loss))

            # Save snapshot at intervals and at the final epoch
            if i % snapshot_interval == 0 or i == n_epochs - 1:
                snapshots.append((i + 1, model))  # 1-indexed epoch

        return model, losses, snapshots

    return make_loss_fn, train_step, train_model


@app.cell(hide_code=True)
def _(mo):
    # State for model and training results
    get_model, set_model = mo.state(None)
    get_losses, set_losses = mo.state([])
    get_trained, set_trained = mo.state(False)
    # State for tracking if fit is stale (parameters changed since last train)
    get_train_params, set_train_params = mo.state(None)
    get_last_pred, set_last_pred = mo.state(None)
    # State for stored fits (list of dicts with 'pred', 'losses', 'label', 'x_plot')
    get_stored_fits, set_stored_fits = mo.state([])
    # State for model snapshots during training
    get_snapshots, set_snapshots = mo.state([])
    return (get_model, set_model, get_losses, set_losses, get_trained, set_trained,
            get_train_params, set_train_params, get_last_pred, set_last_pred,
            get_snapshots, set_snapshots,
            get_stored_fits, set_stored_fits)


@app.cell(hide_code=True)
def _(mo, get_losses):
    # Epoch slider for visualizing training progress
    _losses = get_losses() or []
    _max_epoch = max(1, len(_losses))
    epoch_slider = mo.ui.slider(
        start=1, stop=_max_epoch, step=max(1, _max_epoch // 100),
        value=_max_epoch,
        label='Display epoch'
    )
    return (epoch_slider,)


@app.cell(hide_code=True)
def _(
    jax, jnp, np, optax,
    ode_dropdown, decay_rate_slider, growth_rate_slider, frequency_slider, damping_slider,
    width_slider, depth_slider, activation_dropdown,
    lr_slider, collocation_slider, epochs_slider, x_max_slider,
    train_range_slider, resample_checkbox,
    train_button, reset_button, store_button, clear_stored_button,
    get_ode_system, get_activation, NeuralODE, make_loss_fn, train_model,
    get_model, set_model, get_losses, set_losses, get_trained, set_trained,
    get_train_params, set_train_params, get_last_pred, set_last_pred,
    get_snapshots, set_snapshots,
    get_stored_fits, set_stored_fits,
):
    # Get ODE system with parameters
    ode_type = ode_dropdown.value
    x_max = x_max_slider.value
    decay_rate = decay_rate_slider.value
    growth_rate = growth_rate_slider.value
    frequency = frequency_slider.value
    damping = damping_slider.value
    f_ode, y0, analytic, ode_label = get_ode_system(
        ode_type, decay_rate=decay_rate, growth_rate=growth_rate,
        frequency=frequency, damping=damping
    )

    # Get network parameters
    width = width_slider.value
    depth = depth_slider.value
    activation_name = activation_dropdown.value
    activation = get_activation(activation_name)
    learning_rate = 10 ** lr_slider.value
    n_colloc = collocation_slider.value
    n_epochs = epochs_slider.value

    # Get training range (clamped to domain)
    train_min = max(1e-6, min(train_range_slider.value[0], x_max - 0.1))
    train_max = min(train_range_slider.value[1], x_max)
    resample = resample_checkbox.value

    # Current parameters dict for staleness tracking
    current_params = {
        'ode': ode_type,
        'x_max': x_max,
        'decay_rate': decay_rate,
        'growth_rate': growth_rate,
        'frequency': frequency,
        'damping': damping,
        'width': width,
        'depth': depth,
        'activation': activation_name,
        'lr': lr_slider.value,
        'n_colloc': n_colloc,
        'epochs': n_epochs,
        'train_min': train_min,
        'train_max': train_max,
        'resample': resample,
    }

    # Create optimizer
    optimizer = optax.adam(learning_rate)

    # Create collocation points (fixed grid for non-resample mode)
    x_colloc = jnp.linspace(train_min, train_max, n_colloc)

    # Initialize key
    key = jax.random.PRNGKey(42)

    # Handle reset button
    if reset_button.value:
        _model = NeuralODE(key, width, depth, activation, y0)
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

        # Check if model's y0 matches current ODE's y0
        _existing_model = get_model()
        _y0_mismatch = _existing_model is not None and float(_existing_model.y0) != float(y0)

        if _is_stale or _train_params is None or _y0_mismatch:
            # Parameters changed, first train, or y0 mismatch - start fresh with new model
            _current_model = NeuralODE(key, width, depth, activation, y0)
            _prev_losses = []
        else:
            # Continue training existing model
            _current_model = _existing_model
            _prev_losses = get_losses() or []

        # Create loss function for this ODE
        _loss_fn = make_loss_fn(f_ode)

        # Train model (with optional resampling)
        _train_key = jax.random.PRNGKey(len(_prev_losses) + 1)  # Different key each training session
        _trained_model, _new_losses, _new_snapshots = train_model(
            _current_model, optimizer, _loss_fn, x_colloc, n_epochs,
            resample=resample, x_min=train_min, x_max=train_max, n_points=n_colloc, key=_train_key
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
        _X_plot = jnp.linspace(0, x_max, 200)
        _y_pred = np.array(jax.vmap(_trained_model)(_X_plot))
        set_last_pred((_X_plot, _y_pred))

    # Handle store button - save current fit for comparison
    if store_button.value:
        _last_pred = get_last_pred()
        _losses = get_losses()
        if _last_pred is not None and _losses:
            _stored = get_stored_fits()
            _label = f"Fit {len(_stored) + 1}: W={width}, D={depth}, {activation_name}"
            _new_fit = {
                'x_plot': np.array(_last_pred[0]),
                'pred': np.array(_last_pred[1]),
                'losses': list(_losses),
                'label': _label,
                'train_range': (train_min, train_max),
            }
            set_stored_fits(_stored + [_new_fit])

    # Handle clear stored button
    if clear_stored_button.value:
        set_stored_fits([])

    # Initialize model if needed, or reinitialize if y0 changed
    _init_model = get_model()
    if _init_model is None or float(_init_model.y0) != float(y0):
        set_model(NeuralODE(key, width, depth, activation, y0))

    return (ode_type, x_max, f_ode, y0, analytic, ode_label, width, depth, activation, learning_rate,
            n_colloc, n_epochs, optimizer, x_colloc, key, current_params, train_min, train_max, resample)


@app.cell(hide_code=True)
def _(
    jax, jnp, np, plt,
    ode_type, x_max, y0, analytic, ode_label, x_colloc, current_params, train_min, train_max, resample,
    get_model, get_losses, get_trained,
    get_train_params, get_last_pred, get_stored_fits,
    get_snapshots, epoch_slider,
):
    # Create figure with two vertically stacked subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.8))

    # Top plot: Solution
    X_plot = np.linspace(0, x_max, 200)
    y_analytic = np.array(jax.vmap(analytic)(jnp.array(X_plot)))

    # Plot analytic solution
    ax1.plot(X_plot, y_analytic, 'k--', lw=2, label='Analytic solution', alpha=0.8)

    # Shade training region
    ax1.axvspan(train_min, train_max, alpha=0.1, color='C0', label='Training region')

    # Plot training points on x-axis (show fixed grid if not resampling)
    if not resample:
        ax1.scatter(np.array(x_colloc), np.zeros_like(np.array(x_colloc)),
                    c='C0', s=30, alpha=0.6, marker='|', label='Training points', zorder=5)
    else:
        # Just show range markers for resample mode
        ax1.axvline(train_min, color='C0', ls=':', alpha=0.5)
        ax1.axvline(train_max, color='C0', ls=':', alpha=0.5)

    # Plot initial condition
    ax1.scatter([0], [y0], c='red', s=80, marker='o', label=f'y(0) = {y0}', zorder=10)

    # Plot stored fits first (so current fit appears on top)
    _stored_fits = get_stored_fits()
    _stored_colors = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for _i, _fit in enumerate(_stored_fits):
        _color = _stored_colors[_i % len(_stored_colors)]
        ax1.plot(_fit['x_plot'], _fit['pred'], color=_color, lw=1.5, alpha=0.8,
                 label=_fit['label'], zorder=4)
        # Plot stored loss curves with same color and label
        _stored_epochs = np.arange(1, len(_fit['losses']) + 1)
        ax2.semilogy(_stored_epochs, _fit['losses'], color=_color, lw=1.5, alpha=0.8,
                     label=_fit['label'])

    # Check if fit is stale (parameters changed since last training)
    _train_params = get_train_params()
    _last_pred = get_last_pred()
    _is_stale = _train_params is not None and _train_params != current_params

    # Plot Neural ODE prediction if trained
    _model = get_model()
    _snapshots = get_snapshots() or []
    _selected_epoch = epoch_slider.value

    if _model is not None and get_trained():
        if _is_stale and _last_pred is not None:
            # Show stale prediction in light grey
            ax1.plot(_last_pred[0], _last_pred[1], color='lightgrey', lw=2,
                     label='Current (stale)', zorder=3)
        elif _snapshots:
            # Find nearest snapshot to selected epoch
            _nearest_snapshot = min(_snapshots, key=lambda s: abs(s[0] - _selected_epoch))
            _snapshot_epoch, _snapshot_model = _nearest_snapshot
            X_jax_plot = jnp.array(X_plot)
            _y_pred = jax.vmap(_snapshot_model)(X_jax_plot)
            ax1.plot(X_plot, np.array(_y_pred), 'C1-', lw=2,
                     label=f'Neural ODE (epoch {_snapshot_epoch})')
        else:
            # No snapshots yet, show current model
            X_jax_plot = jnp.array(X_plot)
            _y_pred = jax.vmap(_model)(X_jax_plot)
            ax1.plot(X_plot, np.array(_y_pred), 'C1-', lw=2, label='Neural ODE')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim(-0.1, x_max + 0.1)

    # Set y limits based on ODE type
    if ode_type == 'linear_growth':
        ax1.set_ylim(-0.5, np.exp(x_max) * 1.1)
    elif ode_type == 'damped_osc':
        ax1.set_ylim(-1.2, 1.2)
    else:
        ax1.set_ylim(-0.2, 1.5)

    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_title(f'ODE Solution: {ode_label}')
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Loss curve
    _losses = get_losses()
    if _losses or _stored_fits:
        if _losses:
            _epochs = np.arange(1, len(_losses) + 1)
            if _is_stale:
                # Show stale loss curve in light grey
                ax2.semilogy(_epochs, _losses, color='lightgrey', lw=1.5, label='Current (stale)')
            else:
                ax2.semilogy(_epochs, _losses, 'C1-', lw=1.5, label='Current fit')
            # Add vertical line at selected epoch
            if _snapshots and not _is_stale:
                _nearest_snapshot = min(_snapshots, key=lambda s: abs(s[0] - _selected_epoch))
                _snapshot_epoch = _nearest_snapshot[0]
                ax2.axvline(_snapshot_epoch, color='black', lw=3, ls='--')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Physics Loss')
        ax2.set_title('Training Loss (ODE Residual)')
        ax2.grid(True, alpha=0.3, which='both')
        # Set x-axis limit to max epochs across all fits
        _max_epochs = max([len(_losses)] if _losses else [0] +
                         [len(f['losses']) for f in _stored_fits])
        if _max_epochs > 0:
            ax2.set_xlim(1, _max_epochs)
        if _stored_fits or _losses:
            ax2.legend(loc='upper right', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'Click "Train" to start',
                 ha='center', va='center', transform=ax2.transAxes,
                 fontsize=14, color='gray')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Physics Loss')
        ax2.set_title('Training Loss (ODE Residual)')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_output = fig
    return (plot_output,)


@app.cell(hide_code=True)
def _(
    mo, ode_type,
    ode_dropdown, decay_rate_slider, growth_rate_slider, frequency_slider, damping_slider,
    width_slider, depth_slider, activation_dropdown,
    lr_slider, collocation_slider, epochs_slider, x_max_slider,
    train_range_slider, resample_checkbox, epoch_slider,
    train_button, reset_button, store_button, clear_stored_button,
):
    # Show ODE-specific parameter controls
    if ode_type == 'exp_decay':
        ode_params_html = f'{decay_rate_slider}'
    elif ode_type == 'damped_osc':
        ode_params_html = f'{frequency_slider}{damping_slider}'
    elif ode_type == 'linear_growth':
        ode_params_html = f'{growth_rate_slider}'
    else:
        ode_params_html = ''

    sidebar = mo.Html(f'''
    <div class="app-sidebar">
        <h4>ODE Selection</h4>
        {ode_dropdown}
        {ode_params_html}

        <h4>Network</h4>
        {width_slider}
        {depth_slider}
        {activation_dropdown}

        <h4>Training</h4>
        {lr_slider}
        {collocation_slider}
        {epochs_slider}
        {x_max_slider}
        {train_range_slider}
        {resample_checkbox}

        <div style="display: flex; gap: 0.5em; margin-top: 1em; flex-wrap: wrap;">
            {train_button}
            {reset_button}
            {store_button}
            {clear_stored_button}
        </div>

        <h4>Visualization</h4>
        {epoch_slider}
    </div>
    ''')
    return (sidebar,)


@app.cell(hide_code=True)
def _(mo, width, depth, n_colloc):
    # Compute number of parameters:
    # - Input layer: Linear('scalar', width) = 1*width + width (bias) = 2*width
    # - Hidden layers: (depth-1) * Linear(width, width) = (depth-1) * width*(width+1)
    # - Output layer: Linear(width, 'scalar', use_bias=False) = width
    n_params = 2 * width + (depth - 1) * width * (width + 1) + width
    ratio = n_colloc / n_params
    param_info = mo.Html(f'''
    <div style="text-align: center; padding: 0.5em; color: #666; font-size: 14px;">
        Training points N={n_colloc} | Parameters P={n_params:,} | N/P={ratio:.2f}
    </div>
    ''')
    return (param_info,)


@app.cell(hide_code=True)
def _(mo, header, plot_output, sidebar, param_info):
    mo.vstack([
        header,
        mo.Html(f'''
        <div class="app-layout">
            <div class="app-plot">
                <div style="display: flex; flex-direction: column; align-items: center;">
                    {mo.as_html(plot_output)}
                    {param_info}
                </div>
            </div>
            {sidebar}
        </div>
        '''),
    ])
    return


if __name__ == "__main__":
    app.run()
