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
#     "diffrax",
#     "altair==5.5.0",
#     "pandas",
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
    import pandas as pd

    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax

    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'cpu')

    import altair as alt
    return alt, eqx, jax, jnp, mo, np, optax, pd


@app.cell(hide_code=True)
def _():
    import qrcode
    import io
    import base64

    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://sciml.warwick.ac.uk/live/normalising-flows-demo/')
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
                <p style='font-size: 24px; margin: 0; padding: 0; line-height: 1.3;'><b>Normalising Flows Demo</b>
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
def _(jax, jnp):
    def make_target(name):
        """Return (means, stds, weights) for a Gaussian mixture target distribution."""
        if name == 'Bimodal':
            means = jnp.array([-2.0, 2.0])
            stds = jnp.array([0.5, 0.5])
            weights = jnp.array([0.5, 0.5])
        elif name == 'Trimodal':
            means = jnp.array([-3.0, 0.0, 3.0])
            stds = jnp.array([0.4, 0.4, 0.4])
            weights = jnp.array([0.33, 0.34, 0.33])
        elif name == 'Skewed':
            means = jnp.array([-1.0, 2.0])
            stds = jnp.array([0.4, 0.8])
            weights = jnp.array([0.7, 0.3])
        elif name == 'Comb':
            means = jnp.array([-3.0, -1.5, 0.0, 1.5, 3.0])
            stds = jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])
            weights = jnp.array([0.2, 0.2, 0.2, 0.2, 0.2])
        else:
            means = jnp.array([-2.0, 2.0])
            stds = jnp.array([0.5, 0.5])
            weights = jnp.array([0.5, 0.5])
        return means, stds, weights

    def target_log_prob(z, means, stds, weights):
        """Log probability of Gaussian mixture."""
        log_components = -0.5 * ((z - means) / stds) ** 2 - jnp.log(stds) - 0.5 * jnp.log(2 * jnp.pi)
        return jnp.log(jnp.sum(weights * jnp.exp(log_components)))

    def target_prob(z, means, stds, weights):
        """Probability density of Gaussian mixture."""
        components = weights / (stds * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * ((z - means) / stds) ** 2)
        return jnp.sum(components)

    def sample_target(key, n, means, stds, weights):
        """Sample from Gaussian mixture."""
        k1, k2 = jax.random.split(key)
        # Choose component
        indices = jax.random.choice(k1, len(means), shape=(n,), p=weights)
        # Sample from chosen component
        normals = jax.random.normal(k2, shape=(n,))
        return means[indices] + stds[indices] * normals

    return make_target, target_log_prob, target_prob, sample_target


@app.cell(hide_code=True)
def _(eqx, jax, jnp):
    class FlowNetwork(eqx.Module):
        """Time-conditioned MLP for continuous normalising flow velocity field.

        Takes (z, t) as input, outputs scalar velocity dz/dt.
        """
        layers: list

        def __init__(self, key, width, depth):
            keys = jax.random.split(key, depth + 1)
            layers = [eqx.nn.Linear(2, width, key=keys[0])]
            for i in range(depth - 1):
                layers.append(eqx.nn.Linear(width, width, key=keys[i + 1]))
            layers.append(eqx.nn.Linear(width, 1, key=keys[-1]))
            self.layers = layers

        def __call__(self, z, t):
            h = jnp.array([z, t])
            for layer in self.layers[:-1]:
                h = jax.nn.tanh(layer(h))
            return self.layers[-1](h).squeeze()

    return (FlowNetwork,)


@app.cell(hide_code=True)
def _(eqx, jax, jnp):
    def compute_log_prob(model, x, T, n_steps):
        """Compute log p(x) by integrating backwards from t=T to t=0.

        Solves the augmented ODE:
            dz/dt = f(z, t)
            d(log p)/dt = -df/dz
        backwards in time, accumulating the log-determinant.
        """
        dt = T / n_steps
        z = x
        log_det = 0.0
        t = T
        for _ in range(n_steps):
            # df/dz at current (z, t)
            dfz = jax.grad(lambda _z: model(_z, t))(z)
            # Backward Euler step
            z = z - dt * model(z, t)
            # Accumulate log-det: d(log p)/dt = -df/dz, integrated backward
            log_det = log_det - dt * dfz
            t = t - dt
        # Base distribution log prob: standard normal
        log_p_base = -0.5 * z ** 2 - 0.5 * jnp.log(2 * jnp.pi)
        return log_p_base + log_det

    def flow_forward(model, z0, T, n_steps):
        """Integrate forward from t=0 to t=T, returning full trajectory."""
        dt = T / n_steps
        trajectory = [z0]
        z = z0
        for k in range(n_steps):
            t = k * dt
            z = z + dt * model(z, t)
            trajectory.append(z)
        return jnp.stack(trajectory)

    def make_loss_fn(target_means, target_stds, target_weights, T, n_steps):
        """Create NLL loss function for training."""
        def loss_fn(model, x_batch):
            def neg_log_prob(x):
                return -compute_log_prob(model, x, T, n_steps)
            return jnp.mean(jax.vmap(neg_log_prob)(x_batch))
        return loss_fn

    @eqx.filter_jit
    def train_step(model, opt_state, optimizer, loss_fn, x_batch):
        """Single training step."""
        def batch_loss(model):
            return loss_fn(model, x_batch)
        loss, grads = eqx.filter_value_and_grad(batch_loss)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    def train_model(model, optimizer, loss_fn, x_train, n_epochs):
        """Train for n_epochs, returning loss history and snapshots."""
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        losses = []
        snapshots = []
        snapshot_interval = max(1, n_epochs // 100)

        for i in range(n_epochs):
            model, opt_state, loss = train_step(model, opt_state, optimizer, loss_fn, x_train)
            losses.append(float(loss))

            if i % snapshot_interval == 0 or i == n_epochs - 1:
                snapshots.append((i + 1, model))

        return model, losses, snapshots

    return compute_log_prob, flow_forward, make_loss_fn, train_step, train_model


@app.cell(hide_code=True)
def _(mo):
    # UI controls
    target_dropdown = mo.ui.dropdown(
        options=['Bimodal', 'Trimodal', 'Skewed', 'Comb'],
        value='Bimodal',
        label='Target distribution'
    )
    width_slider = mo.ui.slider(16, 128, 16, 32, label='Network width')
    depth_slider = mo.ui.slider(1, 4, 1, 2, label='Network depth')
    flow_T_slider = mo.ui.slider(0.5, 2.0, 0.1, 1.0, label='Flow time T')
    ode_steps_slider = mo.ui.slider(5, 50, 5, 10, label='ODE steps (Euler)')
    lr_slider = mo.ui.slider(-4, -1, 0.5, -3, label='Learning rate (10^x)')
    n_samples_slider = mo.ui.slider(100, 2000, 100, 200, label='Training samples')
    epochs_slider = mo.ui.slider(200, 10000, 200, 400, label='Training epochs')

    # Regularization controls
    weight_decay_slider = mo.ui.slider(-5, -2, 0.5, -4, label='Weight decay (10^x)')
    grad_clip_slider = mo.ui.slider(0.5, 10.0, 0.5, 1.0, label='Grad clip norm')
    cosine_schedule_checkbox = mo.ui.checkbox(value=True, label='Cosine LR schedule')

    train_button = mo.ui.run_button(label='Train')
    reset_button = mo.ui.run_button(label='Reset')
    return (
        target_dropdown, width_slider, depth_slider, flow_T_slider,
        ode_steps_slider, lr_slider, n_samples_slider, epochs_slider,
        weight_decay_slider, grad_clip_slider, cosine_schedule_checkbox,
        train_button, reset_button,
    )


@app.cell(hide_code=True)
def _(mo):
    # State
    get_model, set_model = mo.state(None)
    get_losses, set_losses = mo.state([])
    get_trained, set_trained = mo.state(False)
    get_snapshots, set_snapshots = mo.state([])
    get_train_params, set_train_params = mo.state(None)
    get_train_status, set_train_status = mo.state('')
    return (
        get_model, set_model, get_losses, set_losses,
        get_trained, set_trained, get_snapshots, set_snapshots,
        get_train_params, set_train_params,
        get_train_status, set_train_status,
    )


@app.cell(hide_code=True)
def _(mo, get_losses):
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
    eqx, jax, jnp, optax,
    target_dropdown, width_slider, depth_slider, flow_T_slider,
    ode_steps_slider, lr_slider, n_samples_slider, epochs_slider,
    weight_decay_slider, grad_clip_slider, cosine_schedule_checkbox,
    train_button, reset_button,
    FlowNetwork, make_target, sample_target, make_loss_fn, train_model,
    get_model, set_model, get_losses, set_losses,
    get_trained, set_trained, get_snapshots, set_snapshots,
    get_train_params, set_train_params,
    set_train_status,
):
    # Read control values
    target_name = target_dropdown.value
    width = width_slider.value
    depth = depth_slider.value
    T = flow_T_slider.value
    n_steps = ode_steps_slider.value
    learning_rate = 10 ** lr_slider.value
    n_samples = n_samples_slider.value
    n_epochs = epochs_slider.value
    weight_decay = 10 ** weight_decay_slider.value
    grad_clip_norm = grad_clip_slider.value
    use_cosine = cosine_schedule_checkbox.value

    current_params = {
        'target': target_name, 'width': width, 'depth': depth,
        'T': T, 'n_steps': n_steps, 'lr': lr_slider.value,
        'n_samples': n_samples, 'epochs': n_epochs,
        'weight_decay': weight_decay_slider.value,
        'grad_clip': grad_clip_norm, 'cosine': use_cosine,
    }

    target_means, target_stds, target_weights = make_target(target_name)
    key = jax.random.PRNGKey(42)

    # Build optimizer: grad clip → AdamW, with optional cosine schedule
    def _build_optimizer(n_steps_total):
        if use_cosine:
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=min(100, n_steps_total // 10),
                decay_steps=n_steps_total,
                end_value=learning_rate * 0.01,
            )
        else:
            schedule = learning_rate
        return optax.chain(
            optax.clip_by_global_norm(grad_clip_norm),
            optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
        )

    # Handle reset
    if reset_button.value:
        _model = FlowNetwork(key, width, depth)
        set_model(_model)
        set_losses([])
        set_trained(False)
        set_snapshots([])
        set_train_params(None)
        set_train_status('')

    # Handle train
    if train_button.value:
        _train_params = get_train_params()
        _is_stale = _train_params is not None and _train_params != current_params

        if _is_stale or _train_params is None:
            _current_model = FlowNetwork(key, width, depth)
            _prev_losses = []
        else:
            _current_model = get_model()
            _prev_losses = get_losses() or []

        # Sample training data from target
        _data_key = jax.random.PRNGKey(len(_prev_losses) + 7)
        _x_train = sample_target(_data_key, n_samples, target_means, target_stds, target_weights)

        # Create loss and train
        import time as _time
        _loss_fn = make_loss_fn(target_means, target_stds, target_weights, T, n_steps)
        optimizer = _build_optimizer(n_epochs)
        _t0 = _time.time()
        _trained_model, _new_losses, _new_snapshots = train_model(
            _current_model, optimizer, _loss_fn, _x_train, n_epochs,
        )
        _elapsed = _time.time() - _t0

        set_model(_trained_model)
        _all_losses = _prev_losses + _new_losses
        set_losses(_all_losses)
        set_trained(True)

        _epoch_offset = len(_prev_losses)
        _adjusted = [(ep + _epoch_offset, m) for ep, m in _new_snapshots]
        if _epoch_offset > 0:
            _prev_snaps = get_snapshots() or []
            set_snapshots(_prev_snaps + _adjusted)
        else:
            set_snapshots(_adjusted)
        set_train_params(current_params.copy())
        _n_params = sum(p.size for p in jax.tree.leaves(eqx.filter(_trained_model, eqx.is_array)))
        set_train_status(f'{n_epochs} epochs in {_elapsed:.1f}s — loss: {_new_losses[-1]:.3f} — {_n_params:,} params')

    # Init model if needed
    if get_model() is None:
        set_model(FlowNetwork(key, width, depth))

    return (target_name, width, depth, T, n_steps, n_samples, n_epochs,
            current_params, target_means, target_stds, target_weights)


@app.cell(hide_code=True)
def _(
    alt, jax, jnp, np, pd,
    T, n_steps, target_name,
    target_means, target_stds, target_weights,
    target_prob, compute_log_prob, flow_forward,
    get_model, get_losses, get_trained, get_snapshots,
    get_train_params, current_params, epoch_slider,
):
    _PLOT_W = 700
    z_grid = np.linspace(-5, 5, 300)

    # --- Compute densities ---
    base_density = np.exp(-0.5 * z_grid ** 2) / np.sqrt(2 * np.pi)
    target_density = np.array([float(target_prob(z, target_means, target_stds, target_weights))
                               for z in jnp.array(z_grid)])

    _model = get_model()
    _snapshots = get_snapshots() or []
    _selected_epoch = epoch_slider.value
    _train_params = get_train_params()
    _is_stale = _train_params is not None and _train_params != current_params
    _has_trained = _model is not None and get_trained() and not _is_stale

    if _has_trained and _snapshots:
        _nearest = min(_snapshots, key=lambda s: abs(s[0] - _selected_epoch))
        _snap_epoch, _snap_model = _nearest
    elif _has_trained:
        _snap_epoch, _snap_model = 0, _model
    else:
        _snap_epoch, _snap_model = None, None

    # --- Top panel: output density ---
    _top_records = [{'z': float(z), 'density': float(d), 'series': f'Target ({target_name})'}
                    for z, d in zip(z_grid, target_density)]

    if _snap_model is not None:
        _z_jax = jnp.array(z_grid)
        _log_probs = jax.vmap(lambda x: compute_log_prob(_snap_model, x, T, n_steps))(_z_jax)
        _learned = np.clip(np.exp(np.array(_log_probs)), 0, None)
        _learned_label = f'Learned (epoch {_snap_epoch})'
        _top_records += [{'z': float(z), 'density': float(d), 'series': _learned_label}
                         for z, d in zip(z_grid, _learned)]
    else:
        _learned_label = None

    _top_df = pd.DataFrame(_top_records)
    _target_label = f'Target ({target_name})'
    _domain = [_target_label] + ([_learned_label] if _learned_label else [])
    _colors = ['black'] + (['#e34a33'] if _learned_label else [])
    _dashes = [[6, 4]] + ([[]] if _learned_label else [])

    _top_lines = alt.Chart(_top_df).mark_line(strokeWidth=4).encode(
        x=alt.X('z:Q', scale=alt.Scale(domain=[-5, 5]), title='x'),
        y=alt.Y('density:Q', title='p(x)'),
        color=alt.Color('series:N', scale=alt.Scale(domain=_domain, range=_colors),
                         legend=alt.Legend(orient='top-right', title=None)),
        strokeDash=alt.StrokeDash('series:N', scale=alt.Scale(domain=_domain, range=_dashes),
                                   legend=None),
    )

    if _learned_label:
        _fill_df = pd.DataFrame({'z': z_grid, 'density': _learned})
        _fill = alt.Chart(_fill_df).mark_area(opacity=0.3, color='#e34a33').encode(
            x='z:Q', y='density:Q')
        _top_chart = (_fill + _top_lines).properties(width=_PLOT_W, height=170,
                                                      title='Output density p(z(T))')
    else:
        _top_chart = _top_lines.properties(width=_PLOT_W, height=170,
                                            title='Output density p(z(T))')

    # --- Bottom panel: base density ---
    _bot_df = pd.DataFrame({'z': z_grid, 'density': base_density})
    _bot_area = alt.Chart(_bot_df).mark_area(opacity=0.3, color='steelblue').encode(
        x=alt.X('z:Q', scale=alt.Scale(domain=[-5, 5]), title='z'),
        y=alt.Y('density:Q', title='p(z)'),
    )
    _bot_line = alt.Chart(_bot_df).mark_line(color='steelblue', strokeWidth=4).encode(
        x='z:Q', y='density:Q')
    _bot_chart = (_bot_area + _bot_line).properties(width=_PLOT_W, height=170,
                                                     title='Base density p(z(0)) = N(0,1)')

    # --- Middle panel: flow heatmap + streamlines ---
    if _snap_model is not None:
        _nz = 120
        _z0_heat = jnp.linspace(-5, 5, _nz)

        def _traj_with_logdet(z0):
            _dt = T / n_steps
            z = z0
            log_det = 0.0
            zs = [z0]
            lds = [0.0]
            for k in range(n_steps):
                t_k = k * _dt
                dfz = jax.grad(lambda _z: _snap_model(_z, t_k))(z)
                z = z + _dt * _snap_model(z, t_k)
                log_det = log_det + _dt * dfz
                zs.append(z)
                lds.append(log_det)
            return jnp.stack(zs), jnp.stack(lds)

        _all_zs, _all_lds = jax.vmap(_traj_with_logdet)(_z0_heat)
        _all_zs_np = np.array(_all_zs)
        _all_lds_np = np.array(_all_lds)

        _base_lp = -0.5 * np.array(_z0_heat)**2 - 0.5 * np.log(2 * np.pi)
        _dens = np.clip(np.exp(_base_lp[:, None] - _all_lds_np), 0, None)

        _t_vals = np.linspace(0, T, n_steps + 1)
        _z_reg = np.linspace(-5, 5, _nz)
        _heat = np.zeros((_nz, n_steps + 1))
        for _ti in range(n_steps + 1):
            _zpos = _all_zs_np[:, _ti]
            _dvals = _dens[:, _ti]
            _si = np.argsort(_zpos)
            _heat[:, _ti] = np.interp(_z_reg, _zpos[_si], _dvals[_si], left=0, right=0)

        # Heatmap as mark_rect
        _dz = _z_reg[1] - _z_reg[0]
        _dt_step = _t_vals[1] - _t_vals[0] if len(_t_vals) > 1 else T
        _zz, _tt = np.meshgrid(_z_reg, _t_vals, indexing='ij')
        _heat_df = pd.DataFrame({
            'z': _zz.ravel(), 't': _tt.ravel(),
            'z2': (_zz + _dz).ravel(), 't2': np.minimum((_tt + _dt_step), T).ravel(),
            'density': _heat.ravel(),
        })
        _heatmap = alt.Chart(_heat_df).mark_rect(stroke=None, clip=True).encode(
            x=alt.X('z:Q', scale=alt.Scale(domain=[-5, 5]), title='z'),
            x2='z2:Q',
            y=alt.Y('t:Q', scale=alt.Scale(domain=[0, T], clamp=True), title='t'),
            y2='t2:Q',
            color=alt.Color('density:Q',
                            scale=alt.Scale(scheme='oranges', domainMin=0),
                            legend=None),
        )

        # Streamlines
        _n_streams = 30
        _z0_str = jnp.linspace(-3.5, 3.5, _n_streams)
        _trajs = np.array(jax.vmap(lambda z0: flow_forward(_snap_model, z0, T, n_steps))(_z0_str))
        _stream_records = []
        for _j in range(_n_streams):
            for _ti in range(len(_t_vals)):
                _stream_records.append({'z': float(_trajs[_j, _ti]), 't': float(_t_vals[_ti]), 'id': _j})
        _stream_df = pd.DataFrame(_stream_records)
        _streams = alt.Chart(_stream_df).mark_line(
            color='#333333', opacity=0.5, strokeWidth=1.4
        ).encode(x='z:Q', y='t:Q', detail='id:N')

        _mid_chart = (_heatmap + _streams).properties(
            width=_PLOT_W, height=168, title=f'Flow trajectories (epoch {_snap_epoch})')
    else:
        _placeholder = pd.DataFrame({'z': [0], 't': [T / 2], 'label': ['Click "Train" to see flow']})
        _mid_chart = alt.Chart(_placeholder).mark_text(fontSize=21, color='gray').encode(
            x=alt.X('z:Q', scale=alt.Scale(domain=[-5, 5]), title='z'),
            y=alt.Y('t:Q', scale=alt.Scale(domain=[0, T]), title='t'),
            text='label:N',
        ).properties(width=_PLOT_W, height=168, title='Flow trajectories')

    plot_output = alt.vconcat(
        _top_chart, _mid_chart, _bot_chart
    ).resolve_scale(color='independent').configure_axis(
        labelFontSize=15, titleFontSize=16,
    ).configure_title(
        fontSize=18,
    ).configure_legend(
        labelFontSize=14, titleFontSize=15,
    )
    return (plot_output,)


@app.cell(hide_code=True)
def _(
    mo,
    target_dropdown, width_slider, depth_slider, flow_T_slider,
    ode_steps_slider, lr_slider, n_samples_slider, epochs_slider,
    weight_decay_slider, grad_clip_slider, cosine_schedule_checkbox,
    train_button, reset_button, epoch_slider,
    get_train_status,
):
    _status = get_train_status()
    _status_html = (f'<div style="font-size:12px;color:#495057;margin-top:0.3em;">{_status}</div>'
                    if _status else '')
    sidebar = mo.Html(f'''
    <div class="app-sidebar">
        <h4>Target Distribution</h4>
        {target_dropdown}

        <h4>Network</h4>
        {width_slider}
        {depth_slider}

        <h4>Flow Settings</h4>
        {flow_T_slider}
        {ode_steps_slider}

        <h4>Training</h4>
        {lr_slider}
        {n_samples_slider}
        {epochs_slider}

        <h4>Regularization</h4>
        {weight_decay_slider}
        {grad_clip_slider}
        {cosine_schedule_checkbox}

        <div style="display: flex; gap: 0.5em; margin-top: 1em; flex-wrap: wrap;">
            {train_button}
            {reset_button}
        </div>
        {_status_html}

        <h4>Visualization</h4>
        {epoch_slider}
    </div>
    ''')
    return (sidebar,)


@app.cell(hide_code=True)
def _(mo, header, plot_output, sidebar):
    mo.vstack([
        header,
        mo.Html(f'''
        <div class="app-layout">
            <div class="app-plot">
                {mo.as_html(plot_output)}
            </div>
            {sidebar}
        </div>
        '''),
    ])
    return


if __name__ == "__main__":
    app.run()
