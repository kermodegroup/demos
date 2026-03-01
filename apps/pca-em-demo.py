# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "altair",
#     "pandas",
#     "numpy==2.2.5",
#     "pillow",
#     "qrcode==8.2",
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

        .square-chart-container {
            width: 100%;
            max-width: calc(100vh - 80px - 2em);
            aspect-ratio: 1;
            position: relative;
            overflow: hidden;
        }

        .square-chart-container iframe {
            position: absolute !important;
            top: 0;
            left: 0;
            width: 100% !important;
            height: 100% !important;
            border: none;
        }

        .app-sidebar-container {
            z-index: 10;
            position: relative;
            flex-shrink: 0;
            width: 320px;
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

    return alt, mo, np, pd


@app.cell(hide_code=True)
def _():
    import qrcode
    import io
    import base64

    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://sciml.warwick.ac.uk/wasm/pca-em-demo/')
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
                <p style='font-size: 24px; margin: 0; padding: 0; line-height: 1.3;'><b>PCA as an EM Algorithm</b>
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
    # Data generation controls
    n_slider = mo.ui.slider(10, 100, 5, 40, label='$N$ data points')
    sigma_slider = mo.ui.slider(0.1, 2.0, 0.1, 0.5, label='Noise $\\sigma$')
    angle_slider = mo.ui.slider(0, 180, 5, 30, label='True PC angle')
    stretch_slider = mo.ui.slider(1.0, 5.0, 0.5, 3.0, label='Stretch')
    seed_slider = mo.ui.slider(0, 20, 1, 7, label='Random seed')

    # View checkboxes
    show_true_pc = mo.ui.checkbox(True, label='Show true PC')
    show_springs = mo.ui.checkbox(True, label='Show springs')
    show_projections = mo.ui.checkbox(True, label='Show projections')

    return (
        n_slider, sigma_slider, angle_slider, stretch_slider, seed_slider,
        show_true_pc, show_springs, show_projections,
    )


@app.cell(hide_code=True)
def _(mo):
    get_state, set_state = mo.state(None)
    return get_state, set_state


@app.cell(hide_code=True)
def _(np):
    def generate_pca_data(N, sigma, angle_deg, stretch, seed):
        """Generate 2D data with a dominant principal component direction."""
        rng = np.random.default_rng(seed)
        angle_rad = np.radians(angle_deg)
        W_true = np.array([np.cos(angle_rad), np.sin(angle_rad)])

        # Latent variable z ~ N(0, stretch^2)
        z = rng.normal(0, stretch, N)
        # Data: x = W_true * z + noise
        noise = rng.normal(0, sigma, (N, 2))
        X = np.outer(z, W_true) + noise

        return X, W_true

    def e_step(X, W):
        """E-step: orthogonal projection onto W.

        z_n = (W^T W)^{-1} W^T x_n = (W . x_n) / ||W||^2
        """
        Z = X @ W / np.dot(W, W)
        return Z

    def m_step(X, Z):
        """M-step: rotate rod to minimize spring energy.

        W_new = (sum_n x_n z_n) / (sum_n z_n^2)
        """
        W_new = (X.T @ Z) / (Z @ Z)
        return W_new

    def compute_spring_energy(X, W, Z):
        """Total spring energy: sum_n ||x_n - W z_n||^2."""
        reconstructions = np.outer(Z, W)
        return float(np.sum((X - reconstructions) ** 2))

    def compute_angle_to_true(W, W_true):
        """Angle between current W and true PC (in degrees), in [0, 90]."""
        cos_angle = abs(np.dot(W, W_true)) / (np.linalg.norm(W) * np.linalg.norm(W_true))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    return generate_pca_data, e_step, m_step, compute_spring_energy, compute_angle_to_true


@app.cell(hide_code=True)
def _(
    mo, np, get_state, set_state,
    e_step, m_step, compute_spring_energy, compute_angle_to_true,
):
    def on_e_step(_):
        s = get_state()
        if s is None:
            return
        X, W, W_true = s['X'], s['W'], s['W_true']
        Z = e_step(X, W)
        energy = compute_spring_energy(X, W, Z)
        angle = compute_angle_to_true(W, W_true)
        iteration = s['iteration'] + 1
        history = s['history'] + [{'step': iteration, 'type': 'E', 'energy': energy, 'angle': angle, 'W_norm': float(np.linalg.norm(W))}]
        set_state({**s, 'Z': Z, 'iteration': iteration, 'last_step': 'E-step', 'history': history})

    def on_m_step(_):
        s = get_state()
        if s is None:
            return
        X, Z, W_true = s['X'], s['Z'], s['W_true']
        if Z is None:
            return
        W_new = m_step(X, Z)
        energy = compute_spring_energy(X, W_new, Z)
        angle = compute_angle_to_true(W_new, W_true)
        iteration = s['iteration'] + 1
        history = s['history'] + [{'step': iteration, 'type': 'M', 'energy': energy, 'angle': angle, 'W_norm': float(np.linalg.norm(W_new))}]
        set_state({**s, 'W': W_new, 'iteration': iteration, 'last_step': 'M-step', 'history': history})

    def on_full_em(_):
        s = get_state()
        if s is None:
            return
        X, W, W_true = s['X'], s['W'], s['W_true']
        iteration = s['iteration']
        history = list(s['history'])

        # E-step
        Z = e_step(X, W)
        energy = compute_spring_energy(X, W, Z)
        angle = compute_angle_to_true(W, W_true)
        iteration += 1
        history.append({'step': iteration, 'type': 'E', 'energy': energy, 'angle': angle, 'W_norm': float(np.linalg.norm(W))})

        # M-step
        W_new = m_step(X, Z)
        energy = compute_spring_energy(X, W_new, Z)
        angle = compute_angle_to_true(W_new, W_true)
        iteration += 1
        history.append({'step': iteration, 'type': 'M', 'energy': energy, 'angle': angle, 'W_norm': float(np.linalg.norm(W_new))})

        set_state({**s, 'W': W_new, 'Z': Z, 'iteration': iteration, 'last_step': 'Full EM', 'history': history})

    def on_reset(_):
        s = get_state()
        if s is None:
            return
        rng = np.random.default_rng(12345)
        W_init = rng.normal(0, 1, 2)
        W_init = W_init / np.linalg.norm(W_init)
        set_state({**s, 'W': W_init, 'Z': None, 'iteration': 0, 'last_step': 'Init', 'history': []})

    e_step_btn = mo.ui.button(label="E-step", on_click=on_e_step)
    m_step_btn = mo.ui.button(label="M-step", on_click=on_m_step)
    full_em_btn = mo.ui.button(label="Full EM", on_click=on_full_em)
    reset_btn = mo.ui.button(label="Reset", on_click=on_reset)

    return e_step_btn, m_step_btn, full_em_btn, reset_btn


@app.cell(hide_code=True)
def _(
    np, get_state, set_state,
    generate_pca_data,
    n_slider, sigma_slider, angle_slider, stretch_slider, seed_slider,
):
    # React to slider changes and initialize/reinitialize state
    _N = n_slider.value
    _sigma = sigma_slider.value
    _angle = angle_slider.value
    _stretch = stretch_slider.value
    _seed = seed_slider.value

    _data_hash = f"{_N}_{_sigma}_{_angle}_{_stretch}_{_seed}"

    _current = get_state()
    if _current is None or _current.get('data_hash') != _data_hash:
        _X, _W_true = generate_pca_data(_N, _sigma, _angle, _stretch, _seed)
        _rng = np.random.default_rng(12345)
        _W_init = _rng.normal(0, 1, 2)
        _W_init = _W_init / np.linalg.norm(_W_init)
        set_state({
            'X': _X,
            'W': _W_init,
            'Z': None,
            'iteration': 0,
            'last_step': 'Init',
            'history': [],
            'W_true': _W_true,
            'data_hash': _data_hash,
        })

    return


@app.cell(hide_code=True)
def _(
    alt, np, pd, get_state,
    show_true_pc, show_springs, show_projections,
):
    _s = get_state()

    if _s is not None:
        _X = _s['X']
        _W = _s['W']
        _Z = _s['Z']
        _W_true = _s['W_true']
        _iteration = _s['iteration']
        _last_step = _s['last_step']

        _N_pts = _X.shape[0]

        # Fixed axis limits based on data only (won't jump between steps)
        _data_max = float(np.abs(_X).max()) * 1.15 + 0.5

        _x_scale = alt.Scale(domain=[-_data_max, _data_max])
        _y_scale = alt.Scale(domain=[-_data_max, _data_max])

        # Compute projections (E-step z values are orthogonal projections)
        if _Z is not None:
            _proj = np.outer(_Z, _W)  # N x 2
            _spring_lengths = np.sqrt(np.sum((_X - _proj) ** 2, axis=1))
        else:
            _proj = np.zeros_like(_X)
            _spring_lengths = np.zeros(_N_pts)

        _layers = []

        # --- Legend ---
        _legend_items = [
            {'label': 'Data points', 'color': '#2ca02c'},
            {'label': 'W (rod)', 'color': 'red'},
        ]
        if show_true_pc.value:
            _legend_items.append({'label': 'True PC', 'color': 'black'})
        if show_projections.value and _Z is not None:
            _legend_items.append({'label': 'Projections', 'color': 'cyan'})
        if show_springs.value and _Z is not None:
            _legend_items.append({'label': 'Springs', 'color': '#ff7f0e'})

        _legend_df = pd.DataFrame(_legend_items)
        _legend_layer = alt.Chart(_legend_df).mark_circle(opacity=0, size=120).encode(
            color=alt.Color(
                'label:N',
                scale=alt.Scale(
                    domain=[item['label'] for item in _legend_items],
                    range=[item['color'] for item in _legend_items],
                ),
                legend=alt.Legend(title=None, orient='top-left'),
            ),
        )
        _layers.append(_legend_layer)

        # --- Layer 1: Spring lines ---
        if show_springs.value and _Z is not None:
            _spring_rows = []
            for _i in range(_N_pts):
                _spring_rows.append({
                    'x': float(_X[_i, 0]), 'y': float(_X[_i, 1]),
                    'point_idx': _i, 'endpoint': 0, 'spring_length': float(_spring_lengths[_i]),
                })
                _spring_rows.append({
                    'x': float(_proj[_i, 0]), 'y': float(_proj[_i, 1]),
                    'point_idx': _i, 'endpoint': 1, 'spring_length': float(_spring_lengths[_i]),
                })
            _spring_df = pd.DataFrame(_spring_rows)

            _spring_layer = alt.Chart(_spring_df).mark_line(
                strokeWidth=2.5, opacity=0.5, color='#ff7f0e'
            ).encode(
                x=alt.X('x:Q', scale=_x_scale),
                y=alt.Y('y:Q', scale=_y_scale),
                detail='point_idx:N',
                order='endpoint:O',
            )
            _layers.append(_spring_layer)

        # --- Layer 2: Rod (current W) ---
        _W_norm = np.linalg.norm(_W)
        if _W_norm > 0:
            _W_dir = _W / _W_norm
        else:
            _W_dir = np.array([1.0, 0.0])
        _t_extent = _data_max
        _rod_df = pd.DataFrame({
            'x': [-_t_extent * _W_dir[0], _t_extent * _W_dir[0]],
            'y': [-_t_extent * _W_dir[1], _t_extent * _W_dir[1]],
        })
        _rod_layer = alt.Chart(_rod_df).mark_line(
            color='red', strokeWidth=3
        ).encode(
            x=alt.X('x:Q', scale=_x_scale),
            y=alt.Y('y:Q', scale=_y_scale),
        )
        _layers.append(_rod_layer)

        # --- Layer 3: True PC line ---
        if show_true_pc.value:
            _true_dir = _W_true / np.linalg.norm(_W_true)
            _true_pc_df = pd.DataFrame({
                'x': [-_t_extent * _true_dir[0], _t_extent * _true_dir[0]],
                'y': [-_t_extent * _true_dir[1], _t_extent * _true_dir[1]],
            })
            _true_pc_layer = alt.Chart(_true_pc_df).mark_line(
                strokeDash=[5, 5], color='black', strokeWidth=2, opacity=0.5
            ).encode(
                x=alt.X('x:Q', scale=_x_scale),
                y=alt.Y('y:Q', scale=_y_scale),
            )
            _layers.append(_true_pc_layer)

        # --- Layer 4: Projection points ---
        if show_projections.value and _Z is not None:
            _proj_df = pd.DataFrame({
                'x': _proj[:, 0],
                'y': _proj[:, 1],
                'z_value': _Z,
                'point_idx': list(range(_N_pts)),
            })
            _proj_layer = alt.Chart(_proj_df).mark_circle(
                color='cyan', size=80, stroke='darkblue', strokeWidth=1
            ).encode(
                x=alt.X('x:Q', scale=_x_scale),
                y=alt.Y('y:Q', scale=_y_scale),
                tooltip=[
                    alt.Tooltip('point_idx:N', title='Point'),
                    alt.Tooltip('z_value:Q', title='z', format='.3f'),
                    alt.Tooltip('x:Q', title='proj x\u2081', format='.3f'),
                    alt.Tooltip('y:Q', title='proj x\u2082', format='.3f'),
                ],
            )
            _layers.append(_proj_layer)

        # --- Layer 5: Data points ---
        _data_df = pd.DataFrame({
            'x': _X[:, 0],
            'y': _X[:, 1],
            'point_idx': list(range(_N_pts)),
            'z_value': _Z if _Z is not None else np.zeros(_N_pts),
            'proj_x': _proj[:, 0],
            'proj_y': _proj[:, 1],
            'spring_length': _spring_lengths,
        })

        _hover = alt.selection_point(on='pointerover', nearest=True, fields=['point_idx'])

        _data_layer = alt.Chart(_data_df).mark_circle(
            color='#2ca02c', size=120, stroke='white', strokeWidth=1
        ).encode(
            x=alt.X('x:Q', scale=_x_scale, title='x\u2081'),
            y=alt.Y('y:Q', scale=_y_scale, title='x\u2082'),
            opacity=alt.condition(_hover, alt.value(1.0), alt.value(0.7)),
            tooltip=[
                alt.Tooltip('point_idx:N', title='Point'),
                alt.Tooltip('x:Q', title='x\u2081', format='.3f'),
                alt.Tooltip('y:Q', title='x\u2082', format='.3f'),
                alt.Tooltip('z_value:Q', title='z (latent)', format='.3f'),
                alt.Tooltip('proj_x:Q', title='proj x\u2081', format='.3f'),
                alt.Tooltip('proj_y:Q', title='proj x\u2082', format='.3f'),
                alt.Tooltip('spring_length:Q', title='spring length', format='.3f'),
            ],
        ).add_params(_hover)
        _layers.append(_data_layer)

        # --- Combine ---
        main_chart = alt.layer(*_layers).properties(
            width='container', height='container',
            title=f'EM Iteration {_iteration} \u2014 {_last_step}'
        ).configure_axis(
            grid=True, gridOpacity=0.2,
            labelFontSize=20, titleFontSize=24,
            tickCount=5,
        ).configure_title(
            fontSize=28
        ).configure_legend(
            titleFontSize=18, labelFontSize=16
        )
    else:
        main_chart = alt.Chart(pd.DataFrame({'x': [0], 'y': [0]})).mark_point().encode(
            x='x:Q', y='y:Q'
        ).properties(title='Loading...')

    return (main_chart,)


@app.cell(hide_code=True)
def _(alt, mo, pd, get_state):
    _s_conv = get_state()

    if _s_conv is not None and len(_s_conv.get('history', [])) > 0:
        _history = _s_conv['history']
        _iteration = _s_conv['iteration']

        # Latest metrics
        _latest = _history[-1]
        _energy_val = _latest['energy']
        _angle_val = _latest['angle']
        _w_norm_val = _latest['W_norm']

        _metrics_html = f'''
        <table style="width: 100%; font-size: 12px; border-collapse: collapse;">
            <tr><td style="padding: 2px 4px;">Iteration</td><td style="text-align: right; font-family: monospace; padding: 2px 4px;">{_iteration}</td></tr>
            <tr><td style="padding: 2px 4px;">Spring energy</td><td style="text-align: right; font-family: monospace; padding: 2px 4px;">{_energy_val:.2f}</td></tr>
            <tr><td style="padding: 2px 4px;">Angle to true</td><td style="text-align: right; font-family: monospace; padding: 2px 4px;">{_angle_val:.2f}\u00b0</td></tr>
            <tr><td style="padding: 2px 4px;">\u2016W\u2016</td><td style="text-align: right; font-family: monospace; padding: 2px 4px;">{_w_norm_val:.4f}</td></tr>
        </table>
        '''

        # Convergence mini-chart
        _hist_df = pd.DataFrame(_history)

        _energy_line = alt.Chart(_hist_df).mark_line(
            point=True, color='#e45756', strokeWidth=2
        ).encode(
            x=alt.X('step:Q', title='Step'),
            y=alt.Y('energy:Q', title='Energy', scale=alt.Scale(type='log'), axis=alt.Axis(titleColor='#e45756')),
            tooltip=[
                alt.Tooltip('step:Q', title='Step'),
                alt.Tooltip('type:N', title='Type'),
                alt.Tooltip('energy:Q', title='Energy', format='.2f'),
            ],
        )

        _angle_line = alt.Chart(_hist_df).mark_line(
            point=True, color='#4c78a8', strokeWidth=2
        ).encode(
            x=alt.X('step:Q', title='Step'),
            y=alt.Y('angle:Q', title='Angle (\u00b0)', scale=alt.Scale(type='log'), axis=alt.Axis(titleColor='#4c78a8')),
            tooltip=[
                alt.Tooltip('step:Q', title='Step'),
                alt.Tooltip('type:N', title='Type'),
                alt.Tooltip('angle:Q', title='Angle', format='.2f'),
            ],
        )

        _conv_chart = alt.layer(_energy_line, _angle_line).resolve_scale(
            y='independent'
        ).properties(
            width='container', height=120,
        ).configure_axis(
            labelFontSize=12, titleFontSize=14,
        )

        convergence_panel = mo.Html(f'''
        {_metrics_html}
        <div style="margin-top: 0.5em;">{mo.as_html(_conv_chart)}</div>
        ''')
    else:
        convergence_panel = mo.Html('''
        <table style="width: 100%; font-size: 12px; border-collapse: collapse;">
            <tr><td style="padding: 2px 4px;">Iteration</td><td style="text-align: right; font-family: monospace; padding: 2px 4px;">0</td></tr>
            <tr><td style="padding: 2px 4px;">Spring energy</td><td style="text-align: right; font-family: monospace; padding: 2px 4px;">\u2014</td></tr>
            <tr><td style="padding: 2px 4px;">Angle to true</td><td style="text-align: right; font-family: monospace; padding: 2px 4px;">\u2014</td></tr>
            <tr><td style="padding: 2px 4px;">\u2016W\u2016</td><td style="text-align: right; font-family: monospace; padding: 2px 4px;">\u2014</td></tr>
        </table>
        <p style="font-size: 11px; color: #888; margin-top: 0.5em;">Click E-step or M-step to begin.</p>
        ''')

    return (convergence_panel,)


@app.cell(hide_code=True)
def _(
    mo,
    n_slider, sigma_slider, angle_slider, stretch_slider, seed_slider,
    show_true_pc, show_springs, show_projections,
    e_step_btn, m_step_btn, full_em_btn, reset_btn,
    convergence_panel,
):
    data_section = mo.vstack([
        mo.Html("<h4>Data</h4>"),
        n_slider,
        sigma_slider,
        angle_slider,
        stretch_slider,
        seed_slider,
    ], gap="0.3em")

    algorithm_section = mo.vstack([
        mo.Html("<h4>Algorithm</h4>"),
        mo.hstack([e_step_btn, m_step_btn, full_em_btn, reset_btn], gap="0.5em"),
    ], gap="0.3em")

    view_section = mo.vstack([
        mo.Html("<h4>View</h4>"),
        show_true_pc,
        show_springs,
        show_projections,
    ], gap="0.3em")

    convergence_section = mo.vstack([
        mo.Html("<h4>Convergence</h4>"),
        convergence_panel,
    ], gap="0.3em")

    sidebar = mo.vstack([data_section, algorithm_section, view_section, convergence_section], gap="1em")
    return (sidebar,)


@app.cell(hide_code=True)
def _(mo, header, main_chart, sidebar):
    sidebar_html = mo.Html(f'''
    <div class="app-sidebar">
        {sidebar}
    </div>
    ''')
    _spec_json = main_chart.to_json()
    _chart_html = f'''<!DOCTYPE html>
    <html><head>
    <script src="https://cdn.jsdelivr.net/npm/vega@6"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@6.1.0"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@7"></script>
    <style>html, body {{ margin:0; padding:0; width:100%; height:100%; overflow:hidden; }}
    #vis {{ width:100%; height:100%; }}</style>
    </head><body><div id="vis"></div>
    <script>vegaEmbed('#vis', {_spec_json}, {{actions:true}});</script>
    </body></html>'''
    _chart_iframe = mo.iframe(_chart_html)
    mo.Html(f'''
    {header}
    <div class="app-layout">
        <div class="app-plot"><div class="square-chart-container">{_chart_iframe}</div></div>
        <div class="app-sidebar-container">
            {sidebar_html}
        </div>
    </div>
    ''')
    return


if __name__ == "__main__":
    app.run()
