# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "altair",
#     "pandas",
#     "numpy==2.2.5",
#     "scikit-learn==1.6.1",
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
            max-height: calc(100vh - 120px);
            overflow-y: auto;
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
    import base64
    import io
    import zlib
    from sklearn.decomposition import PCA
    from PIL import Image

    return Image, PCA, alt, base64, io, mo, np, pd, zlib


@app.cell(hide_code=True)
def _():
    import qrcode
    import io as _io
    import base64 as _base64

    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://kermodegroup.github.io/demos/pca-demo/')
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    buffer = _io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    qr_base64 = _base64.b64encode(buffer.read()).decode()
    return (qr_base64,)


@app.cell(hide_code=True)
def _(mo, qr_base64):
    header = mo.Html(f'''
    <div class="app-header">
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 0; padding: 0;">
            <div>
                <p style='font-size: 24px; margin: 0; padding: 0; line-height: 1.3;'><b>PCA Explorer &mdash; MNIST</b>
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
def _(base64, io, np, zlib):
    # --- Load MNIST 28x28 data (compressed inline) ---
    from _mnist_data import MNIST_DATA_B64

    _buf = io.BytesIO(zlib.decompress(base64.b64decode(MNIST_DATA_B64)))
    _npz = np.load(_buf)
    X_data = _npz['X'].astype(np.float64) / 255.0  # normalize to [0,1]
    y_labels = _npz['y']

    return X_data, y_labels


@app.cell(hide_code=True)
def _(PCA, X_data, np):
    # Fit PCA with up to 100 components
    pca_full = PCA(n_components=100)
    X_transformed = pca_full.fit_transform(X_data)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    return X_transformed, cumulative_variance, pca_full


@app.cell(hide_code=True)
def _(mo):
    # UI controls
    n_components_slider = mo.ui.slider(
        start=1, stop=100, step=1, value=20,
        label="Reconstruction components (M)",
        debounce=True,
    )

    pc_options = {f"PC {i+1}": str(i) for i in range(20)}
    x_axis_dropdown = mo.ui.dropdown(
        options=pc_options, value="PC 1", label="X axis"
    )
    y_axis_dropdown = mo.ui.dropdown(
        options=pc_options, value="PC 2", label="Y axis"
    )

    digit_options = [str(i) for i in range(10)]
    digit_filter = mo.ui.multiselect(
        options=digit_options, value=digit_options, label="Digits"
    )

    ppca_checkbox = mo.ui.checkbox(label="Enable Probabilistic PCA", value=False)
    generate_button = mo.ui.button(label="Generate Samples", value=0, on_click=lambda v: v + 1)

    return digit_filter, generate_button, n_components_slider, pc_options, ppca_checkbox, x_axis_dropdown, y_axis_dropdown


@app.cell(hide_code=True)
def _(mo):
    # State for PPCA generated samples
    get_ppca_samples, set_ppca_samples = mo.state(None)
    return get_ppca_samples, set_ppca_samples


@app.cell(hide_code=True)
def _(PCA, X_data, generate_button, n_components_slider, np, pca_full, set_ppca_samples):
    # PPCA sample generation - runs when button is clicked
    _click_count = generate_button.value
    if _click_count > 0:
        _M = n_components_slider.value
        _pca_m = PCA(n_components=_M)
        _pca_m.fit(X_data)
        _mu = _pca_m.mean_
        _C = _pca_m.get_covariance()
        _noise_var = _pca_m.noise_variance_
        _generated_signals = np.random.multivariate_normal(_mu, _C, size=4)
        _generated_projected = pca_full.transform(_generated_signals)
        set_ppca_samples({
            'signals': _generated_signals,
            'projected': _generated_projected,
            'M': _M,
            'noise_var': _noise_var,
        })
    return


@app.cell(hide_code=True)
def _(X_transformed, alt, digit_filter, mo, np, pd, x_axis_dropdown, y_axis_dropdown, y_labels):
    # Scatter plot construction
    pc_x_idx = int(x_axis_dropdown.value)
    pc_y_idx = int(y_axis_dropdown.value)
    selected_digits = [int(d) for d in digit_filter.value]

    scatter_df = pd.DataFrame({
        'pc_x': X_transformed[:, pc_x_idx],
        'pc_y': X_transformed[:, pc_y_idx],
        'digit': [str(int(d)) for d in y_labels],
        'idx': np.arange(len(X_transformed)),
    })

    # Filter by selected digits
    if selected_digits:
        mask = scatter_df['digit'].isin([str(d) for d in selected_digits])
        scatter_df = scatter_df[mask].reset_index(drop=True)

    # Selection for hover interaction
    hover_select = alt.selection_point(
        on='mouseover', nearest=True, fields=['idx'], name='hover_select'
    )

    # Main scatter — use digit value as text glyph
    scatter = alt.Chart(scatter_df).mark_text(fontSize=18, fontWeight='bold').encode(
        x=alt.X('pc_x:Q', title=f'PC {pc_x_idx + 1}'),
        y=alt.Y('pc_y:Q', title=f'PC {pc_y_idx + 1}'),
        text='digit:N',
        color=alt.Color('digit:N', scale=alt.Scale(scheme='category10'), legend=None),
        opacity=alt.condition(hover_select, alt.value(1.0), alt.value(0.5)),
        tooltip=[
            alt.Tooltip('digit:N', title='Digit'),
            alt.Tooltip('pc_x:Q', title=f'PC {pc_x_idx + 1}', format='.2f'),
            alt.Tooltip('pc_y:Q', title=f'PC {pc_y_idx + 1}', format='.2f'),
            alt.Tooltip('idx:Q', title='Index'),
        ],
    ).add_params(hover_select)

    chart = scatter.properties(
        width='container', height=500,
        title='PCA Projection of MNIST'
    ).configure_axis(
        grid=True, gridOpacity=0.3,
        labelFontSize=14, titleFontSize=16
    ).configure_title(fontSize=18)

    interactive_chart = mo.ui.altair_chart(chart, chart_selection=False)

    return interactive_chart, scatter_df


@app.cell(hide_code=True)
def _(interactive_chart):
    # Pass-through display cell
    chart_display = interactive_chart
    return (chart_display,)


@app.cell(hide_code=True)
def _(Image, X_data, base64, interactive_chart, io, mo, n_components_slider, np, pd, pca_full, scatter_df, y_labels):
    # Helper: render a 784-vector as a base64 PNG for inline display
    def _digit_to_html(arr_784, label="", width=112):
        img = Image.fromarray(
            (arr_784.reshape(28, 28).clip(0, 1) * 255).astype('uint8'), mode='L'
        )
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode()
        return (
            f'<div style="text-align:center">'
            f'<img src="data:image/png;base64,{b64}" width="{width}" '
            f'style="image-rendering:pixelated"/>'
            f'<br><small>{label}</small></div>'
        )

    # Selection handling + image preview
    _M = n_components_slider.value
    _filtered = interactive_chart.apply_selection(scatter_df)

    selected_idx = None
    if _filtered is not None and isinstance(_filtered, pd.DataFrame) and len(_filtered) > 0 and len(_filtered) < len(scatter_df):
        selected_idx = int(_filtered['idx'].iloc[0])

    if selected_idx is not None:
        original = X_data[selected_idx]
        coeffs = pca_full.transform(X_data[selected_idx:selected_idx+1])[0]
        reconstruction = pca_full.mean_.copy()
        for k in range(_M):
            reconstruction += coeffs[k] * pca_full.components_[k]
        rmse = np.sqrt(np.mean((original - reconstruction) ** 2))
        digit = int(y_labels[selected_idx])

        _orig_html = _digit_to_html(original, "Original")
        _recon_html = _digit_to_html(reconstruction, "Reconstruction")

        signal_preview = mo.vstack([
            mo.Html("<h4>Image Preview</h4>"),
            mo.Html(f"<div style='text-align:center; font-weight:bold;'>Digit {digit} &mdash; RMSE={rmse:.3f}, M={_M}</div>"),
            mo.Html(
                f'<div style="display:flex; justify-content:center; gap:1em; align-items:flex-start;">'
                f'{_orig_html}{_recon_html}'
                f'</div>'
            ),
        ], gap="0.3em")
    else:
        # Show mean image when nothing is selected
        _mean_html = _digit_to_html(pca_full.mean_, "Mean image")
        signal_preview = mo.vstack([
            mo.Html("<h4>Image Preview</h4>"),
            mo.Html(f'<div style="display:flex; justify-content:center;">{_mean_html}</div>'),
            mo.Html("<small style='color: #666;'>Hover over a point to see its digit</small>"),
        ], gap="0.3em")

    return (signal_preview,)


@app.cell(hide_code=True)
def _(Image, base64, get_ppca_samples, io, mo, n_components_slider, np, pca_full, ppca_checkbox):
    # PPCA generated samples display — render as digit images
    _ppca_data = get_ppca_samples()
    if ppca_checkbox.value and _ppca_data is not None:
        _signals = _ppca_data['signals']
        _M_gen = _ppca_data['M']
        _noise_var = _ppca_data['noise_var']

        _img_htmls = []
        for _i, _sig in enumerate(_signals):
            _img = Image.fromarray(
                (_sig.reshape(28, 28).clip(0, 1) * 255).astype('uint8'), mode='L'
            )
            _buf = io.BytesIO()
            _img.save(_buf, format='PNG')
            _b64 = base64.b64encode(_buf.getvalue()).decode()
            _img_htmls.append(
                f'<img src="data:image/png;base64,{_b64}" width="56" '
                f'style="image-rendering:pixelated; border:1px solid #ccc;"/>'
            )

        ppca_samples_display = mo.vstack([
            mo.Html(f"<div style='text-align:center;'><small>M={_M_gen}, \u03c3\u00b2={_noise_var:.4f}</small></div>"),
            mo.Html(
                f'<div style="display:flex; justify-content:center; gap:4px; flex-wrap:wrap;">'
                f'{"".join(_img_htmls)}'
                f'</div>'
            ),
        ], gap="0.3em")
    else:
        ppca_samples_display = mo.Html("")

    # Compute noise variance for display even when not generating
    _M = n_components_slider.value
    _n_components = 100
    if ppca_checkbox.value:
        explained = np.sum(pca_full.explained_variance_[:_M])
        total = np.sum(pca_full.explained_variance_)
        _noise_var_est = (total - explained) / (_n_components - _M) if _M < _n_components else 0.0
        ppca_info = mo.Html(f"<small>\u03c3\u00b2 \u2248 {_noise_var_est:.4f} (estimated for M={_M})</small>")
    else:
        ppca_info = mo.Html("")

    return ppca_info, ppca_samples_display


@app.cell(hide_code=True)
def _(alt, cumulative_variance, n_components_slider, np, pd):
    # Explained variance chart
    _M = n_components_slider.value
    var_df = pd.DataFrame({
        'Components': np.arange(1, len(cumulative_variance) + 1),
        'Cumulative Variance': cumulative_variance,
    })

    var_line = alt.Chart(var_df).mark_line(point=True, color='#1f77b4').encode(
        x=alt.X('Components:Q', title='Components', scale=alt.Scale(domain=[1, 100])),
        y=alt.Y('Cumulative Variance:Q', title='Cumulative Var.', scale=alt.Scale(domain=[0, 1])),
        tooltip=[
            alt.Tooltip('Components:Q'),
            alt.Tooltip('Cumulative Variance:Q', format='.1%'),
        ],
    )

    rule_df = pd.DataFrame({'x': [_M]})
    rule = alt.Chart(rule_df).mark_rule(strokeDash=[5, 5], color='red').encode(
        x='x:Q'
    )

    text_df = pd.DataFrame({
        'x': [_M],
        'y': [cumulative_variance[_M - 1]],
        'label': [f'{cumulative_variance[_M - 1]:.1%}'],
    })
    text_mark = alt.Chart(text_df).mark_text(
        align='left', dx=5, dy=-8, fontSize=11, color='red'
    ).encode(
        x='x:Q', y='y:Q', text='label:N'
    )

    variance_chart = (var_line + rule + text_mark).properties(
        width='container', height=120,
        title='Explained Variance'
    )

    return (variance_chart,)


@app.cell(hide_code=True)
def _(
    digit_filter, generate_button, mo, n_components_slider,
    ppca_checkbox, ppca_info, ppca_samples_display,
    signal_preview, variance_chart,
    x_axis_dropdown, y_axis_dropdown,
):
    # Sidebar assembly
    controls_section = mo.vstack([
        mo.Html("<h4>PCA Controls</h4>"),
        n_components_slider,
        x_axis_dropdown,
        y_axis_dropdown,
    ], gap="0.3em")

    digit_section = mo.vstack([
        mo.Html("<h4>Digit Filter</h4>"),
        digit_filter,
    ], gap="0.3em")

    ppca_section = mo.vstack([
        mo.Html("<h4>Probabilistic PCA</h4>"),
        ppca_checkbox,
        ppca_info,
        generate_button,
        ppca_samples_display,
    ], gap="0.3em")

    variance_section = mo.vstack([
        mo.Html("<h4>Explained Variance</h4>"),
        variance_chart,
    ], gap="0.3em")

    sidebar = mo.vstack([
        signal_preview, controls_section, digit_section,
        ppca_section, variance_section,
    ], gap="1em")

    sidebar_html = mo.Html(f'''
    <div class="app-sidebar">
        {sidebar}
    </div>
    ''')
    return (sidebar_html,)


@app.cell(hide_code=True)
def _(mo, header, chart_display, sidebar_html):
    # Layout assembly
    mo.Html(f'''
    {header}
    <div class="app-layout">
        <div class="app-plot">
            {mo.as_html(chart_display)}
        </div>
        <div class="app-sidebar-container">
            {sidebar_html}
        </div>
    </div>
    ''')
    return


if __name__ == "__main__":
    app.run()
