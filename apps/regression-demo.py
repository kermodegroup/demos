# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "numpy==2.2.5",
#     "popsregression==0.3.5",
#     "scikit-learn==1.6.1",
#     "seaborn==0.13.2",
#     "pillow",
#     "qrcode==8.2",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.Html('''
    <style>
        body, .marimo-container {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        /* Main layout container - use full viewport height */
        body {
            height: 100vh;
            overflow: hidden;
        }

        .marimo-container {
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* Header section - fixed height */
        .app-header {
            flex-shrink: 0;
            padding: 10px 0;
        }

        /* Dashboard section - fixed height with internal scrolling if needed */
        .app-dashboard {
            flex-shrink: 0;
            max-height: 40vh;
            overflow-y: auto;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 8px;
            margin: 0 auto 10px auto;
            max-width: 90%;
        }

        /* Plot section - takes remaining space */
        .app-plot {
            flex-grow: 1;
            flex-shrink: 1;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            min-height: 0;
        }

        /* Ensure matplotlib figures scale properly */
        .app-plot img,
        .app-plot svg {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
    </style>
    ''')
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import BayesianRidge
    from sklearn.preprocessing import PolynomialFeatures
    from POPSRegression import POPSRegression
    from scipy.linalg import cho_factor, cho_solve

    # Customize default plotting style
    import seaborn as sns
    sns.set_context('talk')
    return (
        BayesianRidge,
        POPSRegression,
        PolynomialFeatures,
        cho_factor,
        cho_solve,
        mo,
        np,
        plt,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    # Demo scenes, chosen with ?scene=<name> in the URL (the links in the Data tab reload the page with it).
    # Each sets the starting values of the controls; everything can still be changed by hand afterwards.
    # The P slider and method checkboxes are deliberately not driven by state (circular dependencies), so a
    # scene sets their defaults at load time rather than changing them live.
    # Scenes 1-5 use near-deterministic labels (sigma = 0.001), as for DFT-trained interatomic potentials, where the
    # error is approximation error; scene 6 adds label noise (sigma = 0.101) to show where the posterior is right.
    # The LJ scenes keep data on the smooth part of the curve, x in [0, 10] (filter range with "keep inside"); x < 0 is
    # the clipped wall.  There P = 6 is clearly misspecified; P = 10 is nearly well specified.
    SCENES = {
        "lj-bayes": dict(title="1 · Bayesian (LJ)", function="lj", filter=(0.0, 10.0), invert=True, P=6, sigma=0.001, methods={"bayesian"}),
        "lj-conformal": dict(title="2 · + Conformal", function="lj", filter=(0.0, 10.0), invert=True, P=6, sigma=0.001,
                             methods={"bayesian", "conformal"}),
        "lj-pops": dict(title="3 · + POPS", function="lj", filter=(0.0, 10.0), invert=True, P=6, sigma=0.001,
                        methods={"bayesian", "conformal", "pops"}),
        "lj-cr3": dict(title="4 · + CR3", function="lj", filter=(0.0, 10.0), invert=True, P=6, sigma=0.001,
                       methods={"bayesian", "conformal", "pops", "cr3"}, samples=True),
        "witch-gap": dict(title="5 · Gap (witch, P = 6)", function="witch", filter=(0.0, 5.0), P=6, sigma=0.001,
                          methods={"conformal", "pops", "cr3"}),
        "sin-noisy": dict(title="6 · Noisy labels: where the posterior is right (sin, P = 14, σ = 0.1)", function="sin",
                          filter=(0.0, 5.0), P=14, sigma=0.101, methods={"bayesian", "conformal", "pops", "cr3"}),
    }
    _name = mo.query_params()["scene"]
    scene = SCENES.get(_name, {})
    scene_links = " · ".join(f"<a href='?scene={k}' target='_self'>{v['title']}</a>" for k, v in SCENES.items())
    scene_bar = mo.md(f"**Demo scenes:** {scene_links} · <a href='?' target='_self'>reset</a>"
                      + (f" — *current: {scene['title']}*" if scene else ""))
    return SCENES, scene, scene_bar


@app.cell
def _(np):
    def g(X, noise_variance, function_type='sin', custom_code=None):
        """Generate data from different ground truth functions"""
        if function_type == 'sin':
            y = np.sin(X)
        elif function_type == 'witch':
            # Witch of Agnesi: 1/(1+x²)
            y = 1.0 / (1.0 + X**2)
        elif function_type == 'sinc':
            # sinc(x) = sin(x)/x, handle x=0 carefully
            y = np.where(X == 0, 1.0, np.sin(X) / X)
        elif function_type == 'step':
            # Step function
            y = np.where(X < 0, -0.5, 0.5)
        elif function_type == 'runge':
            # Runge function: 1/(1+25x²) - pathological for polynomial interpolation
            y = 1.0 / (1.0 + 25.0 * X**2)
        elif function_type == 'lj':
            # Lennard-Jones potential: V(r) = 4ε[(σ/r)^12 - (σ/r)^6]
            # r = x/5 + 1.5 maps x=0 to r=1.5 (at sigma), x=5 to r=2.5, x=10 to r=3.5
            sigma = 1.5
            r = X / 5.0 + 1.5
            r = np.maximum(r, 0.3)  # Avoid singularity
            y = 4.0 * ((sigma / r) ** 12 - (sigma / r) ** 6)
            # Clip extreme values for display
            y = np.clip(y, -1.5, 1.5)
        elif function_type == 'custom':
            # Execute user code in restricted namespace
            import math
            import scipy
            # Safe subset of builtins needed for basic operations
            safe_builtins = {
                'range': range, 'len': len, 'sum': sum, 'min': min, 'max': max,
                'abs': abs, 'round': round, 'int': int, 'float': float,
                'list': list, 'tuple': tuple, 'zip': zip, 'enumerate': enumerate,
                'True': True, 'False': False, 'None': None, 'print': print,
            }
            try:
                namespace = {'np': np, 'scipy': scipy, 'math': math, 'X': X}
                exec(custom_code, {"__builtins__": safe_builtins}, namespace)
                if 'y' not in namespace:
                    raise ValueError("Code must define 'y'")
                y = namespace['y']
                # Validate output shape
                y = np.atleast_1d(np.asarray(y, dtype=float))
                if y.shape != X.shape:
                    y = np.broadcast_to(y, X.shape).copy()
            except Exception:
                # Return NaN on error - will suppress plotting
                y = np.full_like(X, np.nan)
        else:
            y = np.sin(X)  # Default to sin

        return y + np.random.normal(scale=np.sqrt(noise_variance), size=X.shape)
    return (g,)


@app.cell
def _(BayesianRidge, np):
    class MyBayesianRidge(BayesianRidge):
        def predict(self, X, return_std=False, aleatoric=False):
            y_pred = super().predict(X)
            if not return_std:
                return y_pred
            y_var = np.sum((X @ self.sigma_) * X, axis=1)
            if aleatoric:
                y_var += 1.0 / self.alpha_
            return y_pred, np.sqrt(y_var)

        def log_marginal_likelihood(self):
            """Compute log marginal likelihood for the fitted Bayesian Ridge model"""
            if not hasattr(self, 'coef_'):
                raise ValueError("Model must be fitted first")
            if hasattr(self, 'scores_') and len(self.scores_) > 0:
                return self.scores_[-1]
            else:
                return 0.0

        def loo_log_likelihood(self, X, y, aleatoric=False):
            """Compute Leave-One-Out cross-validation log likelihood (per point average)."""
            if not hasattr(self, 'coef_'):
                raise ValueError("Model must be fitted first")

            n = len(y)
            y_pred = X @ self.coef_
            pred_var = np.sum((X @ self.sigma_) * X, axis=1)
            noise_var = 1.0 / self.alpha_
            if aleatoric:
                pred_var = pred_var + noise_var

            h = np.sum((X @ self.sigma_) * X, axis=1) * self.alpha_
            h = np.clip(h, 0, 0.999)

            residuals = y - y_pred
            loo_residuals = residuals / (1 - h)
            loo_var = (pred_var + noise_var) / (1 - h)

            loo_var = np.maximum(loo_var, 1e-10)
            log_lik = -0.5 * (np.log(2 * np.pi * loo_var) + loo_residuals**2 / loo_var)
            return np.mean(log_lik)

        def sample_posterior(self, X, n_samples=10):
            """Draw samples from the posterior predictive distribution."""
            if not hasattr(self, 'coef_'):
                raise ValueError("Model must be fitted first")
            coef_samples = np.random.multivariate_normal(self.coef_, self.sigma_, size=n_samples)
            return X @ coef_samples.T

    class ConformalPrediction(MyBayesianRidge):
        def get_scores(self, X, y, aleatoric=False):
            y_pred, y_std = self.predict(X, return_std=True, rescale=False, aleatoric=aleatoric)
            residuals = (y_pred - y) / y_std
            scores = np.abs(residuals)
            return scores

        def calibrate(self, X_calib, y_calib, zeta=0.05, aleatoric=False):
            scores = self.get_scores(X_calib, y_calib, aleatoric=aleatoric)
            n = float(len(y_calib))
            self.qhat = np.quantile(scores, np.ceil((n+1)*(1.0-zeta))/n, method='higher')
            return self.qhat

        def predict(self, X, return_std=False, aleatoric=False, rescale=True):
            res = super().predict(X, return_std=return_std, aleatoric=aleatoric)
            if not return_std:
                return res
            y_pred, y_std = res
            if rescale:
                y_std = y_std * self.qhat
            return y_pred, y_std
    return ConformalPrediction, MyBayesianRidge


@app.cell
def _(MyBayesianRidge, np):
    class CR3Regression(MyBayesianRidge):
        """Cluster-robust (CR3) delete-one-cluster jackknife covariance on top of Bayesian ridge.

        After the Bayesian ridge fit (precision A = alpha Phi^T Phi + lambda I, A^-1 = sigma_), each cluster k of
        training points (contiguous in x, `cluster_size` points) gives its PRESS pull
            g_k = Psi_k^T (I - H_kk)^-1 rho_k,   H_kk = Psi_k A^-1 Psi_k^T,
        with whitened rows Psi = sqrt(alpha) Phi and residuals rho = sqrt(alpha) (y - Phi theta).  Deleting cluster
        k moves the fit by exactly A^-1 g_k (no refit), and the CR3 covariance is the spread of those shifts,
            Sigma_CR3 = A^-1 [sum_k (g_k - gbar)(g_k - gbar)^T] A^-1.
        It measures how much the fit depends on which data it saw, so unlike the posterior it does not collapse when
        the model is misspecified.  Like any sampling variance it shrinks with N, so it is scaled by one factor
        lambda: the RMS standardised error on the calibration set (lambda = 1 without calibration)."""

        def __init__(self, cluster_size=1, **kw):
            super().__init__(**kw)
            self.cluster_size = cluster_size

        def fit(self, X, y, x_order=None):
            super().fit(X, y)
            n = len(y)
            order = np.argsort(x_order) if x_order is not None else np.arange(n)
            k = max(1, int(self.cluster_size))
            clusters = [order[i:i + k] for i in range(0, n, k)]
            Ai = self.sigma_
            sa = np.sqrt(self.alpha_)
            Psi, rho = sa * X, sa * (y - X @ self.coef_)
            G = []
            for idx in clusters:
                P = Psi[idx]
                H = P @ Ai @ P.T
                G.append(P.T @ np.linalg.solve(np.eye(len(idx)) - H, rho[idx]))
            G = np.array(G)
            self.shifts_ = G @ Ai                       # theta_hat - theta_hat_(-k), one row per cluster
            Gc = G - G.mean(0)
            # Keep the factor L (Sigma_CR3 = L L^T) and form variances as ||L^T x||^2: forming x^T Sigma x from Sigma
            # loses every digit for an ill-conditioned basis (raw polynomials: cond ~ 1e18) and can go negative
            self.L_cr3_ = Ai @ Gc.T
            self.cov_cr3_ = self.L_cr3_ @ self.L_cr3_.T
            self.n_clusters_ = len(clusters)
            self.scale_ = 1.0
            return self

        def calibrate(self, X_calib, y_calib, aleatoric=False):
            """lambda = RMS of (y - y_hat) / sigma_CR3 on held-out data (the noise term enters when aleatoric)."""
            mu, sd = self.predict(X_calib, return_std=True, aleatoric=aleatoric, rescale=False)
            self.scale_ = float(np.sqrt(np.mean(((y_calib - mu) / sd) ** 2)))
            return self.scale_

        def predict(self, X, return_std=False, aleatoric=False, rescale=True):
            y_pred = X @ self.coef_ + self.intercept_
            if not return_std:
                return y_pred
            lam = self.scale_ if rescale else 1.0
            y_var = lam ** 2 * np.sum((X @ self.L_cr3_) ** 2, axis=1)
            if aleatoric:
                y_var = y_var + 1.0 / self.alpha_
            return y_pred, np.sqrt(np.maximum(y_var, 0.0))

        def loo_log_likelihood(self, X, y, aleatoric=False):
            return None                                  # the Bayesian LOO does not apply to the CR3 covariance

        def deletion_fits(self, X, n_max=10):
            """Predictions of the exact leave-one-cluster-out refits (the jackknife replicates)."""
            idx = np.linspace(0, len(self.shifts_) - 1, min(n_max, len(self.shifts_))).astype(int)
            return X @ (self.coef_[None, :] - self.shifts_[idx]).T + self.intercept_
    return (CR3Regression,)




@app.cell
def _(np):
    from sklearn.neural_network import MLPRegressor

    class NeuralNetworkRegression:
        """
        Ensemble of MLPRegressors for uncertainty quantification

        Uses sklearn's MLPRegressor with ensemble for uncertainty estimation
        """
        def __init__(self, hidden_layer_sizes=(20,), alpha=0.001, n_ensemble=5, max_iter=1000, tol=1e-3, activation='tanh', ensemble_method='seed'):
            """
            Parameters:
            -----------
            hidden_layer_sizes : tuple, default=(20,)
                Number of neurons in each hidden layer, e.g., (20,) or (20, 10)
            alpha : float, default=0.001
                L2 regularization strength
            n_ensemble : int, default=5
                Number of networks in ensemble for uncertainty estimation
            max_iter : int, default=1000
                Maximum number of iterations for LBFGS solver
            tol : float, default=1e-3
                Tolerance for optimization (relaxed for faster convergence)
            """
            self.hidden_layer_sizes = hidden_layer_sizes
            self.alpha = alpha
            self.n_ensemble = n_ensemble
            self.max_iter = max_iter
            self.tol = tol
            self.activation = activation
            self.ensemble_method = ensemble_method
            self.ensemble_ = []
            self.x_mean_ = None
            self.x_std_ = None

        def fit(self, X, y):
            """Fit ensemble of neural networks using either different seeds or bootstrap"""
            # Standardize inputs for better NN training
            self.x_mean_ = np.mean(X, axis=0, keepdims=True)
            self.x_std_ = np.std(X, axis=0, keepdims=True) + 1e-8  # Avoid division by zero
            X_scaled = (X - self.x_mean_) / self.x_std_

            # Train ensemble
            self.ensemble_ = []
            for i in range(self.n_ensemble):
                if self.ensemble_method == 'bootstrap':
                    # Bootstrap: resample data with replacement
                    n_samples = len(X_scaled)
                    indices = np.random.choice(n_samples, size=n_samples, replace=True)
                    X_boot = X_scaled[indices]
                    y_boot = y[indices]
                    # Use same random seed for all bootstrap models
                    random_state = 42
                else:
                    # Random seed: use different random initialization for each model
                    X_boot = X_scaled
                    y_boot = y
                    random_state = i

                model = MLPRegressor(
                    hidden_layer_sizes=self.hidden_layer_sizes,
                    activation=self.activation,
                    solver='lbfgs',
                    alpha=self.alpha,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=random_state,
                    warm_start=False
                )
                model.fit(X_boot, y_boot)
                self.ensemble_.append(model)

            return self

        def predict(self, X, return_std=False, aleatoric=False):
            """
            Predict using ensemble

            Parameters:
            -----------
            X : array-like
                Input data
            return_std : bool, default=False
                If True, return (predictions, std_dev)
            aleatoric : bool, default=False
                If True, add noise variance to uncertainty (not implemented for NN)

            Returns:
            --------
            y_pred : array
                Mean predictions across ensemble
            y_std : array (if return_std=True)
                Standard deviation across ensemble
            """
            # Standardize inputs using training statistics
            X_scaled = (X - self.x_mean_) / self.x_std_

            # Get predictions from all ensemble members
            predictions = np.array([model.predict(X_scaled) for model in self.ensemble_])

            # Compute mean and std across ensemble
            y_pred = np.mean(predictions, axis=0)

            if return_std:
                y_std = np.std(predictions, axis=0)
                # Note: aleatoric uncertainty not implemented for NN ensemble
                # The ensemble variance already captures both epistemic and some aleatoric uncertainty
                return y_pred, y_std
            else:
                return y_pred

        def predict_ensemble(self, X):
            """
            Return predictions from all ensemble members for CRPS computation.

            Parameters:
            -----------
            X : array-like
                Input data

            Returns:
            --------
            ensemble_predictions : array (n_samples, n_ensemble)
                Predictions from each ensemble member
            """
            X_scaled = (X - self.x_mean_) / self.x_std_
            predictions = np.array([model.predict(X_scaled) for model in self.ensemble_])
            # Transpose to get shape (n_samples, n_ensemble)
            return predictions.T
    return (NeuralNetworkRegression,)


@app.cell
def _(np):
    from sklearn.linear_model import QuantileRegressor

    class QuantileRegressionUQ:
        """
        Quantile regression for uncertainty quantification

        Directly predicts prediction intervals by fitting models at different quantiles.
        Non-parametric approach that doesn't assume a specific distribution.
        """
        def __init__(self, confidence=0.9, fit_intercept=False, alpha=1.0):
            """
            Parameters:
            -----------
            confidence : float, default=0.9
                Confidence level for prediction intervals (e.g., 0.9 for 90% intervals)
            fit_intercept : bool, default=False
                Whether to fit intercept (usually False when using polynomial basis)
            alpha : float, default=1.0
                Regularization strength (higher = more regularization)
            """
            self.confidence = confidence
            self.fit_intercept = fit_intercept
            self.significance = 1 - confidence  # Significance level
            self.alpha = alpha  # Regularization parameter

            # Three quantile models: lower, median, upper
            self.model_lower = QuantileRegressor(
                quantile=self.significance/2,
                alpha=alpha,
                solver='highs',
                fit_intercept=fit_intercept
            )
            self.model_median = QuantileRegressor(
                quantile=0.5,
                alpha=alpha,
                solver='highs',
                fit_intercept=fit_intercept
            )
            self.model_upper = QuantileRegressor(
                quantile=1 - self.significance/2,
                alpha=alpha,
                solver='highs',
                fit_intercept=fit_intercept
            )

        def fit(self, X, y):
            """Fit three quantile regression models"""
            import warnings

            # Suppress convergence warnings and catch fit failures
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')  # Suppress all warnings during fit

                try:
                    self.model_lower.fit(X, y)
                    self.model_median.fit(X, y)
                    self.model_upper.fit(X, y)
                    self.fit_succeeded_ = True
                except Exception as e:
                    # If fitting fails, mark as failed
                    # Predictions will fall back to simple mean
                    self.fit_succeeded_ = False
                    self.fallback_mean_ = np.mean(y)
                    self.fallback_std_ = np.std(y)

            return self

        def predict(self, X, return_std=False, aleatoric=False):
            """
            Predict using quantile regression

            Parameters:
            -----------
            X : array-like
                Input features
            return_std : bool, default=False
                If True, return (predictions, std_deviations)
            aleatoric : bool, default=False
                Not used for quantile regression (intervals already include all uncertainty)

            Returns:
            --------
            y_pred : array
                Median predictions
            y_std : array (if return_std=True)
                Approximate standard deviation from prediction interval width
            """
            # If fitting failed, return fallback predictions
            if not self.fit_succeeded_:
                n_samples = X.shape[0]
                y_pred = np.full(n_samples, self.fallback_mean_)
                if return_std:
                    y_std = np.full(n_samples, self.fallback_std_)
                    return y_pred, y_std
                return y_pred

            y_pred = self.model_median.predict(X)

            if not return_std:
                return y_pred

            y_lower = self.model_lower.predict(X)
            y_upper = self.model_upper.predict(X)

            # Approximate std from interval width
            # For normal distribution, 90% interval is roughly ±1.645σ
            # So interval width ≈ 2*1.645σ → σ ≈ width / 3.29
            # For 95%: width / 3.92, for 80%: width / 2.56
            from scipy.stats import norm
            z_score = norm.ppf(1 - self.significance/2)
            y_std = (y_upper - y_lower) / (2 * z_score)

            return y_pred, y_std

        def predict_quantiles(self, X):
            """
            Return the raw quantile predictions for CRPS computation.

            Returns:
            --------
            y_lower : array - lower quantile predictions
            y_median : array - median predictions
            y_upper : array - upper quantile predictions
            alpha : float - significance level (1 - confidence)
            """
            if not self.fit_succeeded_:
                n_samples = X.shape[0]
                return (np.full(n_samples, self.fallback_mean_),
                        np.full(n_samples, self.fallback_mean_),
                        np.full(n_samples, self.fallback_mean_),
                        self.significance)

            y_lower = self.model_lower.predict(X)
            y_median = self.model_median.predict(X)
            y_upper = self.model_upper.predict(X)

            return y_lower, y_median, y_upper, self.significance
    return (QuantileRegressionUQ,)


@app.cell
def _(cho_factor, cho_solve, np):
    # Pure numpy/scipy GP implementation for WebAssembly compatibility

    def bump_kernel(X1, X2, lengthscale, support_radius, signal_variance=1.0):
        """Wendland C2 compactly supported kernel"""
        dists = np.abs(X1[:, None] - X2[None, :])
        r = dists / (lengthscale * support_radius)
        r_clipped = np.clip(r, 0.0, 1.0)
        K = signal_variance * np.where(r < 1.0, (1.0 - r_clipped)**4 * (4.0 * r_clipped + 1.0), 0.0)
        return K

    def polynomial_kernel(X1, X2, degree, sigma):
        """Polynomial kernel"""
        # Normalize inputs
        x1_norm = X1 * 0.1
        x2_norm = X2 * 0.1
        K = (sigma + x1_norm[:, None] * x2_norm[None, :])**degree
        return K

    def rbf_kernel(X1, X2, lengthscale, signal_variance=1.0):
        """RBF/squared exponential kernel"""
        dists_sq = (X1[:, None] - X2[None, :])**2
        K = signal_variance * np.exp(-0.5 * dists_sq / lengthscale**2)
        return K

    def matern_kernel(X1, X2, lengthscale, signal_variance=1.0, nu=1.5):
        """Matérn kernel with nu=1.5 (Matérn-3/2)"""
        dists = np.abs(X1[:, None] - X2[None, :])
        r = dists / lengthscale
        sqrt3_r = np.sqrt(3.0) * r
        K = signal_variance * (1.0 + sqrt3_r) * np.exp(-sqrt3_r)
        return K

    def custom_kernel(X1, X2, lengthscale, signal_variance, custom_code):
        """Execute user-defined kernel code."""
        import math
        import scipy

        safe_builtins = {
            'range': range, 'len': len, 'sum': sum, 'min': min, 'max': max,
            'abs': abs, 'round': round, 'int': int, 'float': float,
            'list': list, 'tuple': tuple, 'zip': zip, 'enumerate': enumerate,
            'True': True, 'False': False, 'None': None, 'print': print,
        }

        try:
            namespace = {
                'np': np, 'scipy': scipy, 'math': math,
                'X1': np.atleast_1d(X1), 'X2': np.atleast_1d(X2),
                'lengthscale': lengthscale, 'signal_variance': signal_variance
            }
            exec(custom_code, {"__builtins__": safe_builtins}, namespace)

            if 'K' not in namespace:
                return None, "Code must define 'K'"

            K = np.atleast_2d(np.asarray(namespace['K'], dtype=float))

            expected_shape = (len(np.atleast_1d(X1)), len(np.atleast_1d(X2)))
            if K.shape != expected_shape:
                return None, f"K shape {K.shape} doesn't match expected {expected_shape}"

            if np.any(np.isnan(K)) or np.any(np.isinf(K)):
                return None, "Kernel matrix contains NaN or Inf values"

            return K, None

        except Exception as e:
            return None, str(e)

    def compute_kernel_matrix(X1, X2, kernel_type, params):
        """Helper to compute kernel matrix given type and params"""
        if kernel_type == 'custom':
            K, error = custom_kernel(
                X1, X2,
                params.get('lengthscale', 1.0),
                params.get('signal_variance', 1.0),
                params.get('custom_code', '')
            )
            if error:
                raise ValueError(f"Custom kernel error: {error}")
            return K
        elif kernel_type == 'bump':
            return bump_kernel(X1, X2, params['lengthscale'], params['support_radius'],
                              params.get('signal_variance', 1.0))
        elif kernel_type == 'polynomial':
            return polynomial_kernel(X1, X2, params['degree'], params['sigma'])
        elif kernel_type == 'rbf':
            return rbf_kernel(X1, X2, params['lengthscale'], params.get('signal_variance', 1.0))
        elif kernel_type == 'matern':
            return matern_kernel(X1, X2, params['lengthscale'], params.get('signal_variance', 1.0))
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def gp_marginal_likelihood(X_train, y_train, K_train, noise):
        """
        Compute log marginal likelihood

        log p(y|X) = -0.5 * y^T K^{-1} y - 0.5 * log|K| - n/2 * log(2π)
        """
        n = len(y_train)
        K_noisy = K_train + noise * np.eye(n)

        try:
            L, lower = cho_factor(K_noisy, lower=True)
        except np.linalg.LinAlgError:
            return -np.inf

        # y^T K^{-1} y
        alpha = cho_solve((L, lower), y_train)
        data_fit = -0.5 * np.dot(y_train, alpha)

        # log|K|
        complexity = -np.sum(np.log(np.diag(L)))

        # constant
        constant = -0.5 * n * np.log(2 * np.pi)

        return data_fit + complexity + constant

    def gp_joint_marginal_likelihood(X_train, y_train, K_train, noise, Phi_train, mean_regularization_strength):
        """
        Compute log marginal likelihood with joint Bayesian inference over basis mean coefficients

        Model: y = Φβ + f + ε
        where β ~ N(0, λ⁻¹I), f ~ GP(0, K), ε ~ N(0, σ²I)

        Marginal: y ~ N(0, K_total) where K_total = ΦΛ⁻¹Φᵀ + K + σ²I

        log p(y|X) = -0.5 * y^T K_total^{-1} y - 0.5 * log|K_total| - n/2 * log(2π)

        Parameters:
        -----------
        X_train : array (n_train,)
        y_train : array (n_train,)
        K_train : array (n_train, n_train) - GP kernel on training data
        noise : float - observation noise variance
        Phi_train : array (n_train, n_features) - pre-computed basis feature matrix
        mean_regularization_strength : float - regularization strength for basis coefficients

        Returns:
        --------
        log_marginal_likelihood : float
        """
        n = len(y_train)

        if Phi_train is None:
            # No basis mean - should not happen, but handle gracefully
            return gp_marginal_likelihood(X_train, y_train, K_train, noise)

        # Normalize basis features for numerical stability
        Phi_train_std = np.std(Phi_train, axis=0, keepdims=True)
        Phi_train_std[Phi_train_std < 1e-10] = 1.0
        Phi_train_normalized = Phi_train / Phi_train_std

        # Prior precision for basis coefficients
        lambda_prior = mean_regularization_strength
        Lambda_inv = (1.0 / lambda_prior) * np.eye(Phi_train_normalized.shape[1])

        # Adjust for normalization
        Lambda_inv_normalized = Lambda_inv / (Phi_train_std.T @ Phi_train_std + 1e-10)

        # Build total covariance: K_total = ΦΛ⁻¹Φᵀ + K_GP + σ²I
        K_beta = Phi_train_normalized @ Lambda_inv_normalized @ Phi_train_normalized.T
        K_total = K_beta + K_train + noise * np.eye(n)

        # Add small jitter for numerical stability
        jitter = 1e-8
        K_total += jitter * np.eye(n)

        try:
            L, lower = cho_factor(K_total, lower=True)
        except np.linalg.LinAlgError:
            return -np.inf

        # y^T K_total^{-1} y
        alpha = cho_solve((L, lower), y_train)
        data_fit = -0.5 * np.dot(y_train, alpha)

        # log|K_total|
        complexity = -np.sum(np.log(np.diag(L)))

        # constant
        constant = -0.5 * n * np.log(2 * np.pi)

        return data_fit + complexity + constant

    def gp_predict(X_train, y_train, X_test, K_train, K_test_train, K_test, noise):
        """
        GP prediction using Cholesky decomposition

        Parameters:
        -----------
        X_train : array (n_train,)
        y_train : array (n_train,)
        X_test : array (n_test,)
        K_train : array (n_train, n_train) - kernel on training data
        K_test_train : array (n_test, n_train) - cross kernel
        K_test : array (n_test, n_test) - kernel on test data
        noise : float - observation noise variance

        Returns:
        --------
        y_mean : array (n_test,)
        y_std : array (n_test,)
        """
        n_train = len(X_train)

        # Add noise to diagonal
        K_noisy = K_train + noise * np.eye(n_train)

        # Cholesky decomposition
        try:
            L, lower = cho_factor(K_noisy, lower=True)
        except np.linalg.LinAlgError:
            # Fallback: add more noise
            K_noisy += 1e-3 * np.eye(n_train)
            L, lower = cho_factor(K_noisy, lower=True)

        # Solve for alpha = K^{-1} y
        alpha = cho_solve((L, lower), y_train)

        # Predictive mean
        y_mean = K_test_train @ alpha

        # Predictive variance
        v = cho_solve((L, lower), K_test_train.T)
        y_var = np.diag(K_test) - np.sum(K_test_train * v.T, axis=1)
        y_var = np.maximum(y_var, 0.0)  # Ensure non-negative
        y_std = np.sqrt(y_var)

        return y_mean, y_std

    def gp_sample_posterior(X_train, y_train, X_test, K_train, K_test_train, K_test, noise, n_samples=10):
        """Draw samples from the GP posterior predictive distribution."""
        n_train = len(X_train)
        n_test = len(X_test)

        K_noisy = K_train + noise * np.eye(n_train)
        try:
            L_train, lower = cho_factor(K_noisy, lower=True)
        except np.linalg.LinAlgError:
            K_noisy += 1e-3 * np.eye(n_train)
            L_train, lower = cho_factor(K_noisy, lower=True)

        alpha = cho_solve((L_train, lower), y_train)
        y_mean = K_test_train @ alpha

        v = cho_solve((L_train, lower), K_test_train.T)
        K_post = K_test - K_test_train @ v
        K_post += 1e-6 * np.eye(n_test)
        K_post = 0.5 * (K_post + K_post.T)

        try:
            L_post = np.linalg.cholesky(K_post)
        except np.linalg.LinAlgError:
            K_post += 1e-4 * np.eye(n_test)
            try:
                L_post = np.linalg.cholesky(K_post)
            except np.linalg.LinAlgError:
                return np.tile(y_mean[:, None], (1, n_samples))

        z = np.random.randn(n_test, n_samples)
        return y_mean[:, None] + L_post @ z

    def gp_loo_log_likelihood(y_train, K_train, noise):
        """
        Compute Leave-One-Out cross-validation log likelihood for GP (per point average).

        Uses efficient closed-form formula:
        - α = K^{-1} y
        - K_inv_ii = diagonal of K^{-1}
        - LOO mean: μ_i = y_i - α_i / K_inv_ii
        - LOO variance: σ²_i = 1 / K_inv_ii

        Parameters:
        -----------
        y_train : array (n,) - training targets
        K_train : array (n, n) - kernel matrix on training data
        noise : float - observation noise variance

        Returns:
        --------
        float - average LOO log likelihood per point
        """
        n = len(y_train)

        # Add noise to kernel
        K_noisy = K_train + noise * np.eye(n)

        try:
            # Compute K^{-1} and K^{-1}y
            L, lower = cho_factor(K_noisy, lower=True)
            alpha = cho_solve((L, lower), y_train)

            # Compute diagonal of K^{-1}
            K_inv = cho_solve((L, lower), np.eye(n))
            K_inv_diag = np.diag(K_inv)

            # Ensure positive diagonal (numerical stability)
            K_inv_diag = np.maximum(K_inv_diag, 1e-10)

            # LOO predictions
            loo_mean = y_train - alpha / K_inv_diag
            loo_var = 1.0 / K_inv_diag

            # LOO residuals
            loo_residuals = y_train - loo_mean

            # Gaussian log likelihood
            log_lik = -0.5 * (np.log(2 * np.pi * loo_var) + loo_residuals**2 / loo_var)

            return np.mean(log_lik)

        except np.linalg.LinAlgError:
            return -np.inf

    def optimize_gp_hyperparameters(X_train, y_train, kernel_type, initial_params, max_iter=50,
                                     use_basis_mean=False, Phi_train=None, joint_inference=False,
                                     mean_regularization_strength=0.1, custom_code=None):
        """
        Optimize GP hyperparameters by maximizing marginal likelihood

        Parameters:
        -----------
        X_train : array (n_train,)
        y_train : array (n_train,)
        kernel_type : str
        initial_params : dict with initial hyperparameter values
        max_iter : int, maximum optimization iterations
        use_basis_mean : bool, whether to use basis mean function
        Phi_train : array (n_train, n_basis) or None - pre-computed basis features
        joint_inference : bool, if True optimize mean regularization jointly
        mean_regularization_strength : float, initial mean regularization strength
        custom_code : str or None, custom kernel code (required if kernel_type='custom')

        Returns:
        --------
        optimized_params : dict with optimized hyperparameters
        log_marginal_likelihood : float
        """
        from scipy.optimize import minimize

        X_train = np.atleast_1d(X_train).reshape(-1)
        y_train = np.atleast_1d(y_train).reshape(-1)

        # Define bounds and parameterization (work in log space for positive params)
        if kernel_type == 'rbf':
            if use_basis_mean and joint_inference and Phi_train is not None:
                # Optimize: log_lengthscale, log_signal_variance, log_noise, log_mean_regularization
                def pack_params(lengthscale, signal_variance, noise, mean_reg):
                    return np.array([np.log(lengthscale), np.log(signal_variance),
                                   np.log(noise), np.log(mean_reg)])

                def unpack_params(x):
                    return {'lengthscale': np.exp(x[0]), 'signal_variance': np.exp(x[1]),
                           'noise': np.exp(x[2]), 'mean_regularization': np.exp(x[3])}

                x0 = pack_params(initial_params.get('lengthscale', 1.0),
                               initial_params.get('signal_variance', 1.0),
                               initial_params.get('noise', 0.1),
                               mean_regularization_strength)
                bounds = [(-2, 2), (-2, 2), (-6, 0), (-6, 1)]
            else:
                # Optimize: log_lengthscale, log_signal_variance, log_noise
                def pack_params(lengthscale, signal_variance, noise):
                    return np.array([np.log(lengthscale), np.log(signal_variance), np.log(noise)])

                def unpack_params(x):
                    return {'lengthscale': np.exp(x[0]), 'signal_variance': np.exp(x[1]),
                           'noise': np.exp(x[2])}

                x0 = pack_params(initial_params.get('lengthscale', 1.0),
                               initial_params.get('signal_variance', 1.0),
                               initial_params.get('noise', 0.1))
                bounds = [(-2, 2), (-2, 2), (-6, 0)]

        elif kernel_type == 'matern':
            # Same hyperparameters as RBF: lengthscale, signal_variance and noise
            if use_basis_mean and joint_inference and Phi_train is not None:
                def pack_params(lengthscale, signal_variance, noise, mean_reg):
                    return np.array([np.log(lengthscale), np.log(signal_variance),
                                   np.log(noise), np.log(mean_reg)])

                def unpack_params(x):
                    return {'lengthscale': np.exp(x[0]), 'signal_variance': np.exp(x[1]),
                           'noise': np.exp(x[2]), 'mean_regularization': np.exp(x[3])}

                x0 = pack_params(initial_params.get('lengthscale', 1.0),
                               initial_params.get('signal_variance', 1.0),
                               initial_params.get('noise', 0.1),
                               mean_regularization_strength)
                bounds = [(-2, 2), (-2, 2), (-6, 0), (-6, 1)]
            else:
                def pack_params(lengthscale, signal_variance, noise):
                    return np.array([np.log(lengthscale), np.log(signal_variance), np.log(noise)])

                def unpack_params(x):
                    return {'lengthscale': np.exp(x[0]), 'signal_variance': np.exp(x[1]),
                           'noise': np.exp(x[2])}

                x0 = pack_params(initial_params.get('lengthscale', 1.0),
                               initial_params.get('signal_variance', 1.0),
                               initial_params.get('noise', 0.1))
                bounds = [(-2, 2), (-2, 2), (-6, 0)]

        elif kernel_type == 'bump':
            if use_basis_mean and joint_inference and Phi_train is not None:
                def pack_params(lengthscale, signal_variance, support_radius, noise, mean_reg):
                    return np.array([np.log(lengthscale), np.log(signal_variance),
                                   np.log(support_radius), np.log(noise), np.log(mean_reg)])

                def unpack_params(x):
                    return {'lengthscale': np.exp(x[0]), 'signal_variance': np.exp(x[1]),
                           'support_radius': np.exp(x[2]), 'noise': np.exp(x[3]),
                           'mean_regularization': np.exp(x[4])}

                x0 = pack_params(initial_params.get('lengthscale', 1.0),
                               initial_params.get('signal_variance', 1.0),
                               initial_params.get('support_radius', 2.0),
                               initial_params.get('noise', 0.1),
                               mean_regularization_strength)
                bounds = [(-2, 2), (-2, 2), (-1, 2), (-6, 0), (-6, 1)]
            else:
                def pack_params(lengthscale, signal_variance, support_radius, noise):
                    return np.array([np.log(lengthscale), np.log(signal_variance),
                                   np.log(support_radius), np.log(noise)])

                def unpack_params(x):
                    return {'lengthscale': np.exp(x[0]), 'signal_variance': np.exp(x[1]),
                           'support_radius': np.exp(x[2]), 'noise': np.exp(x[3])}

                x0 = pack_params(initial_params.get('lengthscale', 1.0),
                               initial_params.get('signal_variance', 1.0),
                               initial_params.get('support_radius', 2.0),
                               initial_params.get('noise', 0.1))
                bounds = [(-2, 2), (-2, 2), (-1, 2), (-6, 0)]

        elif kernel_type == 'polynomial':
            # Optimize: log_sigma, log_noise (keep degree fixed)
            degree = initial_params.get('degree', 5)

            if use_basis_mean and joint_inference and Phi_train is not None:
                def pack_params(sigma, noise, mean_reg):
                    return np.array([np.log(sigma), np.log(noise), np.log(mean_reg)])

                def unpack_params(x):
                    return {'degree': degree, 'sigma': np.exp(x[0]), 'noise': np.exp(x[1]),
                           'mean_regularization': np.exp(x[2])}

                x0 = pack_params(initial_params.get('sigma', 0.1),
                               initial_params.get('noise', 0.1),
                               mean_regularization_strength)
                bounds = [(-3, 1), (-6, 0), (-6, 1)]
            else:
                def pack_params(sigma, noise):
                    return np.array([np.log(sigma), np.log(noise)])

                def unpack_params(x):
                    return {'degree': degree, 'sigma': np.exp(x[0]), 'noise': np.exp(x[1])}

                x0 = pack_params(initial_params.get('sigma', 0.1),
                               initial_params.get('noise', 0.1))
                bounds = [(-3, 1), (-6, 0)]

        elif kernel_type == 'custom':
            if custom_code is None:
                raise ValueError("custom_code required for custom kernel")

            if use_basis_mean and joint_inference and Phi_train is not None:
                def pack_params(lengthscale, signal_variance, noise, mean_reg):
                    return np.array([np.log(lengthscale), np.log(signal_variance),
                                   np.log(noise), np.log(mean_reg)])

                def unpack_params(x):
                    return {'lengthscale': np.exp(x[0]), 'signal_variance': np.exp(x[1]),
                           'noise': np.exp(x[2]), 'mean_regularization': np.exp(x[3]),
                           'custom_code': custom_code}

                x0 = pack_params(initial_params.get('lengthscale', 1.0),
                               initial_params.get('signal_variance', 1.0),
                               initial_params.get('noise', 0.1),
                               mean_regularization_strength)
                bounds = [(-2, 2), (-2, 2), (-6, 0), (-6, 1)]
            else:
                def pack_params(lengthscale, signal_variance, noise):
                    return np.array([np.log(lengthscale), np.log(signal_variance), np.log(noise)])

                def unpack_params(x):
                    return {'lengthscale': np.exp(x[0]), 'signal_variance': np.exp(x[1]),
                           'noise': np.exp(x[2]), 'custom_code': custom_code}

                x0 = pack_params(initial_params.get('lengthscale', 1.0),
                               initial_params.get('signal_variance', 1.0),
                               initial_params.get('noise', 0.1))
                bounds = [(-2, 2), (-2, 2), (-6, 0)]
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        # Negative log marginal likelihood (to minimize)
        def neg_log_marginal_likelihood(x):
            params = unpack_params(x)
            noise = params.pop('noise')
            mean_reg = params.pop('mean_regularization', None)

            K_train = compute_kernel_matrix(X_train, X_train, kernel_type, params)

            # Use joint marginal likelihood if optimizing mean regularization
            if mean_reg is not None and use_basis_mean and joint_inference and Phi_train is not None:
                log_ml = gp_joint_marginal_likelihood(X_train, y_train, K_train, noise,
                                                     Phi_train, mean_reg)
            else:
                log_ml = gp_marginal_likelihood(X_train, y_train, K_train, noise)

            # Return negative for minimization
            return -log_ml

        # Optimize - use L-BFGS-B with Nelder-Mead fallback for custom kernels
        if kernel_type == 'custom':
            # Custom kernels may have non-smooth objectives; try L-BFGS-B first
            try:
                result = minimize(neg_log_marginal_likelihood, x0, bounds=bounds,
                                 method='L-BFGS-B', options={'maxiter': max_iter})
                if not result.success or not np.isfinite(result.fun):
                    raise RuntimeError("L-BFGS-B failed")
            except Exception:
                # Fallback to derivative-free Nelder-Mead (no native bounds)
                def bounded_objective(x):
                    x_clipped = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
                    return neg_log_marginal_likelihood(x_clipped)
                result = minimize(bounded_objective, x0, method='Nelder-Mead',
                                 options={'maxiter': max_iter * 10, 'xatol': 1e-4, 'fatol': 1e-4})
                # Clip result to bounds
                result.x = np.clip(result.x, [b[0] for b in bounds], [b[1] for b in bounds])
        else:
            result = minimize(neg_log_marginal_likelihood, x0, bounds=bounds,
                             method='L-BFGS-B', options={'maxiter': max_iter})

        optimized_params = unpack_params(result.x)
        log_ml = -result.fun

        return optimized_params, log_ml

    def fit_gp_numpy(X_train, y_train, X_test, kernel_type='rbf', use_basis_mean=False, Phi_train=None, Phi_test=None, joint_inference=False, mean_regularization_strength=0.1, **kernel_params):
        """
        Fit GP using pure numpy/scipy

        Parameters:
        -----------
        X_train : array (n_train,)
        y_train : array (n_train,)
        X_test : array (n_test,)
        kernel_type : str, one of 'rbf', 'bump', 'polynomial'
        use_basis_mean : bool, whether to use basis mean function
        Phi_train : array (n_train, n_basis) or None - pre-computed basis features for training
        Phi_test : array (n_test, n_basis) or None - pre-computed basis features for testing
        joint_inference : bool, if True do joint Bayesian inference over mean params and GP
        **kernel_params : kernel hyperparameters

        Returns:
        --------
        y_mean : array (n_test,) - full GP predictions (with basis mean added back)
        y_std : array (n_test,) - total uncertainty (GP + mean if joint inference)
        basis_mean : array (n_test,) or None - basis mean function (for plotting)
        y_std_gp : array (n_test,) or None - GP uncertainty component only (for joint inference)
        y_std_mean : array (n_test,) or None - mean uncertainty component only (for joint inference)
        """
        X_train = np.atleast_1d(X_train).reshape(-1)
        y_train = np.atleast_1d(y_train).reshape(-1)
        X_test = np.atleast_1d(X_test).reshape(-1)

        # Basis mean function (if requested)
        if use_basis_mean and Phi_train is not None and Phi_test is not None:
            from sklearn.linear_model import BayesianRidge

            if joint_inference:
                # Joint Bayesian inference: marginalize over basis coefficients
                # Model: y = Φβ + f + ε
                # where β ~ N(0, λ⁻¹I), f ~ GP(0, K), ε ~ N(0, σ²I)
                #
                # Marginal: y ~ N(0, ΦΛ⁻¹Φᵀ + K + σ²I)
                # This integrates out uncertainty in β!

                # Prior precision for basis coefficients
                lambda_prior = mean_regularization_strength
                Lambda_inv = (1.0 / lambda_prior) * np.eye(Phi_train.shape[1])

                # Normalize basis features to improve conditioning
                Phi_train_std = np.std(Phi_train, axis=0, keepdims=True)
                Phi_train_std[Phi_train_std < 1e-10] = 1.0  # Avoid division by zero
                Phi_train_normalized = Phi_train / Phi_train_std
                Phi_test_normalized = Phi_test / Phi_train_std

                # Update Lambda_inv to account for normalization
                Lambda_inv_normalized = Lambda_inv / (Phi_train_std.T @ Phi_train_std + 1e-10)

                # Store normalized features for joint inference
                Phi_train = Phi_train_normalized
                Phi_test = Phi_test_normalized
                Lambda_inv = Lambda_inv_normalized

                # Posterior mean will be computed after we have K_total
                use_basis_mean = True
                y_train_residual = None  # Not used in joint path
                mean_test = None  # Will be computed in joint inference path after seeing data
            else:
                # Sequential inference: fit basis model first (point estimate)
                basis_model = BayesianRidge(fit_intercept=False)
                basis_model.fit(Phi_train, y_train)

                # Get basis mean predictions
                mean_train = basis_model.predict(Phi_train)
                mean_test = basis_model.predict(Phi_test)

                # Subtract mean from training data (fit GP on residuals)
                y_train_residual = y_train - mean_train
        else:
            y_train_residual = y_train
            mean_test = np.zeros_like(X_test)
            use_basis_mean = False  # Track if we're using it for return value
            joint_inference = False  # Not using joint inference

        # Get noise level
        noise = kernel_params.get('noise', 0.1)

        # Compute kernel matrices
        if kernel_type == 'bump':
            lengthscale = kernel_params.get('lengthscale', 1.0)
            signal_variance = kernel_params.get('signal_variance', 1.0)
            support_radius = kernel_params.get('support_radius', 2.0)
            noise = max(noise, 1e-4)  # Higher noise for stability

            K_train = bump_kernel(X_train, X_train, lengthscale, support_radius, signal_variance)
            K_test_train = bump_kernel(X_test, X_train, lengthscale, support_radius, signal_variance)
            K_test = bump_kernel(X_test, X_test, lengthscale, support_radius, signal_variance)

        elif kernel_type == 'polynomial':
            degree = min(kernel_params.get('degree', 10), 8)  # Cap at 8
            sigma = kernel_params.get('sigma', 0.1)
            noise = max(noise, 1e-3)  # Higher noise for stability

            K_train = polynomial_kernel(X_train, X_train, degree, sigma)
            K_test_train = polynomial_kernel(X_test, X_train, degree, sigma)
            K_test = polynomial_kernel(X_test, X_test, degree, sigma)

        elif kernel_type == 'rbf':
            lengthscale = kernel_params.get('lengthscale', 1.0)
            signal_variance = kernel_params.get('signal_variance', 1.0)
            noise = max(noise, 1e-6)

            K_train = rbf_kernel(X_train, X_train, lengthscale, signal_variance)
            K_test_train = rbf_kernel(X_test, X_train, lengthscale, signal_variance)
            K_test = rbf_kernel(X_test, X_test, lengthscale, signal_variance)

        elif kernel_type == 'matern':
            lengthscale = kernel_params.get('lengthscale', 1.0)
            signal_variance = kernel_params.get('signal_variance', 1.0)
            noise = max(noise, 1e-6)

            K_train = matern_kernel(X_train, X_train, lengthscale, signal_variance)
            K_test_train = matern_kernel(X_test, X_train, lengthscale, signal_variance)
            K_test = matern_kernel(X_test, X_test, lengthscale, signal_variance)

        elif kernel_type == 'custom':
            lengthscale = kernel_params.get('lengthscale', 1.0)
            signal_variance = kernel_params.get('signal_variance', 1.0)
            custom_code = kernel_params.get('custom_code', '')
            noise = max(noise, 1e-6)

            K_train, err = custom_kernel(X_train, X_train, lengthscale, signal_variance, custom_code)
            if err:
                raise ValueError(f"Custom kernel error: {err}")
            K_test_train, err = custom_kernel(X_test, X_train, lengthscale, signal_variance, custom_code)
            if err:
                raise ValueError(f"Custom kernel error: {err}")
            K_test, err = custom_kernel(X_test, X_test, lengthscale, signal_variance, custom_code)
            if err:
                raise ValueError(f"Custom kernel error: {err}")

        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        try:
            if joint_inference and use_basis_mean:
                # Joint Bayesian inference path
                # Build total covariance: K_total = ΦΛ⁻¹Φᵀ + K_GP + σ²I
                K_beta = Phi_train @ Lambda_inv @ Phi_train.T  # Uncertainty from β
                K_total = K_beta + K_train + noise * np.eye(len(X_train))

                try:
                    # Adaptive jitter for numerical stability
                    # Start with small jitter and increase if needed
                    jitter = 1e-8
                    max_attempts = 5

                    for attempt in range(max_attempts):
                        try:
                            # Solve for α = K_total⁻¹ y
                            L = np.linalg.cholesky(K_total + jitter * np.eye(len(X_train)))
                            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                            break  # Success!
                        except np.linalg.LinAlgError:
                            if attempt < max_attempts - 1:
                                # Increase jitter and try again
                                jitter *= 10
                                continue
                            else:
                                # Final attempt failed
                                raise

                    # Prediction mean: [Φ(x*) k(x*,X)] α
                    # This automatically includes contribution from both β and GP
                    K_test_beta = Phi_test @ Lambda_inv @ Phi_train.T
                    K_combined = K_test_beta + K_test_train
                    y_mean = K_combined @ alpha

                    # Prediction variance: includes both β and GP uncertainty
                    # Σ(x*) = Φ(x*)Λ⁻¹Φ(x*)ᵀ + k(x*,x*) - K_combined K_total⁻¹ K_combinedᵀ
                    v = np.linalg.solve(L, K_combined.T)
                    K_test_beta_diag = np.sum((Phi_test @ Lambda_inv) * Phi_test, axis=1)

                    # Separate the uncertainty components:
                    # 1. Prior uncertainty from polynomial mean (before seeing data)
                    y_var_mean_prior = K_test_beta_diag

                    # 2. Prior uncertainty from GP (before seeing data)
                    y_var_gp_prior = np.diag(K_test)

                    # 3. Reduction in uncertainty from observing data
                    y_var_reduction = np.sum(v**2, axis=0)

                    # Total posterior variance
                    y_var_total = y_var_mean_prior + y_var_gp_prior - y_var_reduction
                    y_var_total = np.maximum(y_var_total, 1e-10)  # Numerical stability
                    y_std_total = np.sqrt(y_var_total)

                    # For visualization: approximate the split between mean and GP contributions
                    # This is approximate because they're coupled through the data
                    fraction_mean = y_var_mean_prior / (y_var_mean_prior + y_var_gp_prior + 1e-10)
                    fraction_gp = y_var_gp_prior / (y_var_mean_prior + y_var_gp_prior + 1e-10)

                    y_std_mean_component = y_std_total * np.sqrt(fraction_mean)
                    y_std_gp_component = y_std_total * np.sqrt(fraction_gp)

                    # Compute posterior mean of polynomial coefficients for visualization
                    # E[β|y] = Λ⁻¹Φᵀ K_total⁻¹ y
                    # This gives us the posterior mean estimate of the polynomial component
                    Phi_beta_coef = Lambda_inv @ Phi_train.T @ alpha
                    mean_test_posterior = Phi_test @ Phi_beta_coef

                    # For plotting: return the polynomial posterior mean for visualization
                    # Note: In joint inference, this represents E[Φβ|y], the posterior mean
                    # of the polynomial component after seeing the data
                    return y_mean, y_std_total, mean_test_posterior, y_std_gp_component, y_std_mean_component

                except np.linalg.LinAlgError:
                    print("Warning: Cholesky failed for joint inference")
                    return np.zeros_like(X_test), np.ones_like(X_test), None, None, None
            else:
                # Sequential inference path: fit GP on residuals
                y_mean_residual, y_std = gp_predict(X_train, y_train_residual, X_test,
                                                    K_train, K_test_train, K_test, noise)

                # Add polynomial mean back to predictions
                y_mean = y_mean_residual + mean_test

                # Check for NaNs
                if np.any(np.isnan(y_mean)) or np.any(np.isnan(y_std)):
                    print(f"Warning: NaNs in GP prediction with {kernel_type} kernel")
                    return np.zeros_like(X_test), np.ones_like(X_test), None, None, None

                # Return polynomial mean for plotting (None if not using it)
                # Sequential inference doesn't separate uncertainty components
                return y_mean, y_std, (mean_test if use_basis_mean and not joint_inference else None), None, None

        except Exception as e:
            print(f"Error in GP fitting with {kernel_type} kernel: {e}")
            return np.zeros_like(X_test), np.ones_like(X_test), None, None, None
    return (
        bump_kernel,
        compute_kernel_matrix,
        custom_kernel,
        fit_gp_numpy,
        gp_loo_log_likelihood,
        gp_marginal_likelihood,
        gp_sample_posterior,
        matern_kernel,
        optimize_gp_hyperparameters,
        polynomial_kernel,
        rbf_kernel,
    )


@app.cell
def _(np):
    """Helper functions for probabilistic metrics: Log Likelihood and CRPS"""
    from scipy.stats import norm

    def gaussian_log_likelihood_per_point(y_true, mu, sigma):
        """
        Compute average log likelihood per data point for Gaussian predictions.

        Parameters:
        -----------
        y_true : array - true values
        mu : array - predicted means
        sigma : array - predicted standard deviations

        Returns:
        --------
        float - average log likelihood per point
        """
        # Avoid log(0) by clamping sigma
        sigma = np.maximum(sigma, 1e-10)
        log_lik = -0.5 * (np.log(2 * np.pi * sigma**2) + (y_true - mu)**2 / sigma**2)
        return np.mean(log_lik)

    def crps_gaussian(y_true, mu, sigma):
        """
        Continuous Ranked Probability Score for Gaussian predictions (closed form).

        Lower CRPS is better (0 is perfect).

        Parameters:
        -----------
        y_true : array - true values
        mu : array - predicted means
        sigma : array - predicted standard deviations

        Returns:
        --------
        float - mean CRPS
        """
        sigma = np.maximum(sigma, 1e-10)
        z = (y_true - mu) / sigma
        crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        return np.mean(crps)

    def crps_quantile_approx(y_true, y_lower, y_median, y_upper, alpha):
        """
        Approximate CRPS using pinball losses at 3 quantiles.

        For quantile regression which doesn't output a full distribution.

        Parameters:
        -----------
        y_true : array - true values
        y_lower : array - lower quantile predictions (alpha/2)
        y_median : array - median predictions (0.5)
        y_upper : array - upper quantile predictions (1 - alpha/2)
        alpha : float - significance level (e.g., 0.1 for 90% intervals)

        Returns:
        --------
        float - approximate mean CRPS
        """
        # Pinball loss for lower quantile (alpha/2)
        tau_low = alpha / 2
        err_low = y_true - y_lower
        pinball_low = np.where(err_low >= 0, tau_low * err_low, (tau_low - 1) * err_low)

        # Pinball loss for median (0.5)
        err_med = y_true - y_median
        pinball_med = np.abs(err_med) * 0.5

        # Pinball loss for upper quantile (1 - alpha/2)
        tau_high = 1 - alpha / 2
        err_high = y_true - y_upper
        pinball_high = np.where(err_high >= 0, tau_high * err_high, (tau_high - 1) * err_high)

        # Average pinball losses as CRPS approximation
        return np.mean(pinball_low + pinball_med + pinball_high) / 3

    def crps_ensemble(y_true, ensemble_predictions):
        """
        CRPS using empirical distribution from ensemble members.

        Parameters:
        -----------
        y_true : array (n_samples,) - true values
        ensemble_predictions : array (n_samples, n_members) - predictions from each ensemble member

        Returns:
        --------
        float - mean CRPS
        """
        n_samples = len(y_true)
        crps_values = np.zeros(n_samples)

        for i in range(n_samples):
            y = y_true[i]
            preds = ensemble_predictions[i]
            # CRPS = E|X - y| - 0.5 * E|X - X'|
            # where X, X' are independent draws from the forecast distribution
            crps_values[i] = np.mean(np.abs(preds - y)) - 0.5 * np.mean(np.abs(preds[:, None] - preds[None, :]))

        return np.mean(crps_values)
    return (
        crps_ensemble,
        crps_gaussian,
        crps_quantile_approx,
        gaussian_log_likelihood_per_point,
    )


@app.cell
def _():
    import qrcode
    import io
    import base64

    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://sciml.warwick.ac.uk/wasm/regression-demo/')
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    qr_base64 = base64.b64encode(buffer.read()).decode()
    return (qr_base64,)


@app.cell
def _(mo, qr_base64):
    mo.Html(f'''
    <div class="app-header">
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 0; padding: 0;">
            <div>
                <p style='font-size: 24px; margin: 0; padding: 0; line-height: 1.3;'><b>Bayesian Regression and UQ Demo</b>
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
    return


@app.cell
def _(np):
    """Basis feature functions for linear regression models."""

    def make_rbf_features(X, n_basis, x_min, x_max, lengthscale=None):
        """Create RBF basis features with evenly spaced centers."""
        centers = np.linspace(x_min, x_max, n_basis)
        if lengthscale is None:
            spacing = (x_max - x_min) / (n_basis - 1) if n_basis > 1 else 1.0
            sigma = spacing * 0.5  # Width parameter
        else:
            sigma = lengthscale
        x = X[:, 0]
        features = np.column_stack([
            np.exp(-((x - c)**2) / (2 * sigma**2)) for c in centers
        ])
        return features, centers, sigma

    def make_fourier_features(X, n_basis, x_min, x_max, lengthscale=None):
        """Create Fourier basis features (1, sin, cos pairs)."""
        if lengthscale is None:
            L = (x_max - x_min) / 2  # Half-period
        else:
            L = lengthscale
        x = X[:, 0]
        features = [np.ones(len(x))]  # Constant term
        freq = 1
        while len(features) < n_basis:
            features.append(np.sin(freq * np.pi * x / L))
            if len(features) < n_basis:
                features.append(np.cos(freq * np.pi * x / L))
            freq += 1
        return np.column_stack(features[:n_basis]), L

    def make_lj_features(X, n_basis, x_min, x_max):
        """Create LJ-inspired basis features: constant, 1/r^12, 1/r^6, and inverse powers."""
        offset = 1.5  # Hard-coded to match LJ ground truth
        x = X[:, 0]
        # Use same mapping as LJ ground truth: r = x/5 + offset
        r = x / 5.0 + offset
        r = np.maximum(r, 0.1)  # Safety clamp

        features = []
        # Order: constant, 1/r^12 (repulsive), 1/r^6 (attractive), then inverse powers
        if n_basis >= 1:
            features.append(np.ones(len(r)))  # Constant for vertical offset
        if n_basis >= 2:
            features.append(1.0 / r**12)  # Repulsive term
        if n_basis >= 3:
            features.append(1.0 / r**6)   # Attractive term
        # Additional inverse power terms: 1/r, 1/r^2, 1/r^3, ...
        power = 1
        while len(features) < n_basis:
            features.append(1.0 / r**power)
            power += 1

        return np.column_stack(features[:n_basis]), offset

    def make_custom_features(X, n_basis, custom_code):
        """Execute user code to create custom basis features."""
        import math
        import scipy
        # Safe subset of builtins needed for basic operations
        safe_builtins = {
            'range': range, 'len': len, 'sum': sum, 'min': min, 'max': max,
            'abs': abs, 'round': round, 'int': int, 'float': float,
            'list': list, 'tuple': tuple, 'zip': zip, 'enumerate': enumerate,
            'True': True, 'False': False, 'None': None, 'print': print,
        }
        try:
            namespace = {'np': np, 'scipy': scipy, 'math': math, 'X': X, 'P': n_basis}
            exec(custom_code, {"__builtins__": safe_builtins}, namespace)
            if 'features' not in namespace:
                return None, "Code must define 'features'"
            features = np.atleast_2d(np.asarray(namespace['features'], dtype=float))
            # Ensure correct shape (n_samples, n_features)
            if features.shape[0] != X.shape[0]:
                features = features.T  # Try transpose
            # Check for NaN/Inf values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                return None, "Features contain NaN or Inf values"
            return features[:, :n_basis], None  # Match return signature
        except Exception as e:
            return None, str(e)  # Return error message instead of crashing

    return make_custom_features, make_fourier_features, make_lj_features, make_rbf_features


@app.cell(hide_code=True)
def _(
    CR3Regression,
    ConformalPrediction,
    MyBayesianRidge,
    N_samples,
    NeuralNetworkRegression,
    POPSRegression,
    PolynomialFeatures,
    QuantileRegressionUQ,
    aleatoric,
    aleatoric_gp,
    basis_type_dropdown,
    bayesian,
    bump_kernel,
    compute_kernel_matrix,
    conformal,
    cr3,
    crps_ensemble,
    crps_gaussian,
    crps_quantile_approx,
    fit_gp_numpy,
    function_dropdown,
    g,
    gaussian_log_likelihood_per_point,
    get_P,
    get_basis_lengthscale,
    get_basis_type,
    get_calib_frac,
    get_cr3_calibrate,
    get_cr3_cluster_size,
    get_custom_basis_code,
    get_custom_function_code,
    get_custom_kernel_code,
    get_filter_invert,
    get_filter_max,
    get_filter_min,
    get_gp_joint_inference,
    get_gp_kernel_type,
    get_gp_lengthscale,
    get_gp_mean_regularization,
    get_gp_signal_variance,
    get_gp_support_radius,
    get_last_enabled_method,
    get_leverage_percentile,
    get_n_posterior_samples,
    get_nn_activation,
    get_nn_ensemble_method,
    get_nn_ensemble_size,
    get_nn_hidden_units,
    get_nn_num_layers,
    get_nn_regularization,
    get_percentile_clipping,
    get_pops_posterior,
    get_quantile_confidence,
    get_quantile_regularization,
    get_seed,
    get_show_samples,
    get_zeta,
    gp_loo_log_likelihood,
    gp_marginal_likelihood,
    gp_sample_posterior,
    gp_regression,
    gp_use_basis_mean,
    make_custom_features,
    make_fourier_features,
    make_lj_features,
    make_rbf_features,
    matern_kernel,
    mo,
    neural_network,
    np,
    plt,
    polynomial_kernel,
    pops,
    quantile,
    rbf_kernel,
    custom_kernel,
    seed,
    sigma,
    train_test_split,
):
    def get_data(N_samples=500, sigma=0.1, function_type='sin', custom_code=None):
        x_train = np.append(np.random.uniform(-10, 10, size=N_samples), np.linspace(-10, 10, 2))
        # Filter: normal = exclude inside range, inverted = keep only inside range
        if get_filter_invert():
            x_train = x_train[(x_train >= get_filter_min()) & (x_train <= get_filter_max())]
        else:
            x_train = x_train[(x_train < get_filter_min()) | (x_train > get_filter_max())]
        x_train = np.sort(x_train)
        y_train = g(x_train, noise_variance=sigma**2, function_type=function_type, custom_code=custom_code)
        X_train = x_train[:, None]

        x_test = np.linspace(-10, 10, 1000)
        if function_type == 'lj':
            x_test = x_test[x_test >= 0.0]    # the smooth LJ curve; x < 0 is the artificial clipped wall
        y_test = g(x_test, 0, function_type=function_type, custom_code=custom_code)
        X_test = x_test[:, None]

        return X_train, y_train, X_test, y_test

    fig, ax = plt.subplots(figsize=(14, 5))
    np.random.seed(seed.value)
    _func_type = function_dropdown.value
    _custom_code = get_custom_function_code() if _func_type == 'custom' else None
    X_data, y_data, X_test, y_test = get_data(N_samples.value, sigma=sigma.value, function_type=_func_type, custom_code=_custom_code)

    # Check if filter removed all data
    mo.stop(len(X_data) < 5, mo.md(f"""
    ⚠️ **Error: Filter range removed all data!**

    The current filter range ({get_filter_min():.1f} to {get_filter_max():.1f}) excludes all data points.

    Please adjust the filter range to include some data points.
    """))

    X_train, X_calib, y_train, y_calib = train_test_split(X_data, y_data, test_size=get_calib_frac(), random_state=get_seed())
    n = len(y_calib)

    # Create basis features based on selected type
    basis_type = basis_type_dropdown.value if basis_type_dropdown else get_basis_type()
    P = get_P()
    x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()

    # Store basis parameters for visualization
    rbf_centers, rbf_sigma, fourier_L, lj_offset = None, None, None, None
    _custom_basis_error = None  # Track custom basis errors
    _custom_kernel_error = None  # Track custom kernel errors

    if basis_type == 'polynomial':
        poly = PolynomialFeatures(degree=P-1, include_bias=True)
        Phi_train = poly.fit_transform(X_train)
        Phi_test = poly.transform(X_test)
        Phi_calib = poly.transform(X_calib)
    elif basis_type == 'rbf':
        _lengthscale = get_basis_lengthscale()
        Phi_train, rbf_centers, rbf_sigma = make_rbf_features(X_train, P, x_min, x_max, _lengthscale)
        Phi_test, _, _ = make_rbf_features(X_test, P, x_min, x_max, _lengthscale)
        Phi_calib, _, _ = make_rbf_features(X_calib, P, x_min, x_max, _lengthscale)
    elif basis_type == 'fourier':
        _lengthscale = get_basis_lengthscale()
        Phi_train, fourier_L = make_fourier_features(X_train, P, x_min, x_max, _lengthscale)
        Phi_test, _ = make_fourier_features(X_test, P, x_min, x_max, _lengthscale)
        Phi_calib, _ = make_fourier_features(X_calib, P, x_min, x_max, _lengthscale)
    elif basis_type == 'lj':
        # Restrict test domain to near training data (avoid singularity and extrapolation issues)
        _valid_mask = X_test[:, 0] >= x_min - 2.0  # 2.0 margin before training data
        X_test = X_test[_valid_mask]
        y_test = y_test[_valid_mask]
        # Compute basis (offset=1.5 is hard-coded in make_lj_features to match ground truth)
        Phi_train, _ = make_lj_features(X_train, P, x_min, x_max)
        Phi_test, _ = make_lj_features(X_test, P, x_min, x_max)
        Phi_calib, _ = make_lj_features(X_calib, P, x_min, x_max)
    elif basis_type == 'custom':
        _custom_basis_code = get_custom_basis_code()
        Phi_train, _custom_basis_error = make_custom_features(X_train, P, _custom_basis_code)
        if Phi_train is None:
            # Set flag to skip fitting, will just show data points
            _custom_basis_error = _custom_basis_error or "Unknown error"
            Phi_train = np.zeros((X_train.shape[0], 1))  # Dummy to avoid crashes
            Phi_test = np.zeros((X_test.shape[0], 1))
            Phi_calib = np.zeros((X_calib.shape[0], 1))
        else:
            _custom_basis_error = None
            Phi_test, _ = make_custom_features(X_test, P, _custom_basis_code)
            Phi_calib, _ = make_custom_features(X_calib, P, _custom_basis_code)

    b = MyBayesianRidge(fit_intercept=False)
    # Try new API with posterior parameter, fall back to old API for WASM compatibility
    try:
        p = POPSRegression(
            fit_intercept=False,
            percentile_clipping=get_percentile_clipping(),
            leverage_percentile=get_leverage_percentile(),
            posterior=get_pops_posterior()
        )
    except TypeError:
        # Older version - try without posterior, then without leverage_percentile
        try:
            p = POPSRegression(
                fit_intercept=False,
                percentile_clipping=get_percentile_clipping(),
                leverage_percentile=get_leverage_percentile()
            )
        except TypeError:
            # Oldest version with only percentile_clipping
            p = POPSRegression(
                fit_intercept=False,
                percentile_clipping=get_percentile_clipping()
            )
    c = ConformalPrediction(fit_intercept=False)
    cr3_model = CR3Regression(cluster_size=get_cr3_cluster_size(), fit_intercept=False)
    cr3_scale = None
    q = QuantileRegressionUQ(confidence=get_quantile_confidence(), fit_intercept=False, alpha=get_quantile_regularization())

    ax.plot(X_test[:, 0], y_test, 'k-', label='Truth')
    ax.plot(X_train[:, 0], y_train, 'b.', label='Train')
    ax.plot(X_calib[:, 0], y_calib, 'c.', label='Calibration')
    ax.axvline(get_filter_min(), ls='--', color='k')
    ax.axvline(get_filter_max(), ls='--', color='k')

    gp_log_ml = 0.0  # Initialize
    bayes_log_ml = 0.0  # Initialize
    gp_sparsity = 0.0  # Initialize GP covariance sparsity

    # Test custom kernel if selected
    if get_gp_kernel_type() == 'custom':
        _test_K, _custom_kernel_error = custom_kernel(
            X_train[:, 0], X_train[:, 0],
            get_gp_lengthscale(), get_gp_signal_variance(),
            get_custom_kernel_code()
        )

    # Dictionary to store predictions and metrics for each method
    import time
    method_predictions = {}  # {label: (y_pred, y_std, fit_time)}

    models_to_plot = []
    # Skip fitting if there's a custom basis error or NaN in training data (custom function error)
    _has_valid_data = not np.any(np.isnan(y_train))

    # Show warning in plot if there's an error
    if _custom_basis_error is not None:
        ax.text(0.5, 0.5, '⚠️ Custom basis error', transform=ax.transAxes,
                fontsize=14, ha='center', va='center', color='orange',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='orange'))
    elif _custom_kernel_error is not None and gp_regression.value:
        ax.text(0.5, 0.5, '⚠️ Custom kernel error', transform=ax.transAxes,
                fontsize=14, ha='center', va='center', color='orange',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='orange'))
    elif not _has_valid_data:
        ax.text(0.5, 0.5, '⚠️ Custom function error', transform=ax.transAxes,
                fontsize=14, ha='center', va='center', color='orange',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='orange'))

    if _custom_basis_error is None and _has_valid_data:
        if bayesian.value:
            models_to_plot.append((b, Phi_train, Phi_test, 'C2', 'Bayesian uncertainty', True))
        if conformal.value:
            models_to_plot.append((c, Phi_train, Phi_test, 'C1', 'Conformal prediction', True))
        if pops.value:
            models_to_plot.append((p, Phi_train, Phi_test, 'C0', 'POPS regression', True))
        if quantile.value:
            models_to_plot.append((q, Phi_train, Phi_test, 'C4', 'Quantile regression', True))
        if cr3.value:
            models_to_plot.append((cr3_model, Phi_train, Phi_test, 'C6', 'CR3 jackknife', True))
        if gp_regression.value and _custom_kernel_error is None:
            models_to_plot.append((None, X_train, X_test, 'C3', 'GP regression', False))
        if neural_network.value:
            models_to_plot.append((None, X_train, X_test, 'C5', 'Neural network', False))

    for model_info in models_to_plot:
        if len(model_info) == 6:
            model, X_train_model, X_test_model, color, label, use_poly = model_info
        else:
            continue

        # Start timing for this method
        start_time = time.time()

        if label == 'GP regression':
            # Fit GP with numpy
            y_pred, y_std, poly_mean, y_std_gp, y_std_mean = fit_gp_numpy(
                X_train_model[:, 0], y_train, X_test_model[:, 0],
                kernel_type=get_gp_kernel_type(),
                lengthscale=get_gp_lengthscale(),
                signal_variance=get_gp_signal_variance(),
                support_radius=get_gp_support_radius(),
                degree=get_P(),
                sigma=1.0,
                noise=sigma.value**2,
                use_basis_mean=gp_use_basis_mean.value,
                Phi_train=Phi_train if gp_use_basis_mean.value else None,
                Phi_test=Phi_test if gp_use_basis_mean.value else None,
                joint_inference=get_gp_joint_inference(),
                mean_regularization_strength=get_gp_mean_regularization(),
                custom_code=get_custom_kernel_code() if get_gp_kernel_type() == 'custom' else None
            )

            # Compute log marginal likelihood for display
            params = {
                'lengthscale': get_gp_lengthscale(),
                'signal_variance': get_gp_signal_variance(),
                'support_radius': get_gp_support_radius(),
                'degree': get_P(),
                'sigma': 1.0,
                'custom_code': get_custom_kernel_code() if get_gp_kernel_type() == 'custom' else None
            }
            K_train = compute_kernel_matrix(X_train_model[:, 0], X_train_model[:, 0],
                                           get_gp_kernel_type(), params)
            gp_log_ml = gp_marginal_likelihood(X_train_model[:, 0], y_train, K_train, sigma.value**2)

            # Calculate sparsity of covariance matrix (% of elements < 1e-6)
            gp_sparsity = 100.0 * np.sum(np.abs(K_train) < 1e-6) / K_train.size

            # Plot polynomial mean function if enabled (to show partitioning)
            if poly_mean is not None:
                ax.plot(X_test[:, 0], poly_mean, 'C3--', lw=2, label='GP prior mean', alpha=0.7)

            # Plot separate uncertainty components if joint inference is enabled
            if get_gp_joint_inference() and y_std_mean is not None:
                # Plot total uncertainty (darker shade)
                ax.fill_between(X_test[:, 0], y_pred - y_std, y_pred + y_std,
                               alpha=0.35, color='purple', label='Total uncertainty')
                # Plot polynomial mean uncertainty component (lighter shade)
                ax.fill_between(X_test[:, 0], y_pred - y_std_mean, y_pred + y_std_mean,
                               alpha=0.25, color='orange', label='Mean uncertainty')

            if aleatoric_gp.value:
                # Add aleatoric noise to GP predictions
                y_std = np.sqrt(y_std**2 + sigma.value**2)
        elif label == 'Neural network':
            # Fit neural network ensemble
            # Build hidden layer architecture
            num_layers = get_nn_num_layers()
            hidden_units = get_nn_hidden_units()
            hidden_layer_sizes = tuple([hidden_units] * num_layers)

            # Create and fit neural network
            nn = NeuralNetworkRegression(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=10**get_nn_regularization(),  # Convert from log10 scale
                n_ensemble=get_nn_ensemble_size(),
                activation=get_nn_activation(),
                ensemble_method=get_nn_ensemble_method()
                # Uses default max_iter=1000, tol=1e-3 for reliable convergence
            )
            nn.fit(X_train_model, y_train)

            # Get predictions with uncertainty
            y_pred, y_std = nn.predict(X_test_model, return_std=True)
        else:
            # Existing polynomial-basis models
            if label == 'CR3 jackknife':
                model.fit(X_train_model, y_train, x_order=X_train[:, 0])   # clusters contiguous in x
                if get_cr3_calibrate() and len(y_calib) > 1:
                    cr3_scale = model.calibrate(Phi_calib, y_calib, aleatoric=aleatoric.value)
                else:
                    cr3_scale = 1.0
            else:
                model.fit(X_train_model, y_train)

            # Store log ML for Bayesian models
            if label in ['Bayesian uncertainty', 'Conformal prediction']:
                bayes_log_ml = model.log_marginal_likelihood()

            kwargs = {
                'return_std': True,
                'aleatoric': aleatoric.value,
            }        
            if label == 'Conformal prediction':
                qhat = model.calibrate(Phi_calib, y_calib, zeta=get_zeta(), aleatoric=aleatoric.value)
                kwargs['rescale'] = True

            if label == 'POPS regression':
                y_pred, y_std, y_min, y_max = model.predict(X_test_model, return_std=True, return_bounds=True)
                # Also get bounds at train+calib points for coverage calculation
                _, _, y_min_train, y_max_train = model.predict(Phi_train, return_std=True, return_bounds=True)
                _, _, y_min_calib, y_max_calib = model.predict(Phi_calib, return_std=True, return_bounds=True)
                if aleatoric.value:
                    # Add aleatoric uncertainty to all bounds
                    aleatoric_std = np.sqrt(1.0 / model.alpha_)
                    y_std = np.sqrt(y_std**2 + aleatoric_std**2)
                    y_min = y_min - aleatoric_std
                    y_max = y_max + aleatoric_std
                    y_min_train = y_min_train - aleatoric_std
                    y_max_train = y_max_train + aleatoric_std
                    y_min_calib = y_min_calib - aleatoric_std
                    y_max_calib = y_max_calib + aleatoric_std
            elif label == 'Conformal prediction':
                y_pred, y_std = model.predict(X_test_model, **kwargs)
                # Conformal bounds are y_pred ± y_std (already rescaled by qhat)
                y_min = y_pred - y_std
                y_max = y_pred + y_std
                # Get bounds at train+calib points
                y_pred_train, y_std_train = model.predict(Phi_train, **kwargs)
                y_pred_calib_conf, y_std_calib_conf = model.predict(Phi_calib, **kwargs)
                y_min_train = y_pred_train - y_std_train
                y_max_train = y_pred_train + y_std_train
                y_min_calib = y_pred_calib_conf - y_std_calib_conf
                y_max_calib = y_pred_calib_conf + y_std_calib_conf
            else:
                y_pred, y_std = model.predict(X_test_model, **kwargs)

        # Store predictions and timing for metrics computation
        # Also get predictions at train+calib points for coverage calculation
        fit_time = time.time() - start_time

        if label == 'POPS regression':
            # POPS: use min/max bounds for coverage
            # POPS is external, may not have LOO method
            extra_data = {}
            if hasattr(model, 'loo_log_likelihood'):
                extra_data['loo_log_lik'] = model.loo_log_likelihood(Phi_train, y_train, aleatoric=aleatoric.value)
            method_predictions[label] = (y_pred, y_std, fit_time, y_min, y_max, y_min_train, y_max_train, y_min_calib, y_max_calib, extra_data)
        elif label == 'Conformal prediction':
            # Conformal: use min/max bounds for coverage (already computed above)
            # Conformal inherits from MyBayesianRidge, has LOO method
            extra_data = {}
            if hasattr(model, 'loo_log_likelihood'):
                extra_data['loo_log_lik'] = model.loo_log_likelihood(Phi_train, y_train, aleatoric=aleatoric.value)
            method_predictions[label] = (y_pred, y_std, fit_time, y_min, y_max, y_min_train, y_max_train, y_min_calib, y_max_calib, extra_data)
        elif label == 'GP regression':
            # GP: get std at train/calib points
            y_pred_train, y_std_train = fit_gp_numpy(
                X_train[:, 0], y_train, X_train[:, 0],
                kernel_type=get_gp_kernel_type(), lengthscale=get_gp_lengthscale(),
                signal_variance=get_gp_signal_variance(),
                support_radius=get_gp_support_radius(), degree=get_P(), sigma=1.0,
                noise=sigma.value**2, use_basis_mean=gp_use_basis_mean.value,
                Phi_train=Phi_train if gp_use_basis_mean.value else None,
                Phi_test=Phi_train if gp_use_basis_mean.value else None,
                joint_inference=get_gp_joint_inference(),
                mean_regularization_strength=get_gp_mean_regularization(),
                custom_code=get_custom_kernel_code() if get_gp_kernel_type() == 'custom' else None
            )[:2]
            y_pred_calib, y_std_calib = fit_gp_numpy(
                X_train[:, 0], y_train, X_calib[:, 0],
                kernel_type=get_gp_kernel_type(), lengthscale=get_gp_lengthscale(),
                signal_variance=get_gp_signal_variance(),
                support_radius=get_gp_support_radius(), degree=get_P(), sigma=1.0,
                noise=sigma.value**2, use_basis_mean=gp_use_basis_mean.value,
                Phi_train=Phi_train if gp_use_basis_mean.value else None,
                Phi_test=Phi_calib if gp_use_basis_mean.value else None,
                joint_inference=get_gp_joint_inference(),
                mean_regularization_strength=get_gp_mean_regularization(),
                custom_code=get_custom_kernel_code() if get_gp_kernel_type() == 'custom' else None
            )[:2]
            # Add aleatoric uncertainty if GP aleatoric checkbox is enabled
            if aleatoric_gp.value:
                y_std_train = np.sqrt(y_std_train**2 + sigma.value**2)
                y_std_calib = np.sqrt(y_std_calib**2 + sigma.value**2)
            # Compute LOO for GP using K_train computed earlier
            gp_loo = gp_loo_log_likelihood(y_train, K_train, sigma.value**2)
            extra_data = {'loo_log_lik': gp_loo}
            method_predictions[label] = (y_pred, y_std, fit_time, None, None, y_pred_train, y_std_train, y_pred_calib, y_std_calib, extra_data)
        elif label == 'Neural network':
            # NN: get std at train/calib points
            y_pred_train, y_std_train = nn.predict(X_train, return_std=True)
            y_pred_calib, y_std_calib = nn.predict(X_calib, return_std=True)
            # Store ensemble predictions for CRPS computation
            extra_data = {'ensemble_preds': nn.predict_ensemble(X_test_model)}
            method_predictions[label] = (y_pred, y_std, fit_time, None, None, y_pred_train, y_std_train, y_pred_calib, y_std_calib, extra_data)
        elif label == 'Quantile regression':
            # Quantile regression: store quantile predictions for CRPS
            y_pred_train, y_std_train = model.predict(Phi_train, return_std=True, aleatoric=aleatoric.value)
            y_pred_calib, y_std_calib = model.predict(Phi_calib, return_std=True, aleatoric=aleatoric.value)
            y_lower, y_median, y_upper, q_alpha = model.predict_quantiles(X_test_model)
            extra_data = {'y_lower': y_lower, 'y_median': y_median, 'y_upper': y_upper, 'alpha': q_alpha}
            method_predictions[label] = (y_pred, y_std, fit_time, None, None, y_pred_train, y_std_train, y_pred_calib, y_std_calib, extra_data)
        else:
            # Other polynomial-basis methods (Bayesian, POPS, Conformal): Gaussian distribution
            y_pred_train, y_std_train = model.predict(Phi_train, return_std=True, aleatoric=aleatoric.value)
            y_pred_calib, y_std_calib = model.predict(Phi_calib, return_std=True, aleatoric=aleatoric.value)
            # Compute LOO for methods that support it
            extra_data = {}
            if hasattr(model, 'loo_log_likelihood'):
                extra_data['loo_log_lik'] = model.loo_log_likelihood(Phi_train, y_train, aleatoric=aleatoric.value)
            method_predictions[label] = (y_pred, y_std, fit_time, None, None, y_pred_train, y_std_train, y_pred_calib, y_std_calib, extra_data)

        ax.plot(X_test[:, 0], y_pred, color=color, lw=3)

        # For POPS and Conformal, shade min/max bounds instead of ±1σ
        if label in ['POPS regression', 'Conformal prediction']:
            ax.fill_between(X_test[:, 0], y_min, y_max, alpha=0.5, color=color, label=label)
        else:
            ax.fill_between(X_test[:, 0], y_pred - y_std, y_pred + y_std, alpha=0.5, color=color, label=label)

        # Draw posterior/ensemble samples if enabled (per-method controls)
        if label in ['Bayesian uncertainty', 'Conformal prediction'] and get_show_samples_linear():
            # Sample from Bayesian posterior
            try:
                samples = model.sample_posterior(X_test_model, n_samples=get_n_samples_linear())
                for i in range(samples.shape[1]):
                    ax.plot(X_test[:, 0], samples[:, i], color=color, alpha=0.4, lw=1)
            except Exception:
                pass  # Skip if sampling fails

        elif label == 'CR3 jackknife' and get_show_samples_linear():
            # The exact leave-one-cluster-out refits: the jackknife replicates whose spread is the CR3 covariance
            try:
                _reps = model.deletion_fits(X_test_model, n_max=get_n_samples_linear())
                for i in range(_reps.shape[1]):
                    ax.plot(X_test[:, 0], _reps[:, i], color=color, alpha=0.4, lw=1)
            except Exception:
                pass

        elif label == 'GP regression' and get_show_samples_gp():
            # Sample from GP posterior
            try:
                _gp_params = {
                    'lengthscale': get_gp_lengthscale(),
                    'signal_variance': get_gp_signal_variance(),
                    'support_radius': get_gp_support_radius(),
                    'degree': get_P(), 'sigma': 1.0,
                    'custom_code': get_custom_kernel_code() if get_gp_kernel_type() == 'custom' else None
                }
                _K_train = compute_kernel_matrix(X_train[:, 0], X_train[:, 0], get_gp_kernel_type(), _gp_params)
                _K_test_train = compute_kernel_matrix(X_test[:, 0], X_train[:, 0], get_gp_kernel_type(), _gp_params)
                _K_test = compute_kernel_matrix(X_test[:, 0], X_test[:, 0], get_gp_kernel_type(), _gp_params)
                gp_samples = gp_sample_posterior(X_train[:, 0], y_train, X_test[:, 0],
                                                  _K_train, _K_test_train, _K_test,
                                                  sigma.value**2, n_samples=get_n_samples_gp())
                for i in range(gp_samples.shape[1]):
                    ax.plot(X_test[:, 0], gp_samples[:, i], color=color, alpha=0.4, lw=1)
            except Exception:
                pass  # Skip if sampling fails

        elif label == 'Neural network' and get_show_samples_nn():
            # Show ensemble members (limited by samples slider)
            try:
                ensemble_preds = nn.predict_ensemble(X_test_model)
                n_available = ensemble_preds.shape[1]
                n_to_show = min(get_n_samples_nn(), n_available)
                indices = np.random.choice(n_available, n_to_show, replace=False) if n_to_show < n_available else range(n_available)
                for i in indices:
                    ax.plot(X_test[:, 0], ensemble_preds[:, i], color=color, alpha=0.4, lw=1)
            except Exception:
                pass  # Skip if ensemble not available

        elif label == 'POPS regression' and get_show_samples_linear():
            # POPS stores ensemble in posterior_samples with shape (n_features, n_samples)
            # posterior_samples are DEVIATIONS from coef_, so add coef_ to get actual coefficients
            try:
                if hasattr(model, 'posterior_samples'):
                    samples = model.posterior_samples  # (n_features, n_samples)
                    n_available = samples.shape[1]
                    n_to_show = min(get_n_samples_linear(), n_available)
                    indices = np.random.choice(n_available, n_to_show, replace=False) if n_to_show < n_available else range(n_available)
                    for i in indices:
                        sample_coef = model.coef_ + samples[:, i]
                        pops_pred = X_test_model @ sample_coef + model.intercept_
                        ax.plot(X_test[:, 0], pops_pred, color=color, alpha=0.4, lw=1)
            except Exception:
                pass  # POPS may not expose samples in this version

        # Quantile regression: skip (no posterior to sample)

    # No title needed - all info is in the dashboard and outputs bar
    ax.set_xlim(X_test[:, 0].min(), 10)
    if _func_type == 'lj':
        ax.set_ylim(-1.25, 1.1)   # the smooth LJ curve spans about [-1, 0]; the legend sits above it
    else:
        ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    if _func_type == 'lj':
        ax.legend(loc='upper right', ncol=3, fontsize='small')
    else:
        ax.legend(loc='lower left')

    # Apply tight_layout before adding inset (to avoid warning)
    plt.tight_layout()

    # Add inset plot based on most recently enabled method
    linear_methods_active = bayesian.value or conformal.value or pops.value or quantile.value or cr3.value
    last_method = get_last_enabled_method()

    # Determine which inset to show based on last enabled method
    show_gp_inset = last_method == 'gp' and gp_regression.value
    show_nn_inset = last_method == 'nn' and neural_network.value
    show_linear_inset = last_method == 'linear' and linear_methods_active

    if show_gp_inset or show_nn_inset or show_linear_inset:
        # Create inset axes in the lower right
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        # Position: [x, y, width, height] in axes coordinates
        axins = inset_axes(ax, width="12%", height="25%", loc='lower right',
                          bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax.transAxes,
                          borderpad=1.5)

        if show_gp_inset:
            # Plot kernel function
            x_kernel = np.linspace(0, 2.0, 100)
            kernel_type = get_gp_kernel_type()
            K_values = None

            if kernel_type == 'rbf':
                K_values = rbf_kernel(np.array([0.0]), x_kernel, get_gp_lengthscale(), get_gp_signal_variance())[0, :]
            elif kernel_type == 'bump':
                K_values = bump_kernel(np.array([0.0]), x_kernel, get_gp_lengthscale(), get_gp_support_radius(), get_gp_signal_variance())[0, :]
                # Add vertical line for support radius
                axins.axvline(get_gp_support_radius(), color='k', ls='dashed', lw=0.8, alpha=0.5)
            elif kernel_type == 'matern':
                K_values = matern_kernel(np.array([0.0]), x_kernel, get_gp_lengthscale(), get_gp_signal_variance())[0, :]
            elif kernel_type == 'polynomial':
                K_values = polynomial_kernel(np.array([0.0]), x_kernel, get_P(), 1.0)[0, :]
            elif kernel_type == 'custom':
                try:
                    K_result, err = custom_kernel(
                        np.array([0.0]), x_kernel,
                        get_gp_lengthscale(), get_gp_signal_variance(),
                        get_custom_kernel_code()
                    )
                    if K_result is not None:
                        K_values = K_result[0, :]
                    else:
                        axins.text(0.5, 0.5, '⚠️', transform=axins.transAxes, ha='center', fontsize=14)
                except Exception:
                    axins.text(0.5, 0.5, '⚠️', transform=axins.transAxes, ha='center', fontsize=14)

            if K_values is not None:
                axins.plot(x_kernel, K_values, 'C3', lw=1.5)
            axins.set_xlabel('$r$', fontsize=8)
            axins.set_ylabel('$K(0, r)$', fontsize=8)
            axins.set_title(f'Kernel: {kernel_type}', fontsize=9, pad=3)
            axins.tick_params(labelsize=7)
            axins.grid(True, alpha=0.3, linewidth=0.5)

        elif show_nn_inset:
            # Plot activation function
            x_act = np.linspace(-3, 3, 100)
            activation_type = get_nn_activation()

            if activation_type == 'tanh':
                y_act = np.tanh(x_act)
            elif activation_type == 'relu':
                y_act = np.maximum(0, x_act)

            axins.plot(x_act, y_act, 'C5', lw=1.5)
            axins.set_xlabel('$x$', fontsize=8)
            axins.set_ylabel(f'${activation_type}(x)$', fontsize=8)
            axins.set_title(f'Activation: {activation_type}', fontsize=9, pad=3)
            axins.tick_params(labelsize=7)
            axins.grid(True, alpha=0.3, linewidth=0.5)
            axins.axhline(0, color='k', lw=0.5, alpha=0.3)
            axins.axvline(0, color='k', lw=0.5, alpha=0.3)

        elif show_linear_inset:
            # Plot basis functions using the same make_*_features functions
            import matplotlib
            P_degree = get_P()
            cmap = matplotlib.colormaps['viridis']

            # Create inset x range
            if basis_type == 'polynomial':
                x_inset = np.linspace(-1, 1, 200)
            else:
                x_inset = np.linspace(x_min, x_max, 200)
            X_inset = x_inset[:, None]

            # Get basis features using the same functions as the main computation
            if basis_type == 'polynomial':
                _poly_inset = PolynomialFeatures(degree=P_degree-1, include_bias=True)
                Phi_inset = _poly_inset.fit_transform(X_inset)
                axins.set_title(f'Polynomial (P={P_degree})', fontsize=9, pad=3)
            elif basis_type == 'rbf' and rbf_sigma is not None:
                Phi_inset, _, _ = make_rbf_features(X_inset, P_degree, x_min, x_max, rbf_sigma)
                axins.set_title(f'RBF (P={P_degree})', fontsize=9, pad=3)
            elif basis_type == 'fourier' and fourier_L is not None:
                Phi_inset, _ = make_fourier_features(X_inset, P_degree, x_min, x_max, fourier_L)
                axins.set_title(f'Fourier (P={P_degree})', fontsize=9, pad=3)
            elif basis_type == 'lj':
                Phi_inset, _ = make_lj_features(X_inset, P_degree, x_min, x_max)
                axins.set_title(f'LJ (P={P_degree})', fontsize=9, pad=3)
            elif basis_type == 'custom':
                Phi_inset, _ = make_custom_features(X_inset, P_degree, get_custom_basis_code())
                axins.set_title(f'Custom (P={P_degree})', fontsize=9, pad=3)
            else:
                Phi_inset = None

            # Plot each basis function
            if Phi_inset is not None:
                for _j in range(Phi_inset.shape[1]):
                    y_basis = Phi_inset[:, _j]
                    # Normalize for display (scale to [0, 1] range)
                    if np.max(np.abs(y_basis)) > 0:
                        y_basis = y_basis / np.max(np.abs(y_basis))
                    axins.plot(x_inset, y_basis, color=cmap(_j / max(1, P_degree - 1)), lw=0.9, alpha=0.7)

            axins.set_xlabel('$x$', fontsize=8)
            axins.set_ylabel('Basis', fontsize=8)
            axins.tick_params(labelsize=7)
            axins.grid(True, alpha=0.3, linewidth=0.5)

    # Compute metrics for all active methods
    metrics_dict = {}
    for method_label, (method_y_pred, method_y_std, method_fit_time, method_y_min, method_y_max, method_train_data, method_std_train, method_calib_data, method_std_calib, extra_data) in method_predictions.items():
        # Coverage on train+calib points and on all test points
        if method_label in ['POPS regression', 'Conformal prediction'] and method_train_data is not None:
            # Use min/max bounds for POPS/Conformal coverage
            lower_train = np.minimum(method_train_data, method_std_train)  # y_min_train, y_max_train
            upper_train = np.maximum(method_train_data, method_std_train)
            lower_calib = np.minimum(method_calib_data, method_std_calib)  # y_min_calib, y_max_calib
            upper_calib = np.maximum(method_calib_data, method_std_calib)

            in_interval_train = (y_train >= lower_train) & (y_train <= upper_train)
            in_interval_calib = (y_calib >= lower_calib) & (y_calib <= upper_calib)
            method_coverage_train_calib = 100 * np.mean(np.concatenate([in_interval_train, in_interval_calib]))

            # Coverage on all test points
            lower_test = np.minimum(method_y_min, method_y_max)
            upper_test = np.maximum(method_y_min, method_y_max)
            method_coverage_all = 100 * np.mean((y_test >= lower_test) & (y_test <= upper_test))

            # Mean width on test data for display
            method_mean_width = np.mean(upper_test - lower_test)
        elif method_train_data is not None:
            # Other methods: use ±std
            in_interval_train = np.abs(y_train - method_train_data) <= method_std_train
            in_interval_calib = np.abs(y_calib - method_calib_data) <= method_std_calib
            method_coverage_train_calib = 100 * np.mean(np.concatenate([in_interval_train, in_interval_calib]))

            # Coverage on all test points
            method_coverage_all = 100 * np.mean(np.abs(y_test - method_y_pred) <= method_y_std)

            method_mean_width = np.mean(2 * method_y_std)
        else:
            # Fallback to test data if train/calib not available
            method_coverage_train_calib = 100.0 * np.mean(np.abs(y_test - method_y_pred) <= method_y_std)
            method_coverage_all = method_coverage_train_calib
            method_mean_width = np.mean(2 * method_y_std)

        # MSE (accuracy)
        method_mse = np.mean((y_test - method_y_pred) ** 2)

        # Log likelihood and CRPS (probabilistic metrics)
        if method_label == 'Neural network':
            # NN ensemble: no parametric log likelihood, use empirical CRPS
            method_log_lik = None
            if 'ensemble_preds' in extra_data:
                method_crps = crps_ensemble(y_test, extra_data['ensemble_preds'])
            else:
                method_crps = crps_gaussian(y_test, method_y_pred, method_y_std)
        elif method_label == 'Quantile regression':
            # Quantile regression: no parametric log likelihood, use quantile-based CRPS
            method_log_lik = None
            if 'y_lower' in extra_data:
                method_crps = crps_quantile_approx(
                    y_test,
                    extra_data['y_lower'],
                    extra_data['y_median'],
                    extra_data['y_upper'],
                    extra_data['alpha']
                )
            else:
                method_crps = crps_gaussian(y_test, method_y_pred, method_y_std)
        elif method_label == 'Conformal prediction':
            # The conformal band is the 1 - zeta interval, q-hat times the posterior sd; its Gaussian reading has
            # sd = half-width / z_{1 - zeta/2} (scoring the half-width itself as the sd would overstate the spread)
            from scipy.stats import norm as _norm
            _sd = method_y_std / _norm.ppf(1 - get_zeta() / 2)
            method_log_lik = gaussian_log_likelihood_per_point(y_test, method_y_pred, _sd)
            method_crps = crps_gaussian(y_test, method_y_pred, _sd)
        else:
            # Gaussian methods (Bayesian, GP, POPS, CR3): use Gaussian formulas
            method_log_lik = gaussian_log_likelihood_per_point(y_test, method_y_pred, method_y_std)
            method_crps = crps_gaussian(y_test, method_y_pred, method_y_std)

        # Extract LOO log likelihood from extra_data (if available)
        method_loo_log_lik = extra_data.get('loo_log_lik', None)

        # Coverage of the +-1 sd Gaussian interval on the test curve, the same nominal level (68 %) for every method
        if method_label == 'Conformal prediction':
            from scipy.stats import norm as _norm
            _sd1 = method_y_std / _norm.ppf(1 - get_zeta() / 2)
        else:
            _sd1 = method_y_std
        method_cov_1sd = 100 * np.mean(np.abs(y_test - method_y_pred) <= _sd1)

        metrics_dict[method_label] = {
            'cov_1sd': method_cov_1sd,
            'coverage_train_calib': method_coverage_train_calib,
            'coverage_all': method_coverage_all,
            'mean_width': method_mean_width,
            'mse': method_mse,
            'fit_time': method_fit_time,
            'log_likelihood': method_log_lik,
            'crps': method_crps,
            'loo_log_lik': method_loo_log_lik
        }

    mo.Html(f'''
    <div class="app-plot">
        {mo.center(fig)}
    </div>
    ''')
    return bayes_log_ml, cr3_scale, gp_log_ml, gp_sparsity, metrics_dict, n, qhat


@app.cell
def _(
    bayesian,
    conformal,
    cr3,
    get_N_samples,
    scene,
    scene_bar,
    get_basis_lengthscale,
    get_basis_ls_enabled,
    get_basis_type,
    get_calib_frac,
    get_cr3_calibrate,
    get_cr3_cluster_size,
    get_custom_basis_code,
    get_custom_function_code,
    get_custom_kernel_code,
    get_filter_invert,
    get_filter_max,
    get_filter_min,
    get_function_type,
    get_gp_joint_inference,
    get_gp_kernel_type,
    get_gp_mean_regularization,
    get_leverage_percentile,
    get_n_posterior_samples,
    get_nn_activation,
    get_nn_ensemble_method,
    get_nn_ensemble_size,
    get_nn_hidden_units,
    get_nn_num_layers,
    get_nn_regularization,
    get_percentile_clipping,
    get_pops_posterior,
    get_quantile_confidence,
    get_quantile_regularization,
    get_seed,
    get_show_samples,
    get_sigma,
    get_zeta,
    gp_optimize_button,
    gp_regression,
    gp_use_basis_mean,
    mo,
    neural_network,
    np,
    pops,
    quantile,
    set_N_samples,
    set_P,
    set_basis_lengthscale,
    set_basis_ls_enabled,
    set_basis_type,
    set_calib_frac,
    set_cr3_calibrate,
    set_cr3_cluster_size,
    set_custom_basis_code,
    set_custom_function_code,
    set_custom_kernel_code,
    set_filter_invert,
    set_filter_max,
    set_filter_min,
    set_function_type,
    set_gp_joint_inference,
    set_gp_kernel_type,
    set_gp_lengthscale,
    set_gp_mean_regularization,
    set_gp_signal_variance,
    set_gp_support_radius,
    set_leverage_percentile,
    set_n_posterior_samples,
    set_nn_activation,
    set_nn_ensemble_method,
    set_nn_ensemble_size,
    set_nn_hidden_units,
    set_nn_num_layers,
    set_nn_regularization,
    set_percentile_clipping,
    set_pops_posterior,
    set_quantile_confidence,
    set_quantile_regularization,
    set_seed,
    set_sigma,
    set_zeta,
    show_samples_checkbox,
):
    # NOTE: This cell does NOT depend on get_P, get_gp_lengthscale, or get_gp_support_radius
    # to avoid circular dependency with optimization cells that modify those values
    aleatoric = mo.ui.checkbox(False, label="Include aleatoric uncertainty")

    data_label = mo.md("**Dataset parameters**")

    # Valid function types
    function_dropdown = mo.ui.dropdown(
        options=[
            'sin',
            'witch',
            'sinc',
            'step',
            'runge',
            'lj',
            'custom',
        ],
        label='Ground truth function',
        value=get_function_type(),
        on_change=set_function_type,
    )
    N_samples = mo.ui.slider(50, 1000, 50, get_N_samples(), label='Samples $N$', on_change=set_N_samples)
    sigma = mo.ui.slider(0.001, 0.3, 0.005, get_sigma(), label=r'$\sigma$ noise', on_change=set_sigma)
    seed = mo.ui.slider(0, 10, 1, get_seed(), label="Seed", on_change=set_seed)
    filter_range = mo.ui.range_slider(
        start=-10, stop=10, step=0.5,
        value=[get_filter_min(), get_filter_max()],
        label='Filter range',
        on_change=lambda v: (set_filter_min(v[0]), set_filter_max(v[1]))
    )
    filter_invert = mo.ui.checkbox(get_filter_invert(), label="Keep inside (invert)", on_change=set_filter_invert)

    # Regression parameters with conditional styling
    reg_enabled = bayesian.value or conformal.value or pops.value or quantile.value or cr3.value

    if reg_enabled:
        # Use fixed default value (not state) to avoid circular dependency
        # Manual slider changes still update state via on_change, but slider doesn't react to state changes
        P_slider = mo.ui.slider(1, 15, 1, scene.get("P", 10), label="Parameters $P$", on_change=set_P)
        P_elem = P_slider  # For display
        def on_basis_type_change(v):
            set_basis_type(v)
            set_basis_ls_enabled(v in ['rbf', 'fourier', 'lj'])
        basis_type_dropdown = mo.ui.dropdown(
            options=['polynomial', 'rbf', 'fourier', 'lj', 'custom'],
            value=get_basis_type(),
            on_change=on_basis_type_change
        )
        basis_dropdown = mo.hstack([
            mo.md("Basis type"),
            basis_type_dropdown
        ], justify="space-between", align="center")
        # Lengthscale slider - enabled for RBF/Fourier, disabled for polynomial
        if get_basis_ls_enabled():
            basis_lengthscale_slider = mo.ui.slider(0.5, 10.0, 0.5, get_basis_lengthscale(), label='Basis lengthscale', on_change=set_basis_lengthscale)
        else:
            basis_lengthscale_slider = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0.5, 10.0, 0.5, get_basis_lengthscale(), label='Basis lengthscale', disabled=True)}</div>")
        aleatoric = mo.ui.checkbox(False, label="Include aleatoric uncertainty")
        show_samples_linear = mo.ui.checkbox(get_show_samples_linear(), label="Show posterior samples", on_change=set_show_samples_linear)
        n_samples_linear_slider = mo.ui.slider(5, 20, 1, get_n_samples_linear(), label="Samples to draw", on_change=set_n_samples_linear)
        reg_separator = mo.Html("<hr style='margin: 5px 0; border: 0; border-top: 1px solid #ddd;'>")
    else:
        P_slider = None  # No slider when disabled
        P_elem = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(5, 15, 1, 10, label='Parameters $P$', disabled=True, on_change=set_P)}</div>")
        basis_type_dropdown = None  # Disabled
        basis_dropdown = mo.Html(f"<div style='opacity: 0.4; display: flex; justify-content: space-between; align-items: center;'><span>Basis type</span><select disabled style='padding: 4px;'><option>{get_basis_type()}</option></select></div>")
        basis_lengthscale_slider = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0.5, 10.0, 0.5, 2.0, label='Basis lengthscale', disabled=True)}</div>")
        aleatoric = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.checkbox(False, label='Include aleatoric uncertainty', disabled=True)}</div>")
        show_samples_linear = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.checkbox(False, label='Show posterior samples', disabled=True)}</div>")
        n_samples_linear_slider = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(5, 20, 1, 10, label='Samples to draw', disabled=True)}</div>")
        reg_separator = mo.Html("<hr style='margin: 5px 0; border: 0; border-top: 1px solid #ddd; opacity: 0.4;'>")

    # Conformal prediction section with conditional styling
    if conformal.value:
        calib_frac = mo.ui.slider(0.05, 0.5, 0.05, get_calib_frac(), label="Calibration fraction", on_change=set_calib_frac)
        zeta = mo.ui.slider(0.05, 0.3, 0.05, get_zeta(), label=r"Coverage $\zeta$", on_change=set_zeta)
        cp_separator = mo.Html("<hr style='margin: 5px 0; border: 0; border-top: 1px solid #ddd;'>")
    else:
        calib_frac = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0.05, 0.5, 0.05, get_calib_frac(), label='Calibration fraction', disabled=True, on_change=set_calib_frac)}</div>")
        zeta = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0.05, 0.3, 0.05, get_zeta(), label=r'Coverage $\zeta$', disabled=True, on_change=set_zeta)}</div>")
        cp_separator = mo.Html("<hr style='margin: 5px 0; border: 0; border-top: 1px solid #ddd; opacity: 0.4;'>")

    # POPS regression section with conditional styling
    if pops.value:
        pops_posterior_radio = mo.ui.radio(
            options=['hypercube', 'ensemble'],
            value=get_pops_posterior(),
            label='Posterior',
            on_change=set_pops_posterior
        )
        leverage_percentile_slider = mo.ui.slider(0.0, 50.0, 1.0, get_leverage_percentile(), label="Leverage percentile", on_change=set_leverage_percentile)
        percentile_clipping = mo.ui.slider(0, 10, 1, get_percentile_clipping(), label="Percentile clipping", on_change=set_percentile_clipping)
    else:
        pops_posterior_radio = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.radio(options=['hypercube', 'ensemble'], value=get_pops_posterior(), label='Posterior', disabled=True)}</div>")
        leverage_percentile_slider = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0.0, 50.0, 1.0, get_leverage_percentile(), label='Leverage percentile', disabled=True)}</div>")
        percentile_clipping = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0, 10, 1, get_percentile_clipping(), label='Percentile clipping', disabled=True, on_change=set_percentile_clipping)}</div>")

    # CR3 jackknife section with conditional styling
    if cr3.value:
        cr3_cluster_slider = mo.ui.slider(1, 20, 1, get_cr3_cluster_size(), label="Cluster size", on_change=set_cr3_cluster_size)
        cr3_calibrate_box = mo.ui.checkbox(get_cr3_calibrate(), label="Scale λ from calibration set", on_change=set_cr3_calibrate)
    else:
        cr3_cluster_slider = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(1, 20, 1, get_cr3_cluster_size(), label='Cluster size', disabled=True)}</div>")
        cr3_calibrate_box = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.checkbox(get_cr3_calibrate(), label='Scale λ from calibration set', disabled=True)}</div>")

    # Quantile regression section with conditional styling
    if quantile.value:
        quantile_confidence = mo.ui.slider(0.80, 0.95, 0.05, get_quantile_confidence(), label="Confidence level", on_change=set_quantile_confidence)
        quantile_regularization = mo.ui.slider(0.0, 0.1, 0.001, get_quantile_regularization(), label="Regularization", on_change=set_quantile_regularization)
    else:
        quantile_confidence = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0.80, 0.95, 0.05, get_quantile_confidence(), label='Confidence level', disabled=True, on_change=set_quantile_confidence)}</div>")
        quantile_regularization = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0.0, 0.1, 0.001, get_quantile_regularization(), label='Regularization', disabled=True, on_change=set_quantile_regularization)}</div>")

    # GP regression section with conditional styling
    if gp_regression.value:
        aleatoric_gp = mo.ui.checkbox(False, label="Include aleatoric uncertainty")
        show_samples_gp = mo.ui.checkbox(get_show_samples_gp(), label="Show posterior samples", on_change=set_show_samples_gp)
        n_samples_gp_slider = mo.ui.slider(5, 20, 1, get_n_samples_gp(), label="Samples to draw", on_change=set_n_samples_gp)
        gp_kernel_dropdown = mo.ui.dropdown(
            options=['rbf', 'matern', 'bump', 'custom'],
            value=get_gp_kernel_type(),
            label='Kernel type',
            on_change=set_gp_kernel_type
        )
        # Use fixed default values (not state) to avoid circular dependency
        # Manual slider changes still update state via on_change, but sliders don't react to state changes
        gp_lengthscale_slider = mo.ui.slider(0.1, 5.0, 0.1, 0.5, label='Lengthscale', on_change=set_gp_lengthscale)
        gp_signal_variance_slider = mo.ui.slider(0.1, 5.0, 0.1, 1.0, label='Signal variance', on_change=set_gp_signal_variance)
        gp_support_radius_slider = mo.ui.slider(0.5, 5.0, 0.1, 1.5, label='Support radius (bump only)', on_change=set_gp_support_radius)

        # gp_use_basis_mean checkbox is now created in a separate cell and passed in as dependency
        # We can check its .value here to conditionally show controls
        if gp_use_basis_mean.value:
            gp_joint_inference = mo.ui.checkbox(get_gp_joint_inference(), label="Joint Bayesian inference", on_change=set_gp_joint_inference)
            # Log-scale slider for mean regularization (better range control)
            # Maps -6 to 1 in log10 space → 10^-6 to 10^1 = 0.000001 to 10
            _log_reg = np.log10(get_gp_mean_regularization())
            gp_mean_regularization_log = mo.ui.slider(-6, 1, 0.1, _log_reg, label='Mean regularization (log₁₀)', on_change=lambda v: set_gp_mean_regularization(10**v))
            gp_mean_regularization = gp_mean_regularization_log  # For layout
        else:
            gp_joint_inference = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.checkbox(get_gp_joint_inference(), label='Joint Bayesian inference', disabled=True)}</div>")
            _log_reg = np.log10(get_gp_mean_regularization())
            gp_mean_regularization = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(-6, 1, 0.1, _log_reg, label='Mean regularization (log₁₀)', disabled=True)}</div>")

        gp_lengthscale = gp_lengthscale_slider  # For display
        gp_signal_variance = gp_signal_variance_slider  # For display
        gp_support_radius = gp_support_radius_slider  # For display
        gp_opt_button_elem = gp_optimize_button
        # Add horizontal separator between kernel hyperparameters and mean function controls
        gp_separator = mo.Html("<hr style='margin: 5px 0; border: 0; border-top: 1px solid #ddd;'>")
        # Show gp_use_basis_mean checkbox normally when GP regression is enabled
        gp_use_basis_mean_elem = gp_use_basis_mean
    else:
        aleatoric_gp = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.checkbox(False, label='Include aleatoric uncertainty', disabled=True)}</div>")
        show_samples_gp = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.checkbox(False, label='Show posterior samples', disabled=True)}</div>")
        n_samples_gp_slider = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(5, 20, 1, 10, label='Samples to draw', disabled=True)}</div>")
        # Dropdown doesn't have disabled attribute, just show greyed out
        gp_kernel_dropdown = mo.Html(f"<div style='opacity: 0.4; pointer-events: none;'>{mo.ui.dropdown(['rbf', 'matern', 'bump', 'custom'], value='rbf', label='Kernel type')}</div>")
        gp_lengthscale_slider = None  # No slider when disabled
        gp_signal_variance_slider = None  # No slider when disabled
        gp_support_radius_slider = None  # No slider when disabled
        gp_lengthscale = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0.1, 5.0, 0.1, 0.5, label='Lengthscale', disabled=True)}</div>")
        gp_signal_variance = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0.1, 5.0, 0.1, 1.0, label='Signal variance', disabled=True)}</div>")
        gp_support_radius = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0.5, 5.0, 0.1, 1.5, label='Support radius (bump only)', disabled=True)}</div>")
        gp_separator = mo.Html("<hr style='margin: 5px 0; border: 0; border-top: 1px solid #ddd; opacity: 0.4;'>")
        # Wrap gp_use_basis_mean checkbox with disabled styling when GP regression is off
        gp_use_basis_mean_elem = mo.Html(f"<div style='opacity: 0.4; pointer-events: none;'>{gp_use_basis_mean}</div>")
        gp_joint_inference = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.checkbox(False, label='Joint Bayesian inference', disabled=True)}</div>")
        gp_mean_regularization = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(-6, 1, 0.1, -1, label='Mean regularization (log₁₀)', disabled=True)}</div>")
        gp_opt_button_elem = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.button(label='Optimize hyperparameters', disabled=True)}</div>")

    # Neural network section with conditional styling
    if neural_network.value:
        nn_activation = mo.ui.dropdown(
            options=['tanh', 'relu'],
            value=get_nn_activation(),
            label='Activation function',
            on_change=set_nn_activation
        )
        nn_ensemble_method = mo.ui.dropdown(
            options=['seed', 'bootstrap'],
            value=get_nn_ensemble_method(),
            label='Ensemble method',
            on_change=set_nn_ensemble_method
        )
        nn_hidden_units = mo.ui.slider(5, 50, 5, get_nn_hidden_units(), label='Hidden units', on_change=set_nn_hidden_units)
        nn_num_layers = mo.ui.slider(1, 3, 1, get_nn_num_layers(), label='Hidden layers', on_change=set_nn_num_layers)
        _log_reg_nn = get_nn_regularization()
        nn_regularization = mo.ui.slider(-6, 0, 0.5, _log_reg_nn, label='Regularization (log₁₀)', on_change=set_nn_regularization)
        nn_ensemble_size = mo.ui.slider(3, 10, 1, get_nn_ensemble_size(), label='Ensemble size', on_change=set_nn_ensemble_size)
        show_samples_nn = mo.ui.checkbox(get_show_samples_nn(), label="Show ensemble samples", on_change=set_show_samples_nn)
        n_samples_nn_slider = mo.ui.slider(3, 10, 1, get_n_samples_nn(), label="Samples to draw", on_change=set_n_samples_nn)
    else:
        nn_activation = mo.Html(f"<div style='opacity: 0.4; pointer-events: none;'>{mo.ui.dropdown(['tanh'], value='tanh', label='Activation function')}</div>")
        nn_ensemble_method = mo.Html(f"<div style='opacity: 0.4; pointer-events: none;'>{mo.ui.dropdown(['seed', 'bootstrap'], value='seed', label='Ensemble method')}</div>")
        nn_hidden_units = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(5, 50, 5, get_nn_hidden_units(), label='Hidden units', disabled=True)}</div>")
        nn_num_layers = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(1, 3, 1, get_nn_num_layers(), label='Hidden layers', disabled=True)}</div>")
        nn_regularization = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(-6, 0, 0.5, -3, label='Regularization (log₁₀)', disabled=True)}</div>")
        nn_ensemble_size = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(3, 10, 1, 5, label='Ensemble size', disabled=True)}</div>")
        show_samples_nn = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.checkbox(False, label='Show ensemble samples', disabled=True)}</div>")
        n_samples_nn_slider = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(3, 10, 1, 5, label='Samples to draw', disabled=True)}</div>")

    # Linear Methods tab: Two-column layout
    # First column: Shared settings, Bayesian fit, and Quantile regression
    linear_col1 = mo.vstack([
        P_elem, basis_dropdown, basis_lengthscale_slider, aleatoric,
        show_samples_linear, n_samples_linear_slider,
        mo.Html("<hr style='margin: 10px 0; border: 0; border-top: 1px solid #ddd;'>"),
        mo.left(bayesian),
        mo.Html("<hr style='margin: 5px 0; border: 0; border-top: 1px solid #ddd;'>"),
        mo.left(quantile),
        quantile_confidence,
        quantile_regularization
    ])

    # Second column: Conformal Prediction, POPS regression and CR3 jackknife
    linear_col2 = mo.vstack([
        mo.left(conformal),
        calib_frac, zeta, cp_separator,
        mo.left(pops),
        pops_posterior_radio,
        leverage_percentile_slider,
        percentile_clipping,
        mo.Html("<hr style='margin: 5px 0; border: 0; border-top: 1px solid #ddd;'>"),
        mo.left(cr3),
        cr3_cluster_slider,
        cr3_calibrate_box
    ])

    linear_methods_content = mo.Html(f'''
    <div style="display: flex; gap: 20px;">
        <div style="width: 50%; min-width: 200px;">
            {linear_col1}
        </div>
        <div style="width: 50%; min-width: 200px;">
            {linear_col2}
        </div>
    </div>
    ''')

    # Kernel Methods tab: Two-column layout
    kernel_col1 = mo.vstack([
        mo.left(gp_regression),
        aleatoric_gp,
        show_samples_gp, n_samples_gp_slider,
        gp_kernel_dropdown, gp_lengthscale, gp_signal_variance, gp_support_radius,
        gp_opt_button_elem
    ])

    kernel_col2 = mo.vstack([
        gp_use_basis_mean_elem,
        gp_joint_inference, gp_mean_regularization
    ])

    kernel_methods_content = mo.Html(f'''
    <div style="display: flex; gap: 20px;">
        <div style="width: 50%; min-width: 200px;">
            {kernel_col1}
        </div>
        <div style="width: 50%; min-width: 200px;">
            {kernel_col2}
        </div>
    </div>
    ''')

    # Non-linear Methods tab: Single column with checkbox at top
    nonlinear_methods_tab = mo.vstack([
        mo.left(neural_network),
        show_samples_nn, n_samples_nn_slider,
        nn_activation,
        nn_ensemble_method,
        nn_hidden_units, nn_num_layers,
        nn_regularization, nn_ensemble_size
    ])

    # Custom Function code editor (in accordion, collapsed by default)
    custom_code_editor = mo.ui.code_editor(
        value=get_custom_function_code(),
        language="python",
        min_height=200,
        on_change=set_custom_function_code,
    )
    custom_function_accordion = mo.accordion({
        "Custom Function Code": mo.vstack([
            mo.md("Define `y = f(X)`. Available: `np`, `scipy`, `math`, `X`. Avoid infinite loops."),
            custom_code_editor,
        ])
    }, lazy=True)

    # Custom Basis code editor (in accordion, collapsed by default)
    custom_basis_editor = mo.ui.code_editor(
        value=get_custom_basis_code(),
        language="python",
        min_height=200,
        on_change=set_custom_basis_code,
    )
    custom_basis_accordion = mo.accordion({
        "Custom Basis Code": mo.vstack([
            mo.md("Define `features` (n×P matrix). Available: `np`, `scipy`, `math`, `X`, `P`. Avoid infinite loops."),
            custom_basis_editor,
        ])
    }, lazy=True)

    # Custom Kernel code editor (in accordion, collapsed by default)
    custom_kernel_editor = mo.ui.code_editor(
        value=get_custom_kernel_code(),
        language="python",
        min_height=200,
        on_change=set_custom_kernel_code,
    )
    custom_kernel_accordion = mo.accordion({
        "Custom Kernel Code": mo.vstack([
            mo.md("Define `K` matrix (n1×n2). Available: `np`, `scipy`, `math`, `X1`, `X2`, `lengthscale`, `signal_variance`. Avoid infinite loops."),
            custom_kernel_editor,
        ])
    }, lazy=True)

    # Data tab: Dataset parameters + custom function accordion
    data_tab = mo.vstack([
        scene_bar,
        mo.hstack([function_dropdown, N_samples, sigma], justify="start"),
        mo.hstack([filter_range, filter_invert, seed], justify="start"),
        custom_function_accordion,
    ])

    # Assemble final tabs with accordions
    linear_methods_tab = mo.vstack([linear_methods_content, custom_basis_accordion])
    kernel_methods_tab = mo.vstack([kernel_methods_content, custom_kernel_accordion])

    # Create tabs for method-specific parameters (Data first)
    method_params_tabs = mo.ui.tabs({
        "Data": data_tab,
        "Linear Methods": linear_methods_tab,
        "Kernel Methods": kernel_methods_tab,
        "Non-linear Methods": nonlinear_methods_tab,
    })

    controls = method_params_tabs

    mo.Html(f'''
    <div class="app-dashboard">
        {controls}
    </div>
    ''')
    return (
        N_samples,
        aleatoric,
        aleatoric_gp,
        basis_type_dropdown,
        function_dropdown,
        gp_lengthscale_slider,
        gp_signal_variance_slider,
        gp_support_radius_slider,
        seed,
        sigma,
    )


@app.cell
def _(metrics_dict, mo):
    # Generate and display metrics table in an accordion
    if not metrics_dict:
        metrics_accordion = mo.accordion({"Performance Metrics": mo.md("*No metrics available. Enable a regression method.*")})
    else:
        # Compute best/worst for each metric (for coloring)
        # Higher is better: coverage, log_likelihood, loo_log_lik
        # Lower is better: mse, crps, fit_time
        all_cov_1sd = [abs(m['cov_1sd'] - 68.3) for m in metrics_dict.values()]   # closest to nominal is best
        all_mse = [m['mse'] for m in metrics_dict.values()]
        all_log_lik = [m['log_likelihood'] for m in metrics_dict.values() if m['log_likelihood'] is not None]
        all_loo = [m['loo_log_lik'] for m in metrics_dict.values() if m['loo_log_lik'] is not None]
        all_crps = [m['crps'] for m in metrics_dict.values()]
        all_time = [m['fit_time'] for m in metrics_dict.values()]

        # Best/worst values (handle empty lists for optional metrics)
        best_cov_1sd, worst_cov_1sd = min(all_cov_1sd), max(all_cov_1sd)
        best_mse, worst_mse = min(all_mse), max(all_mse)
        best_log_lik = max(all_log_lik) if all_log_lik else None
        worst_log_lik = min(all_log_lik) if all_log_lik else None
        best_loo = max(all_loo) if all_loo else None
        worst_loo = min(all_loo) if all_loo else None
        best_crps, worst_crps = min(all_crps), max(all_crps)
        best_time, worst_time = min(all_time), max(all_time)

        def get_color(val, best, worst):
            """Return green for best, red for worst, black otherwise."""
            if val == best and val != worst:  # Only color if there's a difference
                return "#28a745"  # green
            elif val == worst and val != best:
                return "#dc3545"  # red
            return "inherit"

        # Create table rows
        rows = []
        for method_name, method_metrics in metrics_dict.items():
            cov_tc = method_metrics['coverage_train_calib']
            cov_all = method_metrics['coverage_all']
            cov_1sd = method_metrics['cov_1sd']
            mse = method_metrics['mse']
            log_lik = method_metrics['log_likelihood']
            loo = method_metrics['loo_log_lik']
            crps = method_metrics['crps']
            m_fit_time = method_metrics['fit_time']

            # Get colors for each metric
            # Cov (t+c) and Cov (all) use each method's own band (+-1 sd, the conformal 1 - zeta interval, POPS min-max
            # bounds), so their nominal levels differ and "highest" is not "best": left uncoloured.  Cov +-1 sd puts every
            # method at 68.3 % and is coloured by closeness to it.
            cov_tc_color = cov_all_color = "inherit"
            cov_1sd_color = get_color(abs(cov_1sd - 68.3), best_cov_1sd, worst_cov_1sd)
            mse_color = get_color(mse, best_mse, worst_mse)
            log_lik_color = get_color(log_lik, best_log_lik, worst_log_lik) if log_lik is not None else "inherit"
            loo_color = get_color(loo, best_loo, worst_loo) if loo is not None else "inherit"
            crps_color = get_color(crps, best_crps, worst_crps)
            time_color = get_color(m_fit_time, best_time, worst_time)

            # Format values
            log_lik_str = f"{log_lik:.2f}" if log_lik is not None else "-"
            loo_str = f"{loo:.2f}" if loo is not None else "-"

            rows.append(f'''
            <tr>
                <td style="text-align: left; padding: 3px 6px;"><b>{method_name}</b></td>
                <td style="text-align: center; padding: 3px 6px; color: {cov_tc_color};">{cov_tc:.1f}%</td>
                <td style="text-align: center; padding: 3px 6px; color: {cov_all_color};">{cov_all:.1f}%</td>
                <td style="text-align: center; padding: 3px 6px; color: {cov_1sd_color};">{cov_1sd:.1f}%</td>
                <td style="text-align: center; padding: 3px 6px; color: {mse_color};">{mse:.4f}</td>
                <td style="text-align: center; padding: 3px 6px; color: {log_lik_color};">{log_lik_str}</td>
                <td style="text-align: center; padding: 3px 6px; color: {loo_color};">{loo_str}</td>
                <td style="text-align: center; padding: 3px 6px; color: {crps_color};">{crps:.4f}</td>
                <td style="text-align: center; padding: 3px 6px; color: {time_color};">{m_fit_time*1000:.1f}ms</td>
            </tr>
            ''')

        metrics_table = mo.Html(f'''
        <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
            <thead>
                <tr style="border-bottom: 2px solid #dee2e6;">
                    <th style="text-align: left; padding: 5px 10px;">Method</th>
                    <th style="text-align: center; padding: 5px 10px;" title="Coverage of each method's own band on train + calibration points; nominal levels differ between methods, so not coloured">Cov (t+c)</th>
                    <th style="text-align: center; padding: 5px 10px;" title="Coverage of each method's own band on the test curve; nominal levels differ between methods, so not coloured">Cov (all)</th>
                    <th style="text-align: center; padding: 5px 10px;" title="Coverage of the ±1σ Gaussian interval on the test curve; nominal 68.3 % for every method (conformal: σ = half-width / z)">Cov ±1σ</th>
                    <th style="text-align: center; padding: 5px 10px;">MSE</th>
                    <th style="text-align: center; padding: 5px 10px;">Log Lik</th>
                    <th style="text-align: center; padding: 5px 10px;">LOO</th>
                    <th style="text-align: center; padding: 5px 10px;">CRPS</th>
                    <th style="text-align: center; padding: 5px 10px;">Fit Time</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        ''')
        metrics_note = mo.md("<small>Cov (t+c) and Cov (all) use each method's own band: ±1σ, the conformal 1 − ζ interval, or the "
                             "POPS min–max bounds. Their nominal levels differ, so they are not coloured: higher is not better. "
                             "<b>Cov ±1σ</b> puts every method at the same nominal 68.3 % and is coloured by closeness to it. Log Lik and CRPS score the "
                             "Gaussian reading on the test curve (conformal: σ = half-width / z<sub>1−ζ/2</sub>).</small>")
        metrics_accordion = mo.accordion({"Performance Metrics": mo.vstack([metrics_table, metrics_note])})
    metrics_accordion
    return


@app.cell
def _(
    bayes_log_ml,
    bayesian,
    conformal,
    cr3,
    cr3_scale,
    gp_log_ml,
    gp_optimize_button,
    gp_regression,
    gp_sparsity,
    mo,
    n,
    opt_message,
    qhat,
):
    # Display computed outputs below the dashboard (only values not in sliders)
    output_items = []

    if bayesian.value and bayes_log_ml != 0.0:
        output_items.append(f"Bayesian log ML: {bayes_log_ml:.1f}")

    if conformal.value:
        output_items.append(f"Calibration size (n): {n}")
        output_items.append(f"Quantile (q̂): {qhat:.2f}")

    if cr3.value and cr3_scale is not None:
        output_items.append(f"CR3 scale (λ): {cr3_scale:.2f}")

    if gp_regression.value:
        # Add optimized hyperparameters if available
        if gp_optimize_button.value and opt_message is not None and isinstance(opt_message, dict):
            if 'error' in opt_message:
                output_items.append(f"Opt error: {opt_message['error']}")
            else:
                # Show optimized log ML if available, otherwise current log ML
                log_ml_value = opt_message.get('log_ml', gp_log_ml)
                output_items.append(f"GP log ML: {log_ml_value:.1f}")

                if opt_message.get('lengthscale') is not None:
                    output_items.append(f"Opt lengthscale: {opt_message['lengthscale']:.3f}")
                if opt_message.get('support_radius') is not None:
                    output_items.append(f"Opt support: {opt_message['support_radius']:.3f}")
                if opt_message.get('noise') is not None:
                    output_items.append(f"Opt noise: {opt_message['noise']:.4f}")
                if opt_message.get('mean_reg') is not None:
                    output_items.append(f"Opt mean reg: {opt_message['mean_reg']:.4f}")
        else:
            # Show current log ML when not optimized
            output_items.append(f"GP log ML: {gp_log_ml:.1f}")

        output_items.append(f"Sparsity: {gp_sparsity:.1f}%")

    # Display outputs
    if output_items:
        outputs_text = " | ".join(output_items)
        output_html = f'''
        <div style="background-color: #e8f4f8; padding: 8px 15px; border-radius: 5px; margin: 0 auto 10px auto; max-width: 90%; text-align: center; font-size: 14px;">
            <b>Outputs:</b> {outputs_text}
        </div>
        '''
        mo.Html(output_html)
    return


@app.cell(hide_code=True)
def _(mo, scene, set_last_enabled_method):
    _on = scene.get('methods', set())
    bayesian = mo.ui.checkbox('bayesian' in _on, label="<b>Bayesian fit</b>",
                               on_change=lambda v: set_last_enabled_method('linear') if v else None)
    conformal = mo.ui.checkbox('conformal' in _on, label="<b>Conformal prediction</b>",
                                on_change=lambda v: set_last_enabled_method('linear') if v else None)
    pops = mo.ui.checkbox('pops' in _on, label="<b>POPS regression</b>",
                          on_change=lambda v: set_last_enabled_method('linear') if v else None)
    quantile = mo.ui.checkbox(False, label="<b>Quantile regression</b>",
                              on_change=lambda v: set_last_enabled_method('linear') if v else None)
    cr3 = mo.ui.checkbox('cr3' in _on, label="<b>CR3 jackknife</b>",
                         on_change=lambda v: set_last_enabled_method('linear') if v else None)
    gp_regression = mo.ui.checkbox(False, label="<b>GP regression</b>",
                                    on_change=lambda v: set_last_enabled_method('gp') if v else None)
    neural_network = mo.ui.checkbox(False, label="<b>Neural network</b>",
                                     on_change=lambda v: set_last_enabled_method('nn') if v else None)
    gp_use_basis_mean = mo.ui.checkbox(False, label="Use linear basis mean function")
    return (
        bayesian,
        conformal,
        cr3,
        gp_regression,
        gp_use_basis_mean,
        neural_network,
        pops,
        quantile,
    )


@app.cell(hide_code=True)
def _(mo):
    # Button definitions - isolated cell with no state dependencies to prevent circular loops
    gp_optimize_button = mo.ui.run_button(label="Optimize hyperparameters")
    return (gp_optimize_button,)


@app.cell(hide_code=True)
def _(mo, scene):
    # Use marimo state to preserve all slider values
    get_N_samples, set_N_samples = mo.state(500)
    get_sigma, set_sigma = mo.state(scene.get('sigma', 0.1))
    get_seed, set_seed = mo.state(0)
    get_filter_min, set_filter_min = mo.state(scene.get('filter', (0.0, 5.0))[0])
    get_filter_max, set_filter_max = mo.state(scene.get('filter', (0.0, 5.0))[1])
    get_filter_invert, set_filter_invert = mo.state(scene.get('invert', False))
    get_function_type, set_function_type = mo.state(scene.get('function', 'sin'))
    get_custom_function_code, set_custom_function_code = mo.state(
        "# Define y as a function of X (numpy array)\n"
        "# Available: np, scipy, math, X\n"
        "y = np.sin(X) + 0.1 * X**2"
    )
    get_P, set_P = mo.state(scene.get('P', 10))
    get_basis_type, set_basis_type = mo.state('polynomial')
    get_basis_lengthscale, set_basis_lengthscale = mo.state(1.5)  # Default matches LJ ground truth offset
    get_basis_ls_enabled, set_basis_ls_enabled = mo.state(False)  # True for rbf/fourier
    get_custom_basis_code, set_custom_basis_code = mo.state(
        "# Define 'features' as 2D array (n_samples, n_features)\n"
        "# Available: np, scipy, math, X (shape n,1), P (num params)\n"
        "x = X[:, 0]\n"
        "features = np.column_stack([x**i for i in range(P)])"
    )
    get_calib_frac, set_calib_frac = mo.state(0.2)
    get_zeta, set_zeta = mo.state(0.05)
    get_percentile_clipping, set_percentile_clipping = mo.state(0)
    get_pops_posterior, set_pops_posterior = mo.state('ensemble')
    get_leverage_percentile, set_leverage_percentile = mo.state(0.0)
    get_cr3_cluster_size, set_cr3_cluster_size = mo.state(1)
    get_cr3_calibrate, set_cr3_calibrate = mo.state(True)

    # GP-specific state
    get_gp_kernel_type, set_gp_kernel_type = mo.state('rbf')
    get_gp_lengthscale, set_gp_lengthscale = mo.state(0.5)
    get_gp_signal_variance, set_gp_signal_variance = mo.state(1.0)
    get_gp_support_radius, set_gp_support_radius = mo.state(1.5)
    # gp_use_basis_mean is now a simple checkbox without state (created in analysis checkboxes cell)
    get_gp_joint_inference, set_gp_joint_inference = mo.state(False)
    get_gp_mean_regularization, set_gp_mean_regularization = mo.state(0.1)
    # Custom kernel code
    get_custom_kernel_code, set_custom_kernel_code = mo.state(
        "# Define K as a 2D kernel matrix (n1 x n2)\n"
        "# Available: np, scipy, math, X1, X2, lengthscale, signal_variance\n"
        "dists_sq = (X1[:, None] - X2[None, :]) ** 2\n"
        "K = signal_variance * np.exp(-0.5 * dists_sq / lengthscale**2)"
    )

    # Neural network-specific state
    get_nn_activation, set_nn_activation = mo.state('tanh')
    get_nn_ensemble_method, set_nn_ensemble_method = mo.state('seed')
    get_nn_hidden_units, set_nn_hidden_units = mo.state(20)
    get_nn_num_layers, set_nn_num_layers = mo.state(1)
    get_nn_regularization, set_nn_regularization = mo.state(-3)  # log10 scale
    get_nn_ensemble_size, set_nn_ensemble_size = mo.state(5)

    # Quantile regression-specific state
    get_quantile_confidence, set_quantile_confidence = mo.state(0.9)
    get_quantile_regularization, set_quantile_regularization = mo.state(0.01)

    # Button click count tracking to prevent infinite loops
    get_bayes_opt_count, set_bayes_opt_count = mo.state(0)
    get_gp_opt_count, set_gp_opt_count = mo.state(0)

    # Track last enabled method for inset display
    get_last_enabled_method, set_last_enabled_method = mo.state('linear')

    # Per-method posterior samples state
    get_show_samples_linear, set_show_samples_linear = mo.state(scene.get('samples', False))
    get_n_samples_linear, set_n_samples_linear = mo.state(10)

    get_show_samples_gp, set_show_samples_gp = mo.state(False)
    get_n_samples_gp, set_n_samples_gp = mo.state(10)

    get_show_samples_nn, set_show_samples_nn = mo.state(False)
    get_n_samples_nn, set_n_samples_nn = mo.state(10)
    return (
        get_N_samples,
        get_P,
        get_basis_lengthscale,
        get_basis_ls_enabled,
        get_basis_type,
        get_calib_frac,
        get_cr3_calibrate,
        get_cr3_cluster_size,
        get_custom_basis_code,
        get_custom_function_code,
        get_custom_kernel_code,
        get_filter_invert,
        get_filter_max,
        get_filter_min,
        get_function_type,
        get_gp_joint_inference,
        get_gp_kernel_type,
        get_gp_lengthscale,
        get_gp_mean_regularization,
        get_gp_signal_variance,
        get_gp_support_radius,
        get_last_enabled_method,
        get_leverage_percentile,
        get_n_samples_gp,
        get_n_samples_linear,
        get_n_samples_nn,
        get_nn_activation,
        get_nn_ensemble_method,
        get_nn_ensemble_size,
        get_nn_hidden_units,
        get_nn_num_layers,
        get_nn_regularization,
        get_percentile_clipping,
        get_pops_posterior,
        get_quantile_confidence,
        get_quantile_regularization,
        get_seed,
        get_show_samples_gp,
        get_show_samples_linear,
        get_show_samples_nn,
        get_sigma,
        get_zeta,
        set_N_samples,
        set_P,
        set_basis_lengthscale,
        set_basis_ls_enabled,
        set_basis_type,
        set_calib_frac,
        set_cr3_calibrate,
        set_cr3_cluster_size,
        set_custom_basis_code,
        set_custom_function_code,
        set_custom_kernel_code,
        set_filter_invert,
        set_filter_max,
        set_filter_min,
        set_function_type,
        set_gp_joint_inference,
        set_gp_kernel_type,
        set_gp_lengthscale,
        set_gp_mean_regularization,
        set_gp_signal_variance,
        set_gp_support_radius,
        set_last_enabled_method,
        set_leverage_percentile,
        set_n_samples_gp,
        set_n_samples_linear,
        set_n_samples_nn,
        set_nn_activation,
        set_nn_ensemble_method,
        set_nn_ensemble_size,
        set_nn_hidden_units,
        set_nn_num_layers,
        set_nn_regularization,
        set_percentile_clipping,
        set_pops_posterior,
        set_quantile_confidence,
        set_quantile_regularization,
        set_seed,
        set_show_samples_gp,
        set_show_samples_linear,
        set_show_samples_nn,
        set_sigma,
        set_zeta,
    )


@app.cell(hide_code=True)
def _(
    N_samples,
    PolynomialFeatures,
    g,
    get_P,
    get_basis_lengthscale,
    get_basis_type,
    get_custom_basis_code,
    get_custom_function_code,
    get_filter_max,
    get_filter_min,
    get_function_type,
    get_gp_kernel_type,
    gp_lengthscale_slider,
    gp_optimize_button,
    gp_support_radius_slider,
    gp_use_basis_mean,
    make_custom_features,
    make_fourier_features,
    make_lj_features,
    make_rbf_features,
    np,
    optimize_gp_hyperparameters,
    seed,
    set_gp_lengthscale,
    set_gp_mean_regularization,
    set_gp_support_radius,
    sigma,
):
    # Perform GP optimization when button is clicked
    # Only run if button clicked AND GP sliders are available (GP regression enabled)
    opt_message = None

    if gp_optimize_button.value and gp_lengthscale_slider is not None:
        # Regenerate training data (use _ prefix to avoid variable redefinition)
        np.random.seed(seed.value)
        _x_train = np.append(np.random.uniform(-10, 10, size=N_samples.value), np.linspace(-10, 10, 2))
        _x_train = _x_train[(_x_train < get_filter_min()) | (_x_train > get_filter_max())]
        _x_train = np.sort(_x_train)
        _func_type = get_function_type()
        _custom_code = get_custom_function_code() if _func_type == 'custom' else None
        _y_train = g(_x_train, noise_variance=sigma.value**2, function_type=_func_type, custom_code=_custom_code)

        # Set initial parameters - read from sliders, NOT state getters (avoids circular dependency)
        initial_params = {
            'lengthscale': gp_lengthscale_slider.value,
            'signal_variance': gp_signal_variance_slider.value,
            'support_radius': gp_support_radius_slider.value,
            'degree': get_P(),
            'sigma': 0.1,
            'noise': sigma.value**2
        }

        # Get basis mean settings
        _use_basis_mean = gp_use_basis_mean.value
        # Use default values to avoid circular dependency (cell modifies set_gp_mean_regularization)
        _joint_inference = False  # Default: False
        # Use default initial value for mean regularization (will be optimized)
        _mean_reg = 0.1

        # Create basis features if using basis mean (use shared make_*_features functions)
        _Phi_train = None
        if _use_basis_mean:
            _X_train = _x_train.reshape(-1, 1)  # Functions expect 2D array
            _x_min, _x_max = _x_train.min(), _x_train.max()
            _P = get_P()
            _basis_type = get_basis_type()

            if _basis_type == 'polynomial':
                _poly = PolynomialFeatures(degree=_P-1, include_bias=True)
                _Phi_train = _poly.fit_transform(_X_train)
            elif _basis_type == 'rbf':
                _Phi_train, _, _ = make_rbf_features(_X_train, _P, _x_min, _x_max, get_basis_lengthscale())
            elif _basis_type == 'fourier':
                _Phi_train, _ = make_fourier_features(_X_train, _P, _x_min, _x_max, get_basis_lengthscale())
            elif _basis_type == 'lj':
                _Phi_train, _ = make_lj_features(_X_train, _P, _x_min, _x_max)
            elif _basis_type == 'custom':
                _Phi_train, _ = make_custom_features(_X_train, _P, get_custom_basis_code())

        # Optimize
        opt_lengthscale = None  # Initialize variables used in output formatting
        opt_signal_variance = None
        opt_support = None
        opt_mean_reg = None
        opt_noise = None

        try:
            _kernel_type = get_gp_kernel_type()
            _custom_kernel_code = get_custom_kernel_code() if _kernel_type == 'custom' else None
            optimized_params, log_ml = optimize_gp_hyperparameters(
                _x_train, _y_train, _kernel_type, initial_params, max_iter=50,
                use_basis_mean=_use_basis_mean,
                Phi_train=_Phi_train,
                joint_inference=_joint_inference,
                mean_regularization_strength=_mean_reg,
                custom_code=_custom_kernel_code
            )

            # Update sliders with optimized values
            if 'lengthscale' in optimized_params:
                # Clamp to slider bounds [0.1, 5.0]
                opt_lengthscale = np.clip(optimized_params['lengthscale'], 0.1, 5.0)
                set_gp_lengthscale(float(opt_lengthscale))

            if 'signal_variance' in optimized_params:
                # Clamp to slider bounds [0.1, 5.0]
                opt_signal_variance = np.clip(optimized_params['signal_variance'], 0.1, 5.0)
                set_gp_signal_variance(float(opt_signal_variance))

            if 'support_radius' in optimized_params:
                # Clamp to slider bounds [0.5, 5.0]
                opt_support = np.clip(optimized_params['support_radius'], 0.5, 5.0)
                set_gp_support_radius(float(opt_support))

            if 'mean_regularization' in optimized_params:
                # Clamp to valid range [10^-6, 10^1]
                opt_mean_reg = np.clip(optimized_params['mean_regularization'], 1e-6, 10.0)
                set_gp_mean_regularization(float(opt_mean_reg))

            if 'noise' in optimized_params:
                opt_noise = optimized_params['noise']

            # Store optimized parameters for display in output cell
            opt_message = {
                'lengthscale': float(opt_lengthscale) if opt_lengthscale is not None else None,
                'signal_variance': float(opt_signal_variance) if opt_signal_variance is not None else None,
                'support_radius': float(opt_support) if opt_support is not None else None,
                'mean_reg': float(opt_mean_reg) if opt_mean_reg is not None else None,
                'noise': float(opt_noise) if opt_noise is not None else None,
                'log_ml': float(log_ml)
            }
        except Exception as e:
            opt_message = {'error': str(e)}
    return (opt_message,)


if __name__ == "__main__":
    app.run()
