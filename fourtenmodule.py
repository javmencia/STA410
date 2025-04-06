import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import FastICA
from scipy import stats
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold


# Bayesian Linear Regression Model
def bayesian_regression_mcmc(X, y, true_beta  = None):
    if true_beta is None:
        true_beta = np.zeros(X.shape[1])
    with pm.Model() as model:
        # Priors
        beta = pm.MvNormal("beta", mu=true_beta, chol=np.eye(X.shape[1]), shape=X.shape[1])
        sigma = pm.HalfCauchy("sigma", beta=2)  # Prior on error term

        # Likelihood
        mu = pm.math.dot(X, beta)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        # Sample from the posterior using MCMC
        trace = pm.sample(4000, return_inferencedata=True,
                          idata_kwargs={"log_likelihood": True})  # Ensure log likelihood is stored

    return model, trace

# Bayesian Ridge Regression Model
def bayesian_ridge_regression(X, y, true_beta  = None):
    if true_beta is None:
        true_beta = np.zeros(X.shape[1])
    with pm.Model() as model:
        # Prior on precision (1/tau^2) for ridge penalty
        tau = pm.HalfCauchy("tau", beta=1)  # Shrinkage parameter (larger beta = weaker prior)

        # Ridge prior on beta (similar to L2 regularization)
        beta = pm.MvNormal("beta", mu=true_beta, cov=tau**2 * np.eye(X.shape[1]), shape=X.shape[1])

        # Prior on the noise term
        sigma = pm.HalfCauchy("sigma", beta=2)

        # Likelihood
        mu = pm.math.dot(X, beta)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        # Sample from the posterior using MCMC
        trace = pm.sample(4000, return_inferencedata=True,
                          idata_kwargs={"log_likelihood": True})
    return model, trace

def bayesian_lasso(X, y, true_beta=None, n_folds=5):
    """Bayesian Lasso regression with cross-validated lambda selection"""
    if true_beta is None:
        true_beta = np.zeros(X.shape[1])
    
    # First perform cross-validation to find optimal alpha (lambda)
    lasso_cv = LassoCV(cv=KFold(n_folds), random_state=42)
    lasso_cv.fit(X, y)
    optimal_alpha = lasso_cv.alpha_
    
    with pm.Model() as model:
        # Lasso (Laplace) prior - equivalent to L1 regularization
        # Scale parameter is 1/optimal_alpha from CV
        beta = pm.Laplace("beta", mu=true_beta, b=1/optimal_alpha, shape=X.shape[1])
        sigma = pm.HalfCauchy("sigma", beta=2)

        # Likelihood
        mu = pm.math.dot(X, beta)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        # Sample from posterior
        trace = pm.sample(4000, return_inferencedata=True,
                         idata_kwargs={"log_likelihood": True})
    
    return model, trace


# Bayesian Robust Regression Model
def bayesian_robust_regression(X, y, true_beta  = None):
    if true_beta is None:
        true_beta = np.zeros(X.shape[1])
    with pm.Model() as model:
        beta = pm.MvNormal("beta", mu=true_beta, chol=np.eye(X.shape[1]), shape=X.shape[1])
        sigma = pm.HalfCauchy("sigma", beta=2)
        nu = pm.Exponential("nu", 1/30)  # Degrees of freedom for the Student-T

        mu = pm.math.dot(X, beta)
        y_obs = pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(4000, return_inferencedata=True,
                          idata_kwargs={"log_likelihood": True})

    return model, trace

def bayesian_regression_vi(X, y, true_beta = None):
    if true_beta is None:
        true_beta = np.zeros(X.shape[1])
    with pm.Model() as model:
        beta = pm.MvNormal("beta", mu=true_beta, chol=np.eye(X.shape[1]), shape=X.shape[1])
        sigma = pm.HalfCauchy("sigma", beta=2)
        mu = pm.math.dot(X, beta)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        approx = pm.fit(n=10000, method="advi")  # Automatic Differentiation Variational Inference (ADVI)
    return model, approx.sample(1000)

def bayesian_pcr(X, y, true_beta=None, n_components=None):
    if true_beta is None:
        true_beta = np.zeros(X.shape[1])
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)  # Shape (n, n_components)

    with pm.Model() as model:
        # Priors on regression coefficients in PCA space
        beta = pm.MvNormal("beta", mu=np.zeros(X_pca.shape[1]), chol=np.eye(X_pca.shape[1]), shape=X_pca.shape[1])
        sigma = pm.HalfCauchy("sigma", beta=2)

        # Likelihood
        mu = pm.math.dot(X_pca, beta)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        # Sample from the posterior
        trace = pm.sample(4000, return_inferencedata=True, idata_kwargs={"log_likelihood": True})

    return model, trace, pca  # Return PCA object for inverse transformation


def bayesian_icr(X, y, true_beta=None, n_components=None):
    if true_beta is None:
        true_beta = np.zeros(X.shape[1])
    
    # Apply ICA
    ica = FastICA(n_components=n_components, random_state=42)
    X_ica = ica.fit_transform(X)

    with pm.Model() as model:
        # Priors on regression coefficients in ICA space
        beta = pm.MvNormal("beta", mu=np.zeros(X_ica.shape[1]), 
                              chol=np.eye(X_ica.shape[1]), 
                              shape=X_ica.shape[1])
        sigma = pm.HalfCauchy("sigma", beta=2)

        # Likelihood
        mu = pm.math.dot(X_ica, beta)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        # Sample from the posterior
        trace = pm.sample(4000, return_inferencedata=True, 
                         idata_kwargs={"log_likelihood": True})

    return model, trace, ica


def compute_metrics(trace, X, y, transformer=None):
    beta_key = [key for key in trace.posterior.keys() if "beta" in key][0]
    beta_samples = trace.posterior[beta_key].mean(dim=["chain", "draw"]).values

    if transformer is not None:  # Convert back to original feature space
        beta_samples = transformer.components_.T @ beta_samples

    y_pred = X @ beta_samples
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    return rmse, beta_samples


def evaluate_model_performance(estimated_beta, true_beta):
    # Ensure both arrays have same length by padding estimated_beta with zeros if needed
    if len(estimated_beta) < len(true_beta):
        padded_estimated = np.zeros(len(true_beta))
        padded_estimated[:len(estimated_beta)] = estimated_beta
        estimated_beta = padded_estimated
    return np.sqrt(np.mean((true_beta - estimated_beta) ** 2))  # RMSE



"""def run_models_and_evaluate(n=20, p=3, true_beta=None, n_components=2, high_corr=False):
    #
    Main function to run all models and evaluate performance
    
    Args:
        n (int): Number of samples
        p (int): Number of predictors
        true_beta (array): True coefficients (if None, will generate)
        n_components (int): Number of components for PCR/ICR
        high_corr (bool): Whether to generate high-correlation data
        
    Returns:
        tuple: RMSE values for all models
    #
    if true_beta is None:
        true_beta = np.logspace(0, 1, p, base=2)  # Generate true_beta based on p

    p = len(true_beta)
    sigma_true = 1

    if not high_corr:
        # Generate simple alternating pattern data
        X = np.ones((n, p))
        for i in range(0, X.shape[1], 2):
            X[i::2, i] = 0
            X[i+1::2, i] = 1
        y = np.dot(X, true_beta) + stats.norm(0, sigma_true).rvs(n)
    else:
        X, y, true_beta = generate_high_dim_data(n=n, p=p)

    # Run all models
    _, trace_mcmc = bayesian_regression_mcmc(X, y, true_beta)
    _, trace_ridge = bayesian_ridge_regression(X, y, true_beta)
    _, trace_lasso = bayesian_lasso(X, y, true_beta)
    _, trace_robust = bayesian_robust_regression(X, y, true_beta)
    _, trace_vi = bayesian_regression_vi(X, y, true_beta)
    
    # Dimensionality reduction models
    pca = PCA(n_components=min(n_components, p))
    X_pca = pca.fit_transform(X)
    _, trace_pcr, _ = bayesian_pcr(X_pca, y, true_beta, n_components)
    
    ica = FastICA(n_components=min(n_components, p), random_state=42)
    X_ica = ica.fit_transform(X)
    _, trace_icr, _ = bayesian_icr(X_ica, y, true_beta, n_components)

    # Extract and evaluate all results
    rmse_values = []
    for trace, transformer in zip(
        [trace_mcmc, trace_ridge, trace_lasso, trace_robust, trace_vi, trace_pcr, trace_icr],
        [None, None, None, None, None, pca, ica]
    ):
        beta = trace.posterior['beta'].mean(dim=('chain', 'draw')).values
        if transformer is not None:
            beta = transformer.components_.T @ beta
        rmse_values.append(evaluate_model_performance(beta, true_beta))
    
    return tuple(rmse_values)

"""

def generate_high_dim_data(n=50, p=100, true_signal_indices=None, true_beta_values=[3.0, -2.0, 4.0], noise_level=1.0):
    """Generate high-dimensional data with sparse true signals."""
    if true_signal_indices is None:
        # Dynamically place signals at 25%, 50%, 75% of p (but ensure they're within bounds)
        true_signal_indices = [int(p * 0.25), int(p * 0.5), int(p * 0.75)]
        true_signal_indices = [min(idx, p-1) for idx in true_signal_indices]  # Ensure no out-of-bounds
    
    true_beta = np.zeros(p)
    true_beta[true_signal_indices] = true_beta_values  # Only a few true signals

    # Generate X with a few latent factors + noise
    latent_dim = len(true_signal_indices)
    latent_factors = np.random.randn(n, latent_dim)  # Latent factors driving y
    X = np.hstack([
        latent_factors[:, 0:1] * np.random.randn(n, p // 3),  # Group 1: Correlated with factor 1
        latent_factors[:, 1:2] * np.random.randn(n, p // 3),  # Group 2: Correlated with factor 2
        latent_factors[:, 2:3] * np.random.randn(n, p // 3),  # Group 3: Correlated with factor 3
        np.random.randn(n, p - 3 * (p // 3))  # Pure noise
    ])

    # y depends only on the latent factors
    y = np.dot(latent_factors, true_beta_values) + stats.norm(0, noise_level).rvs(n)
    return X, y, true_beta

import time
from functools import wraps

def time_limited(limit=30):
    """Decorator to skip function if it runs longer than limit seconds."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = None
            exception = None
            
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                exception = e
            
            elapsed = time.time() - start_time
            if elapsed > limit:
                print(f"Skipping {func.__name__} - exceeded time limit of {limit}s (took {elapsed:.2f}s)")
                return None
            elif exception is not None:
                print(f"Skipping {func.__name__} - encountered error: {str(exception)}")
                return None
            
            return result
        return wrapper
    return decorator

def run_models_and_evaluate(n=20, p=3, true_beta=None, n_components=2, high_corr=False, time_limit=30):
    """
    Main function to run all models and evaluate performance with time limits
    
    Args:
        n (int): Number of samples
        p (int): Number of predictors
        true_beta (array): True coefficients (if None, will generate)
        n_components (int): Number of components for PCR/ICR
        high_corr (bool): Whether to generate high-correlation data
        time_limit (int): Maximum seconds allowed per model
        
    Returns:
        tuple: RMSE values for all models (None for skipped models)
    """
    if true_beta is None:
        true_beta = np.logspace(0, 1, p, base=2)

    p = len(true_beta)
    sigma_true = 1

    if not high_corr:
        X = np.ones((n, p))
        for i in range(0, X.shape[1], 2):
            X[i::2, i] = 0
            X[i+1::2, i] = 1
        y = np.dot(X, true_beta) + stats.norm(0, sigma_true).rvs(n)
    else:
        X, y, true_beta = generate_high_dim_data(n=n, p=p)

    # Time-limited versions of model functions
    @time_limited(time_limit)
    def run_mcmc(): return bayesian_regression_mcmc(X, y, true_beta)
    
    @time_limited(time_limit)
    def run_ridge(): return bayesian_ridge_regression(X, y, true_beta)
    
    @time_limited(time_limit)
    def run_lasso(): return bayesian_lasso(X, y, true_beta)
    
    @time_limited(time_limit)
    def run_robust(): return bayesian_robust_regression(X, y, true_beta)
    
    @time_limited(time_limit)
    def run_vi(): return bayesian_regression_vi(X, y, true_beta)

    # Run models with time limits
    results = []
    traces = []
    transformers = []
    
    for name, runner in [
        ('MCMC', run_mcmc),
        ('Ridge', run_ridge),
        ('Lasso', run_lasso),
        ('Robust', run_robust),
        ('VI', run_vi)
    ]:
        print(f"\nRunning {name}...")
        result = runner()
        if result is not None:
            traces.append(result[1])
            transformers.append(None)
    
    # Dimensionality reduction models
    pca = PCA(n_components=min(n_components, p))
    X_pca = pca.fit_transform(X)
    
    @time_limited(time_limit)
    def run_pcr(): return bayesian_pcr(X_pca, y, true_beta, n_components)
    
    print("\nRunning PCR...")
    pcr_result = run_pcr()
    if pcr_result is not None:
        traces.append(pcr_result[1])
        transformers.append(pca)
    
    ica = FastICA(n_components=min(n_components, p), random_state=42)
    X_ica = ica.fit_transform(X)
    
    @time_limited(time_limit)
    def run_icr(): return bayesian_icr(X_ica, y, true_beta, n_components)
    
    print("\nRunning ICR...")
    icr_result = run_icr()
    if icr_result is not None:
        traces.append(icr_result[1])
        transformers.append(ica)

    # Evaluate completed models
    rmse_values = []
    for trace, transformer in zip(traces, transformers):
        try:
            beta = trace.posterior['beta'].mean(dim=('chain', 'draw')).values
            if transformer is not None:
                beta = transformer.components_.T @ beta
            rmse = evaluate_model_performance(beta, true_beta)
            rmse_values.append(rmse)
        except Exception as e:
            print(f"Error evaluating model: {str(e)}")
            rmse_values.append(None)
    
    # Pad with Nones for skipped models to maintain order
    expected_models = 7  # Original number of models
    while len(rmse_values) < expected_models:
        rmse_values.append(None)
    
    return tuple(rmse_values)

import time
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import traceback
import sys

def run_model_safely(model_func, args, kwargs):
    """Wrapper to run model and capture output safely"""
    old_stdout = sys.stdout
    try:
        # Redirect stdout temporarily to avoid rich/IPython recursion
        sys.stdout = open('/dev/null', 'w') if sys.platform != 'win32' else open('nul', 'w')
        result = model_func(*args, **kwargs)
        return result
    except Exception as e:
        raise e
    finally:
        sys.stdout = old_stdout

def run_and_plot_models(X, y, true_beta=None, n_components=2, time_limit=30):
    """
    Run Bayesian models with proper timeout functionality and plot results
    
    Args:
        X: Input features
        y: Target variable
        true_beta: True coefficients (optional)
        n_components: Number of components for PCR/ICR
        time_limit: Maximum seconds allowed per model
        
    Returns:
        Dictionary containing results and metrics
    """
    models = {
        "Bayesian Linear Regression": bayesian_regression_mcmc,
        "Bayesian Ridge Regression": bayesian_ridge_regression,
        "Bayesian Lasso": bayesian_lasso,
        "Bayesian Robust Regression": bayesian_robust_regression,
        "Bayesian Variational Inference": bayesian_regression_vi,
        "Bayesian PCR": bayesian_pcr,
        "Bayesian ICR": bayesian_icr,
    }
    
    evaluation_mode = true_beta is not None
    num_betas = len(true_beta) if evaluation_mode else X.shape[1]
    
    results = []
    model_traces = {}
    model_metrics = {}
    model_extras = {}
    skipped_models = []
    
    if evaluation_mode:
        active_models = [name for name in models if "PCR" not in name and "ICR" not in name]
        fig, axes = plt.subplots(num_betas, len(active_models), 
                               figsize=(18, 3 * num_betas),
                               sharex=True, sharey=True)
        axes = axes.reshape(num_betas, len(active_models))
    
    # Run models one at a time with timeout
    for col, (model_name, model_func) in enumerate(models.items()):
        print(f"\nRunning {model_name}...")
        start_time = time.time()
        
        # Prepare arguments
        if model_name in ["Bayesian PCR", "Bayesian ICR"]:
            args = (X, y, true_beta, n_components) if evaluation_mode else (X, y)
            kwargs = {'n_components': n_components} if not evaluation_mode else {}
        else:
            args = (X, y, true_beta) if evaluation_mode else (X, y)
            kwargs = {}
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_model_safely, model_func, args, kwargs)
                result = future.result(timeout=time_limit)
                
                # Process successful result
                if model_name in ["Bayesian PCR", "Bayesian ICR"]:
                    model, trace, transformer = result
                    model_extras[model_name] = transformer
                else:
                    model, trace = result
                    transformer = None

                rmse, beta_estimates = compute_metrics(trace, X, y, transformer)
                
                results.append([model_name] + list(beta_estimates) + [rmse])
                model_traces[model_name] = trace
                model_metrics[model_name] = rmse
                
                if evaluation_mode and model_name not in ["Bayesian PCR", "Bayesian ICR"]:
                    try:
                        beta_key = [key for key in trace.posterior.keys() if "beta" in key][0]
                        for i in range(num_betas):
                            az.plot_posterior(trace.posterior[beta_key].sel(beta_dim_0=i),
                                             hdi_prob=0.95, ax=axes[i, col])
                            axes[i, col].set_title(f"{model_name} - Beta[{i}]")
                    except Exception as e:
                        print(f"Could not plot {model_name}: {str(e)}")
                        
                print(f"Completed {model_name} in {time.time()-start_time:.2f}s")
                
        except TimeoutError:
            print(f"Skipping {model_name} - exceeded time limit of {time_limit}s")
            skipped_models.append(model_name)
        except Exception as e:
            print(f"Skipping {model_name} - encountered error:\n{traceback.format_exc()}")
            skipped_models.append(model_name)
    
   