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

def run_and_plot_models(X, y, true_beta=None, n_components=2):
    models = {
        "Bayesian Linear Regression": bayesian_regression_mcmc,
        "Bayesian Ridge Regression": bayesian_ridge_regression,
        "Bayesian Robust Regression": bayesian_robust_regression,
        "Bayesian Variational Inference": bayesian_regression_vi,
        "Bayesian PCR": bayesian_pcr,
        "Bayesian ICR": bayesian_icr
    }
    
    evaluation_mode = true_beta is not None
    num_betas = len(true_beta) if evaluation_mode else X.shape[1]
    
    results = []
    model_traces = {}  # Store traces for all models
    model_metrics = {}  # Store metrics for all models
    
    if evaluation_mode:
        fig, axes = plt.subplots(num_betas, len(models) - 2, figsize=(18, 3 * num_betas),
                               sharex=True, sharey=True)
    
    for col, (model_name, model_func) in enumerate(models.items()):
        print(f"Running {model_name}...")
        
        if model_name in ["Bayesian PCR", "Bayesian ICR"]:
            if evaluation_mode:
                model, trace, transformer = model_func(X, y, true_beta, n_components)
            else:
                model, trace, transformer = model_func(X, y, n_components=n_components)
        else:
            model, trace = model_func(X, y, true_beta) if evaluation_mode else model_func(X, y)
            transformer = None

        rmse, beta_estimates = compute_metrics(trace, X, y, 
                                             transformer if model_name in ["Bayesian PCR", "Bayesian ICR"] else None)

        results.append([model_name] + list(beta_estimates) + [rmse])
        model_traces[model_name] = trace
        model_metrics[model_name] = rmse
        
        if evaluation_mode and model_name not in ["Bayesian PCR", "Bayesian ICR"]:
            beta_key = [key for key in trace.posterior.keys() if "beta" in key][0]
            for i in range(num_betas):
                az.plot_posterior(trace.posterior[beta_key].sel(beta_dim_0=i),
                                 hdi_prob=0.95, ax=axes[i, col])
                axes[i, col].set_title(f"{model_name} - Beta[{i}]")
    
    if evaluation_mode:
        plt.suptitle("Posterior Distributions (Excluding Dimensionality Reduction Models)", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

        # Show components info
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.text(0.5, 0.5, f"PCR Components: {n_components}", fontsize=14, ha='center', va='center')
        ax2.text(0.5, 0.5, f"ICR Components: {n_components}", fontsize=14, ha='center', va='center')
        for ax in [ax1, ax2]:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()
    
    # Create results table
    metric_name = "RMSE" if evaluation_mode else "MSE"
    columns = ["Model"] + [f"Beta[{i}]" for i in range(num_betas)] + [metric_name]
    results_df = pd.DataFrame(results, columns=columns)
    
    print("\nModel Performance Summary:")
    print(results_df)
    
    # Find the model with the lowest error
    best_model_name = min(model_metrics.items(), key=lambda x: x[1])[0]
    best_model_metric = model_metrics[best_model_name]
    best_model_trace = model_traces[best_model_name]
    
    print(f"\nBest model: {best_model_name} with {metric_name}: {best_model_metric:.4f}")
    
    return {
        'best_model': best_model_name,
        'best_metric': best_model_metric,
        'best_trace': best_model_trace,
        'all_results': results_df,
        'all_metrics': model_metrics,
        'all_traces': model_traces
    }

def evaluate_model_performance(estimated_beta, true_beta):
    # Ensure both arrays have same length by padding estimated_beta with zeros if needed
    if len(estimated_beta) < len(true_beta):
        padded_estimated = np.zeros(len(true_beta))
        padded_estimated[:len(estimated_beta)] = estimated_beta
        estimated_beta = padded_estimated
    return np.sqrt(np.mean((true_beta - estimated_beta) ** 2))  # RMSE

def run_models_and_evaluate(n=20, p=3, true_beta=None, n_components=2):
    if true_beta is None:
        if p is None:
            raise ValueError("Either true_beta or p must be provided.")
        true_beta = np.logspace(0, 1, p, base=2)  # Generate true_beta based on p

    p = len(true_beta)
    sigma_true = 1

    # Generate data
    X = np.ones((n, p))
    for i in range(0, X.shape[1], 2):
        X[i::2, i] = 0
        X[i+1::2, i] = 1

    y = np.dot(X, true_beta) + stats.norm(0, sigma_true).rvs(n)

    # Run models
    _, trace_mcmc = bayesian_regression_mcmc(X, y, true_beta)
    _, trace_ridge = bayesian_ridge_regression(X, y, true_beta)
    _, trace_robust = bayesian_robust_regression(X, y, true_beta)
    _, trace_vi = bayesian_regression_vi(X, y, true_beta)
    
    # Dimensionality reduction models
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    _, trace_pcr, _ = bayesian_pcr(X_pca, y, true_beta, n_components)
    
    ica = FastICA(n_components=n_components, random_state=42)
    X_ica = ica.fit_transform(X)
    _, trace_icr, _ = bayesian_icr(X_ica, y, true_beta, n_components)

    # Extract posterior means
    beta_mcmc = trace_mcmc.posterior['beta'].mean(dim=('chain', 'draw')).values
    beta_ridge = trace_ridge.posterior['beta'].mean(dim=('chain', 'draw')).values
    beta_robust = trace_robust.posterior['beta'].mean(dim=('chain', 'draw')).values
    beta_vi = trace_vi.posterior['beta'].mean(dim=('chain', 'draw')).values
    
    # For PCR/ICR, get coefficients in reduced space and transform back
    beta_pcr = trace_pcr.posterior['beta'].mean(dim=('chain', 'draw')).values
    beta_pcr_original = pca.components_.T @ beta_pcr
    
    beta_icr = trace_icr.posterior['beta'].mean(dim=('chain', 'draw')).values
    beta_icr_original = ica.components_.T @ beta_icr

    # Compute RMSE - now with dimension handling
    rmse_mcmc = evaluate_model_performance(beta_mcmc, true_beta)
    rmse_ridge = evaluate_model_performance(beta_ridge, true_beta)
    rmse_robust = evaluate_model_performance(beta_robust, true_beta)
    rmse_vi = evaluate_model_performance(beta_vi, true_beta)
    rmse_pcr = evaluate_model_performance(beta_pcr_original, true_beta)
    rmse_icr = evaluate_model_performance(beta_icr_original, true_beta)

    return rmse_mcmc, rmse_ridge, rmse_robust, rmse_vi, rmse_pcr, rmse_icr

