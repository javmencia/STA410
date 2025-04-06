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

def run_and_plot_models(X, y, true_beta=None, n_components=2):
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
    model_traces = {}  # Store traces for all models
    model_metrics = {}  # Store metrics for all models
    model_extras = {}   # Store any extra return values from models
    
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
            model_extras[model_name] = transformer
        elif model_name == "Bayesian Lasso":
            model, trace = model_func(X, y, true_beta) if evaluation_mode else model_func(X, y)
            transformer = None
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
        'all_traces': model_traces,
        'extras': model_extras
    }

def evaluate_model_performance(estimated_beta, true_beta):
    # Ensure both arrays have same length by padding estimated_beta with zeros if needed
    if len(estimated_beta) < len(true_beta):
        padded_estimated = np.zeros(len(true_beta))
        padded_estimated[:len(estimated_beta)] = estimated_beta
        estimated_beta = padded_estimated
    return np.sqrt(np.mean((true_beta - estimated_beta) ** 2))  # RMSE

def create_rmse_table(rmse_results, p_numbers):
    """
    Creates and displays a styled DataFrame showing RMSE values for each model at different predictor levels.
    
    Args:
        rmse_results (dict): Dictionary containing RMSE values for each model
        p_numbers (list): List of predictor counts tested
        
    Returns:
        pd.DataFrame: Styled DataFrame with RMSE results
    """
    # Create DataFrame from results
    results_df = pd.DataFrame(rmse_results, index=p_numbers)
    results_df.index.name = "Number of Predictors (p)"
    results_df.columns.name = "Model"
    
    # Apply styling
    styled_df = results_df.style\
        .format("{:.4f}")\
        .set_caption("RMSE Comparison Across Models and Predictor Counts")\
        .background_gradient(cmap='viridis', subset=pd.IndexSlice[:, :])\
        .highlight_min(axis=1, color='yellow')
    
    return styled_df

def run_models_and_evaluate(n=20, p=3, true_beta=None, n_components=2, high_corr=False):
    """
    Main function to run all models and evaluate performance
    
    Args:
        n (int): Number of samples
        p (int): Number of predictors
        true_beta (array): True coefficients (if None, will generate)
        n_components (int): Number of components for PCR/ICR
        high_corr (bool): Whether to generate high-correlation data
        
    Returns:
        tuple: RMSE values for all models
    """
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

# Main execution
if __name__ == "__main__":
    p_numbers = [5, 20, 30, 60, 100]
    n = 20
    rmse_results = {"MCMC": [], "Ridge": [], "Lasso": [], "Robust": [], "VI": [], "PCR": [], "ICR": []}

    for p in p_numbers:
        print(f"Running models for p={p}")
        rmse_values = run_models_and_evaluate(n=n, p=p, high_corr=True)
        for model, rmse in zip(rmse_results.keys(), rmse_values):
            rmse_results[model].append(rmse)

    # Create and display table
    rmse_table = create_rmse_table(rmse_results, p_numbers)
    display(rmse_table)

    # Plot results
    plt.figure(figsize=(10, 6))
    for model, values in rmse_results.items():
        plt.plot(p_numbers, values, marker='o', label=model)
    plt.xlabel("Number of Features (p)")
    plt.ylabel("RMSE (vs. True Coefficients)")
    plt.title("Model Performance as Dimensionality Increases")
    plt.legend()
    plt.grid(True)
    plt.show()

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

