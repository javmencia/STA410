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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import traceback
import sys

# Models

## Bayesian Linear Regression Model
def bayesian_regression_mcmc(X, y, true_beta  = None, draws  =4000):
    if draws < 4000:
        tune = draws
    else:
        tune = 1000
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
        trace = pm.sample(draws  =draws, tune = tune, return_inferencedata=True, 
                          idata_kwargs={"log_likelihood": True})  # Ensure log likelihood is stored

    return model, trace

## Bayesian Ridge Regression Model
def bayesian_ridge_regression(X, y, true_beta  = None, draws = 4000):
    if draws < 4000:
        tune = draws
    else:
        tune = 1000
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
        trace = pm.sample(draws = draws, tune = tune, return_inferencedata=True,
                          idata_kwargs={"log_likelihood": True})
    return model, trace

## Bayesian Lasso Regression Model

def bayesian_lasso(X, y, true_beta=None, n_folds=5, draws  =4000):
    """Bayesian Lasso regression with cross-validated lambda selection"""
    if draws < 4000:
        tune = draws
    else:
        tune = 1000

    if true_beta is None:
        true_beta = np.zeros(X.shape[1])
    
    n_folds = min(n_folds, len(y))

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
        trace = pm.sample(draws = draws, tune = tune, return_inferencedata=True,
                         idata_kwargs={"log_likelihood": True})
    
    return model, trace


## Bayesian Robust Regression Model
def bayesian_robust_regression(X, y, true_beta  = None, draws =4000):
    if draws < 4000:
        tune = draws
    else:
        tune = 1000
    if true_beta is None:
        true_beta = np.zeros(X.shape[1])
    with pm.Model() as model:
        beta = pm.MvNormal("beta", mu=true_beta, chol=np.eye(X.shape[1]), shape=X.shape[1])
        sigma = pm.HalfCauchy("sigma", beta=2)
        nu = pm.Exponential("nu", 1/30)  # Degrees of freedom for the Student-T

        mu = pm.math.dot(X, beta)
        y_obs = pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(draws = draws, tune = tune, return_inferencedata=True,
                          idata_kwargs={"log_likelihood": True})

    return model, trace

## Variational Inference

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

## Principal Component Regression

def bayesian_pcr(X, y, true_beta=None, n_components=None, draws=  4000):
    if draws < 4000:
        tune = draws
    else:
        tune = 1000
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
        trace = pm.sample(draws = draws, tune = tune, return_inferencedata=True, idata_kwargs={"log_likelihood": True})

    return model, trace, pca  # Return PCA object for inverse transformation

## ICR

def bayesian_icr(X, y, true_beta=None, n_components=None, draws = 4000):
    if draws < 4000:
        tune = draws
    else:
        tune = 1000
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
        trace = pm.sample(draws  =draws, tune = tune, return_inferencedata=True, 
                         idata_kwargs={"log_likelihood": True})

    return model, trace, ica





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





# Apply to datasets


def run_model_tests(X, y, true_beta=None, n_components=2, max_test_time=5):
    """Run timing tests and return passing models
    Create test-chains to see if it is feasible to run full 5000 draw posterior chains or if we should skip that model. 
    This is done with a chain with just 30 draws and checking if it takes more than 5 seconds to sample from it """
    all_models = {
        "Bayesian Linear Regression": bayesian_regression_mcmc,
        "Bayesian Ridge Regression": bayesian_ridge_regression,
        "Bayesian Lasso": bayesian_lasso,
        "Bayesian Robust Regression": bayesian_robust_regression,
        "Bayesian Variational Inference": bayesian_regression_vi,
        "Bayesian PCR": bayesian_pcr,
        "Bayesian ICR": bayesian_icr,
    }
    
    print(f"\n=== Running timing tests (max {max_test_time}s per model) ===")
    passing_models = {}
    failing_models = {}
    
    for model_name, model_func in all_models.items():
        print(f"\nTesting {model_name}...")
        
        # Prepare arguments
        if model_name in ["Bayesian PCR", "Bayesian ICR"]:
            args = (X, y, true_beta, n_components) if true_beta is not None else (X, y)
            kwargs = {'n_components': n_components, 'draws': 15} if true_beta is None else {'draws': 15}
        elif model_name == "Bayesian Variational Inference":
            args = (X, y, true_beta) if true_beta is not None else (X, y)
            kwargs = {}
        else:
            args = (X, y, true_beta) if true_beta is not None else (X, y)
            kwargs = {'draws': 15}
        
        try:
            start_time = time.time()
            result = model_func(*args, **kwargs)
            duration = time.time() - start_time
            
            if duration > max_test_time:
                failing_models[model_name] = f"Too slow ({duration:.2f}s > {max_test_time}s)"
                print(f"  → Failed: took {duration:.2f}s (timeout)")
            else:
                passing_models[model_name] = model_func
                print(f"  → Passed in {duration:.2f}s")
                
        except Exception as e:
            failing_models[model_name] = str(e)
            print(f"  → Failed: {str(e)}")
    
    # Show which models were filtered out
    if failing_models:
        print("\n=== Models that didn't pass timing tests ===")
        for name, reason in failing_models.items():
            print(f"{name}: {reason}")
    
    return passing_models

def run_and_plot_models(X, y, true_beta=None, n_components=2, max_test_time=5):
    """
    Complete workflow:
    1. Run timing tests to filter models
    2. Run full analysis on passing models
    3. Return results (RMSE and beta estimates)
    """
    # Step 1: Get passing models
    passing_models = run_model_tests(X, y, true_beta, n_components, max_test_time)
    
    # Step 2: Run full analysis
    print("\n=== Running full analysis on passing models ===")
    results = []
    model_traces = {}
    transformers = {}
    
    for model_name, model_func in passing_models.items():
        print(f"\nRunning {model_name}...")
        
        # Prepare arguments for full run
        if model_name in ["Bayesian PCR", "Bayesian ICR"]:
            args = (X, y, true_beta, n_components) if true_beta is not None else (X, y)
            kwargs = {'n_components': n_components, 'draws': 4000} if true_beta is None else {'draws': 4000}
        elif model_name == "Bayesian Variational Inference":
            args = (X, y, true_beta) if true_beta is not None else (X, y)
            kwargs = {}
        else:
            args = (X, y, true_beta) if true_beta is not None else (X, y)
            kwargs = {'draws': 4000}
        
        try:
            result = model_func(*args, **kwargs)
            
            # Process results
            if model_name in ["Bayesian PCR", "Bayesian ICR"]:
                model, trace, transformer = result
                transformers[model_name] = transformer
            else:
                model, trace = result
            
            # Compute metrics
            rmse, beta_estimates = compute_metrics(trace, X, y, transformer if model_name in ["Bayesian PCR", "Bayesian ICR"] else None)
            
            results.append({
                'Model': model_name,
                'RMSE': rmse,
                'Beta Estimates': beta_estimates
            })
            model_traces[model_name] = trace
            
            print(f"Completed {model_name} successfully")
            
        except Exception as e:
            print(f"Failed to run {model_name}: {str(e)}")
            results.append({
                'Model': model_name,
                'RMSE': np.nan,
                'Beta Estimates': [np.nan] * X.shape[1],
                'Error': str(e)
            })
    
    # Step 3: Show final results
    print("\n=== Final Results (only passing models) ===")
    print("\n{:<30} {:<10} {}".format('Model', 'RMSE', 'Beta Estimates'))
    print("-" * 70)
    for res in sorted(results, key=lambda x: x['RMSE'] if not np.isnan(x['RMSE']) else float('inf')):
        beta_str = "  ".join([f"β{i}:{val:.3f}" for i, val in enumerate(res['Beta Estimates'])])
        print("{:<30} {:<10.4f} {}".format(
            res['Model'],
            res['RMSE'] if not np.isnan(res['RMSE']) else -1,
            beta_str
        ))
    
    return {
        'passing_models': passing_models,
        'results': results,
        'traces': model_traces,
        'transformers': transformers
    }



# Create datasets


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

# Helper Functions

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


# Data preprocessing pipeline with feature selection


def preprocess_data(df, response, top_n_features=None):
    """
    Robust data preprocessing pipeline that handles mixed data types
    
    Args:
        df: Input DataFrame
        response: Name of target variable column
        top_n_features: Number of top features to select (None keeps all)
        
    Returns:
        Tuple of (processed_features, target)
    """
    # Separate features and target
    y = df[response]
    X = df.drop(columns=[response])
    
    if top_n_features is None:
        top_n_features = len(X.columns)
    
    # First pass: convert all data to strings to handle mixed types
    X_str = X.astype(str)
    
    # Second pass: identify truly numeric columns
    numeric_cols = []
    categorical_cols = []
    
    for col in X_str.columns:
        # Try converting to numeric
        numeric_vals = pd.to_numeric(X_str[col], errors='coerce')
        if numeric_vals.notna().all():  # All values converted successfully
            numeric_cols.append(col)
            X_str[col] = numeric_vals  # Store as numeric
        else:
            categorical_cols.append(col)
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Apply preprocessing
    try:
        X_processed = preprocessor.fit_transform(X_str)
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        # Fallback: convert all to categorical if mixed types persist
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer(
            transformers=[('cat', categorical_transformer, X_str.columns)])
        X_processed = preprocessor.fit_transform(X_str)
    
    # Get feature names
    if len(categorical_cols) > 0:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        categorical_features = cat_encoder.get_feature_names_out(categorical_cols)
        all_features = numeric_cols + list(categorical_features)
    else:
        all_features = numeric_cols
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X_processed, columns=all_features)
    
    # Select top N features by variance
    if len(all_features) > top_n_features:
        selector = VarianceThreshold()
        selector.fit(X_df)
        variances = selector.variances_
        
        top_indices = np.argsort(variances)[-top_n_features:]
        selected_features = [all_features[i] for i in top_indices]
        
        print(f"\nSelected top {top_n_features} features from {len(all_features)} total features")
        print("Top 10 features by variance:")
        for feat in selected_features[:10]:
            print(f"- {feat}")
        
        return X_df[selected_features], y
    else:
        return X_df, y