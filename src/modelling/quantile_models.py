import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_and_evaluate_quantile_models(
    df: pd.DataFrame,
    key_column_name: str = 'key',
    target_column_name: str = 'target',
    test_size: float = 0.1,
    random_state: int = 42,
    quantiles: List[float] = [0.1, 0.3, 0.7],
    alpha: float = 0.01,
    solver: str = 'highs',
    verbose: bool = True
) -> Tuple[Dict[str, QuantileRegressor], pd.DataFrame, Dict[str, float]]:
    """
    Train multiple quantile regression models and evaluate their performance.
    
    This function trains quantile regression models for specified quantiles, makes
    predictions on the entire dataset, and evaluates the models' performance using
    various metrics including MAE, RMSE, and prediction interval coverage.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing features, key column, and target column.
    key_column_name : str, default='key'
        The name of the column containing unique identifiers for each row.
    target_column_name : str, default='target'
        The name of the column containing the target variable to predict.
    test_size : float, default=0.1
        The proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random seed for reproducibility.
    quantiles : List[float], default=[0.1, 0.3, 0.7]
        List of quantiles to train models for.
    alpha : float, default=0.01
        L1 regularization parameter for the quantile regressor.
    solver : str, default='highs'
        The solver to use for optimization.
    verbose : bool, default=True
        Whether to print progress and result information.
    
    Returns
    -------
    Tuple[Dict[str, QuantileRegressor], pd.DataFrame, Dict[str, float]]
        A tuple containing:
        - Dictionary mapping quantile to trained model
        - DataFrame with original data and predictions
        - Dictionary of evaluation metrics
        
    Example
    -------
    >>> df = pd.read_csv("data/y_and_x_ready_for_training.csv")
    >>> models, results_df, metrics = train_and_evaluate_quantile_models(df)
    """
    # Save the key column for later merging
    key_column = df[key_column_name].copy()
    
    # Drop the key and target columns from features
    df_features = df.drop([key_column_name, target_column_name], axis=1)
    target = df[target_column_name]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, target, test_size=test_size, random_state=random_state
    )
    
    # Use the entire dataset for evaluation
    X_full = df_features
    y_full = target
    
    # Initialize dictionaries to store models and predictions
    models = {}
    predictions = {}
    metrics = {}
    
    # Train models for each quantile
    for quantile in quantiles:
        if verbose:
            print(f"Training {quantile:.1%} percentile model...")
        
        # Create and train the model
        model = QuantileRegressor(quantile=quantile, alpha=alpha, solver=solver)
        model.fit(X_train, y_train)
        
        # Store the model
        models[f"model_{int(quantile*100):d}th"] = model
        
        # Make predictions on the full dataset
        pred = model.predict(X_full)
        predictions[f"prediction_{int(quantile*100):d}th_percentile"] = pred
        
        # Calculate MAE and RMSE
        mae = mean_absolute_error(y_full, pred)
        rmse = np.sqrt(mean_squared_error(y_full, pred))
        
        # Store metrics
        metrics[f"{int(quantile*100):d}th_percentile_mae"] = mae
        metrics[f"{int(quantile*100):d}th_percentile_rmse"] = rmse
    
    if verbose:
        print("Making predictions on whole dataset...")
    
    # Create results dataframe with predictions for the whole dataset
    results_data = {'key': key_column, 'actual_target': target}
    results_data.update(predictions)
    results_df = pd.DataFrame(results_data)
    
    # Calculate prediction interval widths for common intervals
    if 0.1 in quantiles and 0.7 in quantiles:
        results_df['prediction_interval_width_10_70'] = (
            results_df['prediction_70th_percentile'] - 
            results_df['prediction_10th_percentile']
        )
        metrics['avg_prediction_interval_width_10_70'] = results_df['prediction_interval_width_10_70'].mean()
        
        # Check coverage for 10-70 interval
        within_interval_10_70 = (
            (results_df['actual_target'] >= results_df['prediction_10th_percentile']) & 
            (results_df['actual_target'] <= results_df['prediction_70th_percentile'])
        )
        coverage_10_70 = within_interval_10_70.mean()
        metrics['coverage_10_70'] = coverage_10_70
    
    if 0.3 in quantiles and 0.7 in quantiles:
        results_df['prediction_interval_width_30_70'] = (
            results_df['prediction_70th_percentile'] - 
            results_df['prediction_30th_percentile']
        )
        metrics['avg_prediction_interval_width_30_70'] = results_df['prediction_interval_width_30_70'].mean()
        
        # Check coverage for 30-70 interval
        within_interval_30_70 = (
            (results_df['actual_target'] >= results_df['prediction_30th_percentile']) & 
            (results_df['actual_target'] <= results_df['prediction_70th_percentile'])
        )
        coverage_30_70 = within_interval_30_70.mean()
        metrics['coverage_30_70'] = coverage_30_70
    
    # Display results if verbose mode is on
    if verbose:
        print("\nResults Summary:")
        print(f"Number of samples (whole dataset): {len(results_df)}")
        
        if 'avg_prediction_interval_width_10_70' in metrics:
            print(f"Average prediction interval width (10th-70th): "
                  f"{metrics['avg_prediction_interval_width_10_70']:.4f}")
        
        if 'avg_prediction_interval_width_30_70' in metrics:
            print(f"Average prediction interval width (30th-70th): "
                  f"{metrics['avg_prediction_interval_width_30_70']:.4f}")
        
        print("\nFirst 10 predictions:")
        print(results_df.head(10))
        
        print("\nModel Performance (whole dataset):")
        for quantile in quantiles:
            q_int = int(quantile*100)
            print(f"{q_int}th percentile MAE: {metrics[f'{q_int}th_percentile_mae']:.4f}")
        
        for quantile in quantiles:
            q_int = int(quantile*100)
            print(f"{q_int}th percentile RMSE: {metrics[f'{q_int}th_percentile_rmse']:.4f}")
        
        if 'coverage_10_70' in metrics:
            print(f"\nPrediction interval coverage (10th-70th): {metrics['coverage_10_70']:.2%}")
            print("(Expected coverage should be around 60% for 10th-70th percentile interval)")
        
        if 'coverage_30_70' in metrics:
            print(f"Prediction interval coverage (30th-70th): {metrics['coverage_30_70']:.2%}")
            print("(Expected coverage should be around 40% for 30th-70th percentile interval)")
    
    return models, results_df, metrics


def calculate_percentile_score(
    df: pd.DataFrame,
    key_column_name: str = 'key',
    actual_target_column: str = 'actual_target',
    prediction_10th_column: str = 'prediction_10th_percentile',
    prediction_30th_column: str = 'prediction_30th_percentile',
    prediction_70th_column: str = 'prediction_70th_percentile'
) -> pd.DataFrame:
    """
    Calculate a score based on actual_target position relative to prediction percentiles.
    
    The scoring logic is as follows:
    - 0: If actual_target > prediction_70th_percentile (worse than expected)
    - 1: If prediction_30th_percentile < actual_target <= prediction_70th_percentile
    - 2: If prediction_10th_percentile < actual_target <= prediction_30th_percentile
    - 3: If actual_target <= prediction_10th_percentile (better than expected)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns containing key, actual target and predictions
    key_column_name : str, default='key'
        Name of the column containing unique identifiers
    actual_target_column : str, default='actual_target'
        Name of the column containing actual target values
    prediction_10th_column : str, default='prediction_10th_percentile'
        Name of the column containing 10th percentile predictions
    prediction_30th_column : str, default='prediction_30th_percentile'
        Name of the column containing 30th percentile predictions
    prediction_70th_column : str, default='prediction_70th_percentile'
        Name of the column containing 70th percentile predictions
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'key' and 'score_where_worst_is_0'
        
    Example
    -------
    >>> results_df = pd.DataFrame({
    ...     'key': ['A', 'B', 'C', 'D'],
    ...     'actual_target': [5, 15, 25, 35],
    ...     'prediction_10th_percentile': [10, 10, 10, 10],
    ...     'prediction_30th_percentile': [20, 20, 20, 20],
    ...     'prediction_70th_percentile': [30, 30, 30, 30]
    ... })
    >>> calculate_percentile_score(results_df)
    """
    # Create a copy to avoid modifying the original DataFrame
    result_df = df[[key_column_name]].copy()
    
    # Define conditions in order (most restrictive first)
    conditions = [
        df[actual_target_column] > df[prediction_70th_column],  # score = 0
        df[actual_target_column] > df[prediction_30th_column],  # score = 1
        df[actual_target_column] > df[prediction_10th_column],  # score = 2
    ]
    
    # Corresponding scores for each condition
    choices = [0, 1, 2]
    
    # Apply the scoring logic using numpy.select
    # Default value (else case) is 3
    result_df['score_where_worst_is_0'] = np.select(conditions, choices, default=3)
    
    return result_df