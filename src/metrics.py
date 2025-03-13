import numpy as np
import torch
from typing import Tuple, List, Dict, Any, Union, Optional

def dtw(pred: torch.Tensor, actual: torch.Tensor) -> float:
    """
    Calculate Dynamic Time Warping (DTW) distance between predicted and actual time series.
    
    Args:
        pred: Predicted time series tensor
        actual: Actual time series tensor
        
    Returns:
        float: DTW distance
    """
    # Convert tensors to numpy arrays if they're not already
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    
    # Ensure inputs are 1D arrays
    pred = pred.reshape(-1)
    actual = actual.reshape(-1)
    
    # Get sequence lengths
    n, m = len(pred), len(actual)
    
    # Initialize the cost matrix
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix[0, 1:] = np.inf
    dtw_matrix[1:, 0] = np.inf
    
    # Fill the cost matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(pred[i-1] - actual[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1]     # match
            )
    
    # Return the DTW distance
    return float(dtw_matrix[n, m])

def frechet(pred: torch.Tensor, actual: torch.Tensor) -> float:
    """
    Calculate Frechet distance between predicted and actual time series.
    
    The Frechet distance is a measure of similarity between curves that takes into
    account the location and ordering of the points along the curves.
    
    Args:
        pred: Predicted time series tensor
        actual: Actual time series tensor
        
    Returns:
        float: Frechet distance
    """
    # Convert tensors to numpy arrays if they're not already
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    
    # Ensure inputs are 1D arrays
    pred = pred.reshape(-1)
    actual = actual.reshape(-1)
    
    # Get sequence lengths
    n, m = len(pred), len(actual)
    
    # Initialize the cost matrix
    ca = np.zeros((n, m))
    
    # Calculate the Euclidean distance between all pairs of points
    for i in range(n):
        for j in range(m):
            ca[i, j] = (pred[i] - actual[j]) ** 2
    
    # Initialize the Frechet distance matrix
    frechet_matrix = np.zeros((n + 1, m + 1))
    frechet_matrix[0, 0] = 0
    frechet_matrix[0, 1:] = np.inf
    frechet_matrix[1:, 0] = np.inf
    
    # Fill the Frechet distance matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            frechet_matrix[i, j] = max(
                min(
                    frechet_matrix[i-1, j],
                    frechet_matrix[i-1, j-1],
                    frechet_matrix[i, j-1]
                ),
                ca[i-1, j-1]
            )
    
    # Return the Frechet distance
    return float(np.sqrt(frechet_matrix[n, m]))

def mse(pred: torch.Tensor, actual: torch.Tensor) -> float:
    """
    Calculate Mean Squared Error between predicted and actual time series.
    
    Args:
        pred: Predicted time series tensor
        actual: Actual time series tensor
        
    Returns:
        float: MSE value
    """
    if isinstance(pred, torch.Tensor) and isinstance(actual, torch.Tensor):
        return torch.mean((pred - actual) ** 2).item()
    else:
        # Convert to numpy if needed
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(actual, torch.Tensor):
            actual = actual.detach().cpu().numpy()
        return float(np.mean((pred - actual) ** 2))

class MetricsCalculator:
    """
    Class to calculate and track various metrics for time series data.
    """
    
    @staticmethod
    def calculate_metrics(pred: torch.Tensor, actual: torch.Tensor, 
                          metrics_list: List[str]) -> Dict[str, float]:
        """
        Calculate specified metrics between predicted and actual time series.
        
        Args:
            pred: Predicted time series tensor
            actual: Actual time series tensor
            metrics_list: List of metric names to calculate
            
        Returns:
            Dict[str, float]: Dictionary of metric names and values
        """
        results = {}
        
        for metric_name in metrics_list:
            if metric_name == "mse":
                results["mse"] = mse(pred, actual)
            elif metric_name == "dtw":
                results["dtw"] = dtw(pred, actual)
            elif metric_name == "frechet":
                results["frechet"] = frechet(pred, actual)
            # Skip "loss" as it's calculated separately in the training loop
        
        return results
