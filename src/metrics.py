import numpy as np
import torch
from typing import Tuple, List, Dict, Any, Union, Optional

def dtw(pred: torch.Tensor, actual: torch.Tensor) -> float:
    """
    Calculate Dynamic Time Warping (DTW) distance between predicted and actual time series.
    
    DTW is a dynamic programming algorithm that measures similarity between two temporal sequences
    by finding the optimal alignment path that minimizes the cumulative distance between aligned points.
    
    Architecture:
        - Implements classic DTW with O(nm) time and space complexity, where n and m are sequence lengths
        - Uses L1 norm (absolute difference) as the local cost measure between points
        - Employs dynamic programming with a bottom-up approach to fill the cost matrix
        - Boundary conditions enforce sequence start/end alignment with infinity values
    
    Interface:
        - pred: PyTorch tensor or numpy array of predicted values (any shape, will be flattened)
          * Values should be numeric and finite
          * Empty sequences are not supported
        - actual: PyTorch tensor or numpy array of ground truth values (any shape, will be flattened)
          * Values should be numeric and finite
          * Empty sequences are not supported
        - Returns: Python float representing the minimum cumulative distance between sequences
          * Range: [0, +inf) where 0 indicates identical sequences
          * Not normalized by sequence length
    
    Behavior:
        - Detaches tensors from computation graph and moves to CPU before conversion to numpy
        - Thread-safe with no side effects or persistent state
        - Input tensors are not modified
        - Handles sequences of different lengths without padding
    
    Integration:
        - Designed for direct use in evaluation pipelines or via MetricsCalculator
        - Example:
          ```
          distance = dtw(model_predictions, ground_truth)
          print(f"DTW distance: {distance:.4f}")
          ```
    
    Limitations:
        - Standard DTW has quadratic complexity, becoming expensive for long sequences
        - No support for multidimensional time series (will be flattened to 1D)
        - Does not implement FastDTW or other approximation algorithms
        - No support for custom distance metrics beyond absolute difference
    
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
    
    The Frechet distance measures similarity between curves by finding the minimum maximum pointwise
    distance when traversing both curves simultaneously with varying speeds but maintaining order.
    Often described as the "walking dog distance" where a person and dog walk along separate paths.
    
    Architecture:
        - Implements discrete Frechet distance with O(nm) time and space complexity
        - Uses squared Euclidean distance as the local cost measure between points
        - Employs dynamic programming with a bottom-up approach to fill the distance matrix
        - Combines min and max operations to find optimal coupling that minimizes maximum distance
        - Returns the square root of the final matrix value as the true Frechet distance
    
    Interface:
        - pred: PyTorch tensor or numpy array of predicted values (any shape, will be flattened)
          * Values should be numeric and finite
          * Empty sequences are not supported
        - actual: PyTorch tensor or numpy array of ground truth values (any shape, will be flattened)
          * Values should be numeric and finite
          * Empty sequences are not supported
        - Returns: Python float representing the Frechet distance between sequences
          * Range: [0, +inf) where 0 indicates identical sequences
          * Units correspond to the original data units (not squared)
    
    Behavior:
        - Detaches tensors from computation graph and moves to CPU before conversion to numpy
        - Thread-safe with no side effects or persistent state
        - Input tensors are not modified
        - Handles sequences of different lengths without padding
        - Computes discrete rather than continuous Frechet distance
    
    Integration:
        - Designed for direct use in evaluation pipelines or via MetricsCalculator
        - Example:
          ```
          distance = frechet(model_predictions, ground_truth)
          print(f"Frechet distance: {distance:.4f}")
          ```
    
    Limitations:
        - Discrete Frechet has quadratic complexity, becoming expensive for long sequences
        - No support for multidimensional time series (will be flattened to 1D)
        - Does not implement approximation algorithms for faster computation
        - Assumes uniform importance of all points in the sequence
        - Not normalized by sequence length or value range
    
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
    
    MSE is a standard regression loss metric that measures the average squared difference
    between predicted and actual values, providing a quadratic penalty for larger errors.
    
    Architecture:
        - Implements standard MSE calculation with O(n) time and O(1) space complexity
        - Supports both PyTorch tensor and NumPy array inputs with automatic conversion
        - Vectorized implementation for optimal performance on both CPU and GPU
        - No gradient computation when detaching tensors for evaluation purposes
    
    Interface:
        - pred: PyTorch tensor or numpy array of predicted values (any shape, will be flattened)
          * Values should be numeric and finite
          * Empty sequences will return NaN
        - actual: PyTorch tensor or numpy array of ground truth values (any shape, will be flattened)
          * Values should be numeric and finite
          * Empty sequences will return NaN
        - Returns: Python float representing the mean squared error
          * Range: [0, +inf) where 0 indicates perfect prediction
          * Units are squared units of the original data
    
    Behavior:
        - Detaches tensors from computation graph and moves to CPU before conversion to numpy
        - Thread-safe with no side effects or persistent state
        - Input tensors are not modified
        - Handles tensors of different shapes by flattening before calculation
        - Returns scalar float value even for multi-dimensional inputs
    
    Integration:
        - Designed for direct use in evaluation pipelines or via MetricsCalculator
        - Example:
          ```
          error = mse(model_predictions, ground_truth)
          print(f"MSE: {error:.4f}")
          ```
    
    Limitations:
        - Sensitive to outliers due to squared penalty
        - Not scale-invariant (depends on the magnitude of the data)
        - No support for weighted errors or masked values
        - Does not normalize by data range, making cross-dataset comparison difficult
    
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
    Utility class for calculating and aggregating evaluation metrics for time series prediction models.
    
    This class provides a standardized interface for computing multiple performance metrics
    between predicted and actual time series data. It implements a stateless design pattern
    with static methods to facilitate integration with training and evaluation pipelines.
    
    Architecture:
        - Implements a facade pattern over individual metric functions
        - O(1) dispatch complexity to appropriate metric implementations
        - Underlying metric implementations have varying complexities:
          * MSE: O(n) time, O(1) space
          * DTW: O(nm) time, O(nm) space where n,m are sequence lengths
          * Frechet: O(nm) time, O(nm) space where n,m are sequence lengths
    
    Interface:
        - All methods accept PyTorch tensors and handle conversion to appropriate formats
        - Tensor shapes are automatically normalized to 1D for consistent processing
        - Results are returned as Python native types (float) for serialization compatibility
    
    Thread Safety:
        - All methods are stateless and thread-safe
        - No shared state or resources are maintained between calls
    
    Integration:
        - Designed to be called directly from training loops or evaluation scripts
        - Compatible with Weights & Biases and other metric tracking systems
        - Example usage:
          ```
          metrics = MetricsCalculator.calculate_metrics(
              model_output, ground_truth, ["mse", "dtw"]
          )
          wandb.log(metrics)
          ```
    
    Limitations:
        - Does not support batched metric calculation (processes single samples)
        - Large sequence comparisons (DTW, Frechet) may become computationally expensive
        - All metrics assume 1D time series data
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
