import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    
    def fit(self, x:np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum=x.min(axis=0)
        self.maximum=x.max(axis=0)
        
    def transform(self, x:np.ndarray) -> list:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        # Avoid division by zero for features with identical min and max values
        eps = np.finfo(float).eps  # epsilon, a small value to avoid division by zero
        return (x - self.minimum) / np.maximum(diff_max_min, eps)
    
    def fit_transform(self, x:list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
    
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None 

    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x

    def fit(self, x: np.ndarray) -> None:
        x = self._check_is_array(x)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)  # Calculate standard deviation

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        return (x - self.mean) / np.maximum(self.std, np.finfo(float).eps)  # Avoid division by zero

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Fits the StandardScaler to the given data and transforms it.

        Args:
            x: The input data as a NumPy array.

        Returns:
            The transformed data.
        """

        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
    
class LabelEncoder:
    """
    Encodes categorical labels with integer values.

    Similar to scikit-learn's LabelEncoder class.
    """
    def __init__(self):
        self.classes_ = None
    
    def fit(self, y):
        """
        Fits the LabelEncoder to the data.

        Args:
            y: The data to fit, a NumPy array.
        """
        # Convert to numpy array
        y = np.array(y)
        # Find unique classes
        self.classes_ = np.unique(y)
        return self
    
    def transform(self, y):
        """
        Transforms categorical labels to integer values.

        Args:
            y: The data to transform, a NumPy array.

        Returns:
            A NumPy array of integer labels.
        """
        # Convert to numpy array
        y = np.array(y)
        if self.classes_ is None:
            raise ValueError("No classes")
        
        # Match the indices with the classes array 
        return np.array([np.where(self.classes_ == label)[0][0] for label in y])
    
    def fit_transform(self, y):
        """
        Fits the LabelEncoder to the data and transforms it.

        Args:
            y: The data to fit and transform, a NumPy array.

        Returns:
            A NumPy array of integer labels.
        """
        self.fit(y)
        return self.transform(y)