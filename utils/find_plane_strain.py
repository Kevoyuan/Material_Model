
import pandas as pd
import numpy as np
from pathlib import Path

def find_strain(strain_path: Path, eq_strain_path: Path, target_value: float) -> tuple:
    """Calculate strain values using vectorized operations with error handling."""
    try:
        # Read data using context manager
        df_strain = pd.read_csv(strain_path, header=1)
        df_eq = pd.read_csv(eq_strain_path, header=1)
        
        # Validate data structure
        if df_strain.shape[1] < 2 or df_eq.shape[1] < 2:
            raise ValueError("Invalid CSV file structure")

        # Process using vectorized operations
        x = df_strain.iloc[:, 0].values
        y = pd.to_numeric(df_strain.iloc[:, 1], errors='coerce').values
        eq_strain = pd.to_numeric(df_eq.iloc[:, 1], errors='coerce').values

        # Find nearest strain using numpy
        valid_idx = np.where(np.isfinite(y))[0]
        if valid_idx.size == 0:
            return (0.0, 0.0)
            
        nearest_idx = np.abs(y[valid_idx] - target_value).argmin()
        strain_position = x[valid_idx][nearest_idx]
        
        # Calculate adjusted position
        strain_position = abs(strain_position - 0.5 * x.max())
        return (eq_strain[valid_idx][nearest_idx], strain_position)
        
    except (FileNotFoundError, pd.errors.ParserError) as e:
        print(f"Error processing files: {e}")
        return (0.0, 0.0)

def find_plane_strain(fem_model: str) -> tuple:
    """Find plane strain location using optimized path handling."""
    model_path = Path(fem_model)
    return find_strain(
        model_path/"y_strain.csv",
        model_path/"StrainCurve.csv",
        target_value=0.0
    )