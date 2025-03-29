def calculate_strains(fem_model):
    """Calculate strain values from FEM simulation results with error handling."""
    import os
    import pandas as pd
    import numpy as np
    
    def read_strain_data(path):
        """Helper function to read and process strain CSV files."""
        try:
            df = pd.read_csv(path, header=1)
            distance = df.iloc[:, 0] - 0.5 * df.iloc[:, 0].max()
            strain = df.iloc[:, 1]
            return distance.values, strain.values
        except (FileNotFoundError, pd.errors.ParserError) as e:
            print(f"Error reading {path}: {e}")
            return None, None

    # Load data using helper function
    base_path = os.path.join(fem_model, "")
    x_dist, x_strain = read_strain_data(os.path.join(base_path, "x_strain.csv"))
    y_dist, y_strain = read_strain_data(os.path.join(base_path, "y_strain.csv"))
    eq_dist, eq_strain = read_strain_data(os.path.join(base_path, "StrainCurve.csv"))

    # Validate data
    if None in (x_strain, y_strain, eq_strain):
        return (0.0, 0.0, 0.0, 0.0)

    # Find intersection point using vectorized operations
    try:
        intersect_idx = np.argwhere(np.diff(np.sign(x_strain - y_strain))).flatten()
        if len(intersect_idx) == 0:
            raise ValueError("No intersection found between X and Y strains")
            
        biaxial_distance = x_dist[intersect_idx[0]]
        biaxial_value = x_strain[intersect_idx[0]]
    except Exception as e:
        print(f"Intersection error: {e}")
        biaxial_distance = biaxial_value = 0.0

    # Find plane strain location (nearest to zero in Y strain)
    try:
        plane_idx = np.argmin(np.abs(y_strain))
        plane_distance = abs(eq_dist[plane_idx])
        plane_value = eq_strain[plane_idx]
    except Exception as e:
        print(f"Plane strain error: {e}")
        plane_distance = plane_value = 0.0

    return (
        float(biaxial_distance), 
        float(eq_strain[intersect_idx[0]] if intersect_idx.size else 0.0),
        float(plane_distance),
        float(plane_value)
    )
