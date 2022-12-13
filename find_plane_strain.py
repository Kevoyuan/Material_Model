
def find_strain(StrainPath, eq_StrainPath, strain_value):
    # calcuate the triaxial strain, plan strain and uniaxial strain
    # strain_value = 2/3, 1 / np.sqrt(3), 1/3

    df_var = pd.read_csv(StrainPath, header=1)

    column_headers = list(df_var.columns.values)

    x_var = df_var[column_headers[0]]

    y_var = df_var[column_headers[1]]

    y_lim_idx = y_var.argmax()
    x_lim = x_var[y_lim_idx]

    if x_lim > int(0.5 * x_var.max()):
        x_lim = x_var.max() - x_lim

    x_lim = x_lim - 0.8

    y_strain, strain_position_idx = find_nearest(y_var, value=strain_value)

    strain_position = x_var.iloc[strain_position_idx]

    df_strain = pd.read_csv(eq_StrainPath, header=1)

    column_headers_strain = list(df_strain.columns.values)

    df_eq_strain = df_strain[column_headers_strain[1]]

    target_strain = df_eq_strain.iloc[strain_position_idx]
    print("\ny_strain_0:", y_strain)

    print("\ntarget_strain:", target_strain)

    strain_position = abs(strain_position - 0.5 * x_var.max())
    print("\nplane_strain_position:", strain_position)

    return target_strain, strain_position

def find_plane_strain(fem_model):
    # # curve of Triaxiality
    y_strain_path = f"{fem_model}/y_strain.csv"

    # # Strain Curve
    Strain_path = f"{fem_model}/StrainCurve.csv"

    # plane strain occures in y_strain = 0
    y_strain = 0

    plane_strain, distance = find_strain(y_strain_path, Strain_path, y_strain)

    return plane_strain, distance