
import pandas as pd
import numpy as np
from intersect import intersection


def calculate_strains(fem_model):
    x_path = f"{fem_model}/x_strain.csv"
    y_path = f"{fem_model}/y_strain.csv"
    eq_path = f"{fem_model}/StrainCurve.csv"

    dfx = pd.read_csv(x_path, header=1)

    column_headers_x = list(dfx.columns.values)

    dfx_distance = dfx[column_headers_x[0]]

    dfx_strain = dfx[column_headers_x[1]]
    dfx_distance = dfx_distance - 0.5 * dfx_distance.max()

    x1 = np.array(dfx_distance).reshape(-1, 1)
    y1 = np.array(dfx_strain).reshape(-1, 1)

    dfy = pd.read_csv(y_path, header=1)

    column_headers_y = list(dfy.columns.values)

    # dfy_distance = dfy[column_headers_y[0]]

    dfy_strain = dfy[column_headers_y[1]]

    x2 = x1

    y2 = np.array(dfy_strain).reshape(-1, 1)

    # find intersection of x and y strain
    x, y = intersection(x1, y1, x1, y2)

    dfeq = pd.read_csv(eq_path, header=1)

    column_headers_eq = list(dfeq.columns.values)

    dfeq_distance = dfeq[column_headers_eq[0]]

    dfeq_strain = dfeq[column_headers_eq[1]]
    dfeq_distance = dfeq_distance - 0.5 * dfeq_distance.max()

    print(f"\nx = {x}\ny = {y}")

    biaxial_distance = x[1]

    # locate the index of biaxial strain
    bi_distance, bi_idx = find_nearest(y1, value=y[1])
    print(f"idx: {bi_idx}")

    # biaxial_strain
    biaxial_strain = dfeq_strain[bi_idx]

    # locate the index of plane strain
    pl_distance, plane_idx = find_nearest(y2, value=0)

    # plane_strain
    plane_strain = dfeq_strain[plane_idx]
    plane_distance = abs(dfeq_distance[plane_idx])

    print(f"\n\nbixaial distance1 = {biaxial_distance}")

    print(f"\n\nbixaial strain = {biaxial_strain}")

    print(f"\n\nplane distance = {plane_distance}")

    print(f"\n\nplane strain = {plane_strain}")
    return biaxial_distance, biaxial_strain, plane_distance, plane_strain

