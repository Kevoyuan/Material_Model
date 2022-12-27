import glob
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy.optimize import brute


def read_csv_files(filename):
    try:
        # Read the CSV file using pandas
        data = pd.read_csv(filename, header=0)
        column_headers = list(data.columns.values)

        # Extract the data columns from the DataFrame
        length = data[column_headers[0]]
        x = data[column_headers[1]]
        y = data[column_headers[2]]
        eps_x = data[column_headers[3]]
        eps_y = data[column_headers[4]]
        phiM = data[column_headers[5]]

        return length, x, y, eps_x, eps_y, phiM
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None, None, None, None, None, None
    except Exception as e:
        print(f"Error: An error occurred while reading the file: {e}")
        return None, None, None, None, None, None


def calibrate_length(x, y, group, label):
    """
    Calculates the calibrated length for the given x and y values.

    Parameters:
        x (list): The x values to use.
        y (list): The y values to use.
        space_left (list): The search space for the left side of the curve.
        space_right (list): The search space for the right side of the curve.
        label (str): The label for the data points.

    Returns:
        numpy array: The calibrated length values.
    """

    # Fit a polynomial curve to the data
    x_array = np.array(x)
    y_array = np.array(y)
    model = np.poly1d(np.polyfit(x_array, y_array, 30))

    def model_at_x(x):
        return model(x)

    # Use a dictionary to map group names to search spaces
    group_spaces = {
        "R": [[[3, 4]], [[12.4, 13.8]]],
        "6": [[[2, 4]], [[9, 11]]],
        "7": [[[2.5, 4]], [[10, 11.5]]],
    }
    try:
        # Find the minimum y-value of the curve fit using the brute() function
        space_left, space_right = group_spaces[group]

        result_left = brute(model_at_x, space_left)
        result_right = brute(model_at_x, space_right)
    except KeyError:
        raise ValueError("Invalid group name")

    # Calculate the calibrated length
    x_middle = (result_left[0] + result_right[0]) / 2

    calib_length = x - x_middle

    # Plot the data and the curve fit
    # plt.plot(x, y, "-", label=label)
    plt.scatter(result_left, model_at_x(result_left), color="b")

    plt.scatter(result_right, model_at_x(result_right), color="b")

    polyline = np.linspace(0, max(x), len(x))

    plt.plot(polyline, model(polyline), color="orange")

    plt.axvline(x=x_middle, color="r", linestyle="-.")

    return calib_length


def calib_model(x, y):
    x_array = np.array(x)

    y_array = np.array(y)

    polyline = np.linspace(x_array.min(), x_array.max(), 1000)

    model_calib = np.poly1d(np.polyfit(x_array, y_array, 30))
    # print(model)
    # y_fit_calib = model_calib(polyline)
    # plt.plot(polyline, y_fit_calib, color="orange")

    return model_calib


def model_at_x_new(x, model):
    """
    Calculates the y-position of a curve for a given set of x values.

    Parameters:
        x (list): The x values to use.
        model (function): The curve to use.

    Returns:
        numpy array: The y-position of the curve for the given x values.
    """
    y_fit = []
    # Align the x axis
    x_new = np.linspace(x.min(), x.max(), 1000)
    y_fit.append(model(x_new))
    y_fit = np.array(y_fit)
    return y_fit


def plot_diagrams(calib_length, eps_x, eps_y, phiM, label):
    """
    Plots various diagrams using the calibrated length, strain in the x direction, strain in the y direction,
    and bending moment.

    Parameters:
        calib_length (list): The calibrated length of the beam.
        eps_x (list): The strain in the x direction.
        eps_y (list): The strain in the y direction.
        phiM (list): The bending moment.
        label (str): The label to use for the plots.
    """
    # Plot the calibrated length vs. strain in the x direction
    plt.plot(calib_length, eps_x, ".", label=f"{label}_eps_x")

    # Plot the calibrated length vs. strain in the y direction
    plt.plot(calib_length, eps_y, ".", label=f"{label}_eps_y")

    # Plot the calibrated length vs. bending moment
    plt.plot(calib_length, phiM, ".", label=f"{label}_phiM")
    # plt.ylim(-0.1, 0.2)
    plt.legend()

    plt.grid(True, color="grey", linewidth="1.4", linestyle="-.")
    # plt.show()


def fit_curve(y_sum, calib_length, curve, label):

    # use calib_length to fit the model again

    model_calib = calib_model(calib_length, curve)

    y_fit = model_at_x_new(calib_length, model_calib)
    # print(y_fit)
    y_sum.append(y_fit)

    return y_sum


def avg_y_sum(y_sum, avg_calib_length, label):
    y_average = (y_sum[0] + y_sum[1] + y_sum[2]) / 3
    # print("\ny_avg: ", y_average)
    # x_new = np.linspace(min(length_min), max(length_max), 1000)
    # print(min(length_min), max(length_max))

    plt.xlabel("section along middle line in mm")
    plt.ylabel("eq. strain")

    plt.ylim(-0.1, 0.2)
    plt.xticks(np.linspace(-8, 8, 9))

    plt.scatter(avg_calib_length, y_average, s=1.0, label=label)

    return y_average


def save_to_csv(avg_calib_length, eps_x_avg, eps_y_avg, phiM_avg):

    avg_calib_length = np.asarray(avg_calib_length).flatten().tolist()

    eps_x_avg = np.asarray(eps_x_avg).flatten().tolist()

    eps_y_avg = np.asarray(eps_y_avg).flatten().tolist()
    phiM_avg = np.asarray(phiM_avg).flatten().tolist()

    # Create a dictionary with the data
    data = {
        "calib_length": avg_calib_length,
        "eps_x_avg": eps_x_avg,
        "eps_y_avg": eps_y_avg,
        "phiM_avg": phiM_avg,
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)

    return df


def main():

    path = "./YLD_2d_Investigation/experiment_data/test2/experiments"

    group = "6"
    displacement = "3p5"  # for different displacement: 2, 2p5, 3, 3p5
    csv_files = glob.glob(f"{path}/{group}*_{displacement}mm.csv")

    csv_files = sorted(csv_files)
    y_sum_eps_x = []
    y_sum_eps_y = []
    y_sum_phiM = []

    length_max = []
    length_min = []

    for filename in csv_files:

        if "\\" in filename:
            filename = filename.replace("\\", "/")
        print(filename)

        label = filename.split("/")[5].split(".")[0]

        # Read in the data from the CSV file
        length, x, y, eps_x, eps_y, phiM = read_csv_files(filename)

        # Calibrate the length
        calib_length = calibrate_length(length, phiM, group, label)
        # print("\n length: ", calib_length)

        # Plot the resulting diagrams
        # plot_diagrams(calib_length, eps_x, eps_y, phiM, label)

        # Fit a polynomial curve to the calibrated length and position data
        y_sum_eps_x = fit_curve(y_sum_eps_x, calib_length, eps_x, "eps_x")
        y_sum_eps_y = fit_curve(y_sum_eps_y, calib_length, eps_y, "eps_y")
        y_sum_phiM = fit_curve(y_sum_phiM, calib_length, phiM, "phiM")

        length_max.append(calib_length.max())
        length_min.append(calib_length.min())

    avg_calib_length = np.linspace(np.mean(length_min), np.mean(length_max), 1000)
    avg_length = avg_calib_length.reshape(1, -1)
    # print(avg_calib_length)

    # eps_x_avg = avg_y_sum(y_sum_eps_x, avg_length, "avg_eps_x")
    # eps_y_avg = avg_y_sum(y_sum_eps_y, avg_length, "avg_eps_y")
    # phiM_avg = avg_y_sum(y_sum_phiM, avg_length, "avg_phiM")

    plt.title(f"{group}_Strain Distribution in {displacement}mm")
    plt.grid(True, color="grey", linewidth="1.4", linestyle="-.")
    plt.legend()

    # plt.savefig(f"{path}/Strain_Distribution_{group}_{displacement}mm.png")

 
    plt.show()

    # # Save the DataFrame to a CSV file
    # df = save_to_csv(avg_calib_length, eps_x_avg, eps_y_avg, phiM_avg)

    # df.to_csv(f"{path}/avg_data_{group}_{displacement}mm.csv", index=False)


if __name__ == "__main__":
    main()
