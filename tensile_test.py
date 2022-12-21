import glob
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from FEM_model_modify import read_subfolders, plot_3_strains


from scipy.optimize import minimize
from scipy.optimize import brute


def read_csv_files(filename):

    data = pd.read_csv(
        filename,
        header=0,
    )
    column_headers = list(data.columns.values)
    # print(column_headers)

    length = data[column_headers[0]]
    x = data[column_headers[1]]
    y = data[column_headers[2]]
    eps_x = data[column_headers[3]]
    eps_y = data[column_headers[4]]
    phiM = data[column_headers[5]]

    return length, x, y, eps_x, eps_y, phiM


def calibrate_length(x, y, group):
    x_array = np.array(x)
    x_newarr = np.array_split(x_array, 2)

    y_array = np.array(y)
    y_newarr = np.array_split(y_array, 2)

    polyline = np.linspace(0, max(x), len(x))

    model = np.poly1d(np.polyfit(x_array, y_array, 30))
    # print(model)
    y_fit = model(polyline)

    # plt.plot(polyline, model(polyline), color="orange")

    # Find the minimum y-value of the curve fit

    # result = minimize(model, x0=x0, bounds=bounds, args=(x_array))

    def model_at_x(x):
        return model(x)

    if group == "R":
        ########################################################################
        # # Set the initial guess for the x-value: R
        x0_left = [4]
        x0_right = [13.4]

        # Set the bounds for the x-value
        bounds_left = [[3.8, 4]]
        bounds_right = [[12.5, 13.8]]

    elif group == "6":
        ########################################################################
        # Set the initial guess for the x-value: 6
        x0_left = [3.2]
        x0_right = [10.5]

        # Set the bounds for the x-value
        bounds_left = [[2.5, 5]]
        bounds_right = [[9.5, 11]]

    elif group == "7":
        ########################################################################
        # # Set the initial guess for the x-value: 7
        x0_left = [3.3]
        x0_right = [10.7]

        # Set the bounds for the x-value
        bounds_left = [[2.5, 4]]
        bounds_right = [[10, 11.5]]

    # Find the minimum y-value of the curve fit
    result_left = minimize(model_at_x, x0=x0_left, bounds=bounds_left)
    result_right = minimize(model_at_x, x0=x0_right, bounds=bounds_right)

    # Print the minimum y-value and the corresponding x-value
    print(f"Minimum y-value: {result_left.fun}")
    print(f"Corresponding x-value: {result_left.x}")

    print(f"Corresponding x-value: {result_right.x}")

    x_middle = (result_left.x + result_right.x) / 2
    # plt.axvline(x=x_middle, color="r", linestyle="-.")
    print("\nx_middle: ", x_middle)

    calib_length = x - x_middle

    # plt.scatter(x, y, s=1)
    # plt.scatter(result_left.x, result_left.fun, color="b")
    # plt.scatter(result_right.x, result_right.fun, color="b")

    # plt.show()
    return calib_length


def calib_model(x, y):
    x_array = np.array(x)

    y_array = np.array(y)

    polyline = np.linspace(x_array.min(), x_array.max(), 1000)

    model_calib = np.poly1d(np.polyfit(x_array, y_array, 30))
    # print(model)
    y_fit_calib = model_calib(polyline)
    # plt.plot(polyline, y_fit_calib, color="orange")

    return model_calib


def model_at_x_new(x, model):
    y_fit = []
    # print(length_max,length_min)
    # align the x axis
    x_new = np.linspace(x.min(), x.max(), 1000)
    y_fit.append(model(x_new))
    y_fit = np.array(y_fit)
    # print(y_fit)
    return y_fit


def adjR(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results["r_squared"] = 1 - (
        ((1 - (ssreg / sstot)) * (len(y) - 1)) / (len(y) - degree - 1)
    )

    return print(results)


def plot_diagrams(calib_length, eps_x, eps_y, phiM, label):
    # # length = length - 0.5*max(length)
    # plt.plot(calib_length, eps_x, ".", label=f"{label}_eps_x")
    # plt.plot(calib_length, eps_y, ".", label=f"{label}_eps_y")
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

    # print(length_min, length_max)
    # print("\ny_sum: \n",y_sum)

    return y_sum


def avg_y_sum(y_sum, avg_calib_length, label):
    y_average = (y_sum[0] + y_sum[1] + y_sum[2]) / 3
    # print("\ny_avg: ", y_average)
    # x_new = np.linspace(min(length_min), max(length_max), 1000)
    # print(min(length_min), max(length_max))
    avg_calib_length = avg_calib_length.reshape((1, -1))
    plt.xlabel("section along middle line in mm")
    plt.ylabel("eq. strain")
    plt.grid(True, color="grey", linewidth="1.4", linestyle="-.")

    plt.ylim(-0.1, 0.2)
    plt.xticks(np.linspace(-8, 8, 9))

    plt.scatter(avg_calib_length[:, :], y_average[:, :], s=1.0, label=label)
    import pandas as pd

    plt.legend()


def main():

    path = "./YLD_2d_Investigation/experiment_data/test2/experiments"

    group = "R"
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
        # filename = "YLD_2d_Investigation/experiment_data/7p1_3.csv"
        length, x, y, eps_x, eps_y, phiM = read_csv_files(filename)
        calib_length = calibrate_length(length, phiM, group)
        # print("\n length: ", calib_length)
        # plot_diagrams(calib_length, eps_x, eps_y, phiM, label)

        y_sum_eps_x = fit_curve(y_sum_eps_x, calib_length, eps_x, "eps_x")
        y_sum_eps_y = fit_curve(y_sum_eps_y, calib_length, eps_y, "eps_y")
        y_sum_phiM = fit_curve(y_sum_phiM, calib_length, phiM, "phiM")

        length_max.append(calib_length.max())
        length_min.append(calib_length.min())

    avg_calib_length = np.linspace(np.mean(length_min), np.mean(length_max), 1000)
    avg_y_sum(y_sum_eps_x, avg_calib_length, "avg_eps_x")
    avg_y_sum(y_sum_eps_y, avg_calib_length, "avg_eps_y")
    avg_y_sum(y_sum_phiM, avg_calib_length, "avg_phiM")
    plt.title(f"Strain Distribution in {displacement}mm")
    plt.savefig(f"{path}/Strain_Distribution_{group}_{displacement}mm.png")

    plt.show()

    # break

    # foldername = "test/"

    # path = f"./{foldername}"

    # sub_folders = read_subfolders(path)
    # plot_3_strains(path, sub_folders)


if __name__ == "__main__":
    main()
