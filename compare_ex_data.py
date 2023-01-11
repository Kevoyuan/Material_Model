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
        eps_x = data[column_headers[1]]
        eps_y = data[column_headers[2]]
        phiM = data[column_headers[3]]

        # Return the data columns as lists
        return length, eps_x, eps_y, phiM
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None, None, None, None, None, None
    except Exception as e:
        print(f"Error: An error occurred while reading the file: {e}")
        return None, None, None, None, None, None


# Function to plot the data and add it to the current plot
def plot_diagrams(calib_length, eps_x, eps_y, phiM, curvetype, label):
    """
    Plots various diagrams using the calibrated length, strain in the x direction, strain in the y direction,
    and bending moment.

    Parameters:
        calib_length (list): The calibrated length of the beam.
        eps_x (list): The strain in the x direction.
        eps_y (list): The strain in the y direction.
        phiM (list): The bending moment.
        curvetype (str): The curve type to plot. Can be "eps_x", "eps_y", or "phiM".
        label (str): The label to use for the plots.
    """

    # Select the data to plot based on the curve type
    curve_plot = {"eps_x": eps_x, "eps_y": eps_y, "phiM": phiM}
    curve_type = curve_plot[curvetype]

    # Plot the calibrated length vs. the selected curve type
    plt.scatter(calib_length, curve_type, s=1, label=f"{label}_{curvetype}")

    # Set the x-axis ticks and y-axis limits
    plt.xticks(np.linspace(-8, 8, 9))
    plt.ylim(-0.1, 0.2)

    # Add a legend and grid to the plot
    plt.legend()
    plt.grid(True, color="grey", linewidth="1.4", linestyle="-.")


def main():

    # Set the path to the directory containing the CSV files
    path = "./YLD_2d_Investigation/experiment_data/test2/experiments"

    # Set the displacement and curve type for the plots

    displacement = "3p5"  # for different displacement: 2, 2p5, 3, 3p5
    curvetype = "eps_y"  # "eps_x", "eps_y", "phiM"

    # Find all the CSV files in the specified directory that match the specified displacement
    csv_files = glob.glob(f"{path}/avg_data_*_{displacement}mm.csv")

    # Loop through each file and plot the data
    for filename in csv_files:

        # Replace any backslashes in the filename with forward slashes
        if "\\" in filename:
            filename = filename.replace("\\", "/")
        print(filename)

        # Extract the label for the plot from the filename
        label = filename.split("/")[5].split(".")[0]
        label = label.replace("p", ".").replace("_data", "")

        # Read the data from the CSV file and store it in separate lists
        length, eps_x, eps_y, phiM = read_csv_files(filename)

        # Plot the data and add it to the current plot
        plot_diagrams(length, eps_x, eps_y, phiM, curvetype, label)
        
    plt.xlabel("section along middle line in mm")
    plt.ylabel("eq. strain")

    displacement_title = filename.split("_")[-1].replace("p", ".").replace("mm.csv", "")

    plt.title(f"avg_Strain_Distribution_{curvetype}_{displacement_title}mm")
    # print(displacement)
    # Save the plot to a file and display it on the screen
    plt.savefig(f"{path}/avg_Strain_Distribution_{curvetype}_{displacement}mm.png")

    # plt.show()


# Run the main function if this script is being run directly
if __name__ == "__main__":
    main()
