# Standard imports
import os
import time

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def time_function(n, func, **kwargs):
    """
    Function that returns a list of times for ten iterations of running a function.
    """
    times = []
    for i in range(n):
        print(f"Running iteration {i+1}/{n} of {func.__name__}")
        try:
            # Could change this to time.perf_counter() to get a more accurate time
            start = time.time()
            func(**kwargs)  # Run the function
            finish = time.time()
            runtime = finish - start
        except Exception as e:
            print(f"Error: {e}")
            runtime = np.nan
        times.append(runtime)
    return times


def time_method(n, class_instance, method, **kwargs):
    """
    Function that returns a list of times for ten iterations of running a method of a class that has already been instantiated
    """
    # TODO: find a way to print a more appropiate name for the class
    # print(f"\n---------- Timing {method} method of {class_instance.id} ----------\n")
    func = getattr(class_instance, method)
    # TODO: change this to catch errors and continue if it fails returing a list of Nones
    times = time_function(n, func, **kwargs)  # Run the function
    return times


def create_timestamp_directory():
    print("Creating timestamp directory")
    timestamp = time.strftime("%Y-%m-%dT%H:%M")
    recording_directory = os.path.join("tests", "performance", "data", timestamp)
    os.mkdir(recording_directory)
    return recording_directory


def record_results(list_dataframes, recording_directory, name="data.csv"):
    """A function that takes a list of dataframes and records them in a directory"""

    # Create a plot for each module
    for df in list_dataframes:
        plot_results(df, recording_directory)

    # Record the complete dataset
    complete_df = pd.concat(list_dataframes, axis="columns")
    complete_df.to_csv(os.path.join(recording_directory, name))


def plot_results(dataframe, recording_directory=None, date=False, log=False):
    """function that takes a dataframe and plots the results"""

    # Initialize the plot
    print(f"Plotting {dataframe.name}")
    plt.figure(figsize=(15, 10))

    # Plot the data
    for column in dataframe.columns:
        if column != "time_stamp":
            if date:  # TODO: Check if this way of passing in the variables works
                # times = time.strptime(dataframe["time_stamp"], "%Y-%m-%dT%H:%M")
                times = pd.to_datetime(dataframe["time_stamp"])
                # times = type(dataframe["time_stamp"].iloc[0])
                print(times)
                plt.plot(
                    times,
                    dataframe[column],
                    marker="o",
                    markeredgewidth=1,
                    linestyle=":",
                    label=column,
                )
                if log:
                    plt.gca().set_yscale("log")
                # plt.gca().set_xticklabels(  # Renaming the x-ticks to make it more readable
                #     [date.split("T")[0][5:] for date in dataframe["time_stamp"]]
                # )
            else:
                plt.plot(
                    range(1, len(dataframe) + 1),
                    dataframe[column],
                    marker="o",
                    markeredgewidth=1,
                    linestyle=":",
                    label=column,
                )

    # Finalise the plot
    if date:
        plt.xlabel("Dates")
        dates = dataframe["time_stamp"]
        plt.xticks(dates, rotation=45, ha="right")
    else:
        plt.xlabel("Iteration")
        plt.xticks(range(1, len(dataframe) + 1))
    plt.ylabel("Time [s]")
    plt.ylim(bottom=0)
    plt.title(f"Performance of {dataframe.name}")
    plt.legend(loc="upper left")
    if recording_directory is not None:
        if log:
            filename = f"{dataframe.name}_log_plot.png"
        else:
            filename = f"{dataframe.name}_plot.png"
        figure_path = os.path.join(recording_directory, filename)
        plt.savefig(figure_path)
    else:
        plt.show()


# TODO: Can we rename dataframe to df here?
def reformat_average_dataframe(dataframe):
    """function that reformats the dataframe into the 3 separate dataframes and gives each one of them a name"""

    # Munge the dataframes
    dataframe.drop(
        dataframe.columns[dataframe.columns.str.contains("unnamed", case=False)],
        axis="columns",
        inplace=True,
    )
    dataframe = dataframe.sort_values(by="time_stamp")

    # Set up the results dataframes
    core_data = pd.DataFrame()
    core_data.name = "core_functions"
    dataset_data = pd.DataFrame()
    dataset_data.name = "Dataset_methods"
    emulator_data = pd.DataFrame()
    emulator_data.name = "Emulator_methods"

    # Split the dataframe into the 3 dataframes
    for column in dataframe.columns:
        if "dataset_" in column:
            dataset_data[column] = dataframe[column]
        elif "emulator_" in column:
            emulator_data[column] = dataframe[column]
        else:
            core_data[column] = dataframe[column]

    # Add the time stamp to the emulator and dataset dataframes
    emulator_data["time_stamp"] = dataframe["time_stamp"]
    dataset_data["time_stamp"] = dataframe["time_stamp"]

    return [core_data, dataset_data, emulator_data]


class TimeUnit:
    def __init__(self, class_instance, method, label=None, **kwargs):
        self.class_instance = class_instance
        self.method = method
        self.kwargs = kwargs
        if label is None:
            self.label = f"{class_instance.id}_{method}"
        else:
            self.label = label


class TimeBox:
    def __init__(self, name="micol"):
        self.list_methods = []
        self.name = name

    def add_method(self, timethis: TimeUnit):
        self.list_methods.append(timethis)

    # Change the name to be an attribute of the class
    def give_times(self, n):
        df = pd.DataFrame()
        df.name = f"{self.name}"
        for method in self.list_methods:
            df[method.label] = time_method(
                n, method.class_instance, method.method, **method.kwargs
            )
        # recording_directory = Path(__file__).parent / "other_data" / self.name
        # os.mkdir(recording_directory)
        # record_results([df], recording_directory)
        plot_results(df)


def _clip_col(times, nsig=3):
    """
    Remove outliers using sigma clipping
    """
    mu = np.median(times)
    sigma = np.std(times)
    mask = np.abs(times - mu) < nsig * sigma
    return times[mask]


def sigma_clip(df, sigma=3):
    """
    Sigma clip the dataframe
    """
    for column in df.columns:
        df[column] = _clip_col(df[column], sigma)
    return df
