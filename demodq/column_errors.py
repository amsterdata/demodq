import numpy as np


def detect_missing_values(data, column):
    # Flag NaN values
    return data[data[column].isna()]


def detect_outliers_sd(data, column):
    mean = np.mean(data[column])
    std = np.std(data[column])
    # Flag values as outliers if they are more than 3 stddevs away from the mean
    outliers = data[(data[column] > (mean + 3 * std)) | (data[column] < (mean - 3 * std))]
    return outliers


def detect_outliers_iqr(data, column):
    q1 = np.percentile(data[column], 25)
    q3 = np.percentile(data[column], 75)
    iqr = q3-q1
    # Flag values as outliers if they are more than 1.5 IQRs aways from Q1 or Q3
    outliers = data[(data[column] > (q3 + 1.5 * iqr)) | (data[column] < (q1 - 1.5 * iqr))]
    return outliers
