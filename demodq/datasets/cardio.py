import numpy as np
import pandas as pd

from ._base import Dataset


class CardioDataset(Dataset):

    def __init__(self):
        data = pd.read_csv('data/cardio/cardio.csv', sep=';')
        self.data = data

    @property
    def name(self):
        return 'heart'

    @property
    def numerical_columns(self):
        return ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']

    @property
    def categorical_columns(self):
        return ['gender', 'smoke', 'alco', 'active']

    @property
    def numerical_columns_for_prediction(self):
        return ['height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']

    @property
    def categorical_columns_for_prediction(self):
        return ['smoke', 'alco', 'active']

    def extract_label_for_prediction_task(self, data):
        return np.array(data.cardio == 1)

    def as_df(self):
        return self.data

    @property
    def demographic_criteria(self):
        return ['sex']

    def partition_data_by_single_axis(self, data, criteria):
        if criteria == 'sex':
            data_priv = data[data.gender == 1]
            data_nonpriv = data[data.gender != 1]
            return data_priv, data_nonpriv
        elif criteria == 'age@35':
            data_priv = data[data.age > 35*365.25]
            data_nonpriv = data[data.age <= 35*365.25]
            return data_priv, data_nonpriv
        elif criteria == 'age@45':
            data_priv = data[data.age > 45*365.25]
            data_nonpriv = data[data.age <= 45*365.25]
            return data_priv, data_nonpriv
        elif criteria == 'age@55':
            data_priv = data[data.age > 55*365.25]
            data_nonpriv = data[data.age <= 55*365.25]
            return data_priv, data_nonpriv
        else:
            raise ValueError(f"Unsupported: {criteria}")

    def partition_data_by_intersection(self, data, criteria1, criteria2):
        if criteria1 == 'sex' and criteria2 == 'age@35':
            data_priv_priv = data[(data.gender == 1) & (data.age > 35*365.25)]
            data_priv_nonpriv = data[(data.gender == 1) & (data.age <= 35*365.25)]
            data_nonpriv_priv = data[(data.gender != 1) & (data.age > 35*365.25)]
            data_nonpriv_nonpriv = data[(data.gender != 1) & (data.age <= 35*365.25)]
            return data_priv_priv, data_priv_nonpriv, data_nonpriv_priv, data_nonpriv_nonpriv
        elif criteria1 == 'age@35' and criteria2 == 'sex':
            data_priv_priv = data[(data.age > 35*365.25) & (data.gender == 1)]
            data_priv_nonpriv = data[(data.age > 35*365.25) & (data.gender != 1)]
            data_nonpriv_priv = data[(data.age <= 35*365.25) & (data.gender == 1)]
            data_nonpriv_nonpriv = data[(data.age <= 35*365.25) & (data.gender != 1)]
            return data_priv_priv, data_priv_nonpriv, data_nonpriv_priv, data_nonpriv_nonpriv
        elif criteria1 == 'sex' and criteria2 == 'age@45':
            data_priv_priv = data[(data.gender == 1) & (data.age > 45*365.25)]
            data_priv_nonpriv = data[(data.gender == 1) & (data.age <= 45*365.25)]
            data_nonpriv_priv = data[(data.gender != 1) & (data.age > 45*365.25)]
            data_nonpriv_nonpriv = data[(data.gender != 1) & (data.age <= 45*365.25)]
            return data_priv_priv, data_priv_nonpriv, data_nonpriv_priv, data_nonpriv_nonpriv
        elif criteria1 == 'age@45' and criteria2 == 'sex':
            data_priv_priv = data[(data.age > 45*365.25) & (data.gender == 1)]
            data_priv_nonpriv = data[(data.age > 45*365.25) & (data.gender != 1)]
            data_nonpriv_priv = data[(data.age <= 45*365.25) & (data.gender == 1)]
            data_nonpriv_nonpriv = data[(data.age <= 45*365.25) & (data.gender != 1)]
            return data_priv_priv, data_priv_nonpriv, data_nonpriv_priv, data_nonpriv_nonpriv
        elif criteria1 == 'sex' and criteria2 == 'age@55':
            data_priv_priv = data[(data.age > 55*365.25) & (data.gender == 1)]
            data_priv_nonpriv = data[(data.age > 55*365.25) & (data.gender != 1)]
            data_nonpriv_priv = data[(data.age <= 55*365.25) & (data.gender == 1)]
            data_nonpriv_nonpriv = data[(data.age <= 55*365.25) & (data.gender != 1)]
            return data_priv_priv, data_priv_nonpriv, data_nonpriv_priv, data_nonpriv_nonpriv
        elif criteria1 == 'age@55' and criteria2 == 'sex':
            data_priv_priv = data[(data.age > 55*365.25) & (data.gender == 1)]
            data_priv_nonpriv = data[(data.age > 55*365.25) & (data.gender != 1)]
            data_nonpriv_priv = data[(data.age <= 55*365.25) & (data.gender == 1)]
            data_nonpriv_nonpriv = data[(data.age <= 55*365.25) & (data.gender != 1)]
            return data_priv_priv, data_priv_nonpriv, data_nonpriv_priv, data_nonpriv_nonpriv
        else:
            raise ValueError(f"Unsupported: {criteria1} {criteria2}")
