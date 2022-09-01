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

    def partition_data_by(self, data, criteria):
        if criteria == 'sex':
            data_priv = data[data.gender == 1]
            data_nonpriv = data[data.gender != 1]
            return data_priv, data_nonpriv
        else:
            raise ValueError(f"Unsupported: {criteria}")
