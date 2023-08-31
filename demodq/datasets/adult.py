import numpy as np
import pandas as pd

from ._base import Dataset


class AdultDataset(Dataset):

    def __init__(self):
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                   'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                   'hours-per-week', 'native-country', 'income-level']

        data_train = pd.read_csv('data/adult-income/adult.data', na_values=['?'], header=None, names=columns, sep=", ")
        data_test = pd.read_csv('data/adult-income/adult.test', na_values=['?'], header=None, names=columns, sep=", ")

        data = pd.concat([data_train, data_test])

        # Remove nonsensical record
        data = data[data.age != '|1x3 Cross validator']
        data.age = pd.to_numeric(data.age)
        self.data = data

    @property
    def name(self):
        return 'adult'

    @property
    def numerical_columns(self):
        return ['age', 'hours-per-week', 'capital-gain', 'capital-loss']

    @property
    def categorical_columns(self):
        return ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                'race', 'sex', 'native-country']

    @property
    def numerical_columns_for_prediction(self):
        return ['hours-per-week', 'capital-gain', 'capital-loss']

    @property
    def categorical_columns_for_prediction(self):
        return ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']

    def extract_label_for_prediction_task(self, data):
        return np.array((data['income-level'] == '>50K') | (data['income-level'] == '>50K.'))

    def as_df(self):
        return self.data

    @property
    def demographic_criteria(self):
        return ['age', 'race', 'sex']

    def partition_data_by_single_axis(self, data, criteria):
        if criteria == 'race':
            data_priv = data[data.race == 'White']
            data_nonpriv = data[data.race != 'White']
            return data_priv, data_nonpriv
        elif criteria == 'sex':
            data_priv = data[data.sex == 'Male']
            data_nonpriv = data[data.sex != 'Male']
            return data_priv, data_nonpriv
        elif criteria == 'age':
            data_priv = data[data.age > 30]
            data_nonpriv = data[data.age <= 30]
            return data_priv, data_nonpriv
        elif criteria == 'country':
            data_priv = data[data['native-country'] == 'United-States']
            data_nonpriv = data[data['native-country'] != 'United-States']
            return data_priv, data_nonpriv
        else:
            raise ValueError(f"Unsupported: {criteria}")

    def partition_data_by_intersection(self, data, criteria1, criteria2):
        if criteria1 == 'race' and criteria2 == 'sex':
            data_priv_priv = data[(data.race == 'White') & (data.sex == 'Male')]
            data_priv_nonpriv = data[(data.race == 'White') & (data.sex != 'Male')]
            data_nonpriv_priv = data[(data.race != 'White') & (data.sex == 'Male')]
            data_nonpriv_nonpriv = data[(data.race != 'White') & (data.sex != 'Male')]
            return data_priv_priv, data_priv_nonpriv, data_nonpriv_priv, data_nonpriv_nonpriv
        elif criteria1 == 'sex' and criteria2 == 'race':
            data_priv_priv = data[(data.sex == 'Male') & (data.race == 'White')]
            data_priv_nonpriv = data[(data.sex == 'Male') & (data.race != 'White')]
            data_nonpriv_priv = data[(data.sex != 'Male') & (data.race == 'White')]
            data_nonpriv_nonpriv = data[(data.sex != 'Male') & (data.race != 'White')]
            return data_priv_priv, data_priv_nonpriv, data_nonpriv_priv, data_nonpriv_nonpriv
        else:
            raise ValueError(f"Unsupported: {criteria1} {criteria2}")
