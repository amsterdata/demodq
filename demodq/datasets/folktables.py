import numpy as np

from folktables import ACSDataSource

from ._base import Dataset


class FolktablesDataset(Dataset):

    def __init__(self):
        data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
        acs_data = data_source.get_data(states=["CA"], download=True)

        columns = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P', 'PINCP']
        self.data = acs_data[columns]

    @property
    def name(self):
        return 'folktables'

    @property
    def numerical_columns(self):
        return ['AGEP', 'WKHP']

    @property
    def categorical_columns(self):
        return ['COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']

    @property
    def numerical_columns_for_prediction(self):
        return ['WKHP']

    @property
    def categorical_columns_for_prediction(self):
        return ['COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP']

    def extract_label_for_prediction_task(self, data):
        return np.array(data['PINCP'] >= 50000)

    def as_df(self):
        return self.data

    @property
    def demographic_criteria(self):
        return ['race', 'sex', 'age']

    def partition_by(self, criteria):
        return super().partition_by(criteria)

    def partition_data_by_single_axis(self, data, criteria):
        if criteria == 'race':
            data_priv = data[data.RAC1P == 1]
            data_nonpriv = data[data.RAC1P != 1]
            return data_priv, data_nonpriv
        elif criteria == 'sex':
            data_priv = data[data.SEX == 1]
            data_nonpriv = data[data.SEX != 1]
            return data_priv, data_nonpriv
        elif criteria == 'age':
            data_priv = data[data.AGEP > 30]
            data_nonpriv = data[data.AGEP <= 30]
            return data_priv, data_nonpriv
        else:
            raise ValueError(f"Unsupported: {criteria}")

    def partition_data_by_intersection(self, data, criteria1, criteria2):
        if criteria1 == 'race' and criteria2 == 'sex':
            data_priv_priv = data[(data.RAC1P == 1) & (data.SEX == 1)]
            data_priv_nonpriv = data[(data.RAC1P == 1) & (data.SEX != 1)]
            data_nonpriv_priv = data[(data.RAC1P != 1) & (data.SEX == 1)]
            data_nonpriv_nonpriv = data[(data.RAC1P != 1) & (data.SEX != 1)]
            return data_priv_priv, data_priv_nonpriv, data_nonpriv_priv, data_nonpriv_nonpriv
        elif criteria1 == 'sex' and criteria2 == 'race':
            data_priv_priv = data[(data.SEX == 1) & (data.RAC1P == 1)]
            data_priv_nonpriv = data[(data.SEX == 1) & (data.RAC1P != 1)]
            data_nonpriv_priv = data[(data.SEX != 1) & (data.RAC1P == 1)]
            data_nonpriv_nonpriv = data[(data.SEX != 1) & (data.RAC1P != 1)]
            return data_priv_priv, data_priv_nonpriv, data_nonpriv_priv, data_nonpriv_nonpriv
        else:
            raise ValueError(f"Unsupported: {criteria1} {criteria2}")
