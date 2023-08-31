import numpy as np
import pandas as pd

from ._base import Dataset


class CreditDataset(Dataset):

    def __init__(self):
        self.data = pd.read_csv('data/credit/givemesomecredit.csv')

    @property
    def name(self):
        return 'credit'

    @property
    def numerical_columns(self):
        return ['RevolvingUtilizationOfUnsecuredLines', 'age',
                'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
                'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
                'NumberOfDependents']

    @property
    def categorical_columns(self):
        return []

    @property
    def numerical_columns_for_prediction(self):
        return ['RevolvingUtilizationOfUnsecuredLines',
                'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
                'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
                'NumberOfDependents']

    @property
    def categorical_columns_for_prediction(self):
        return []

    def extract_label_for_prediction_task(self, data):
        return np.array(data['SeriousDlqin2yrs'] != 1)

    def as_df(self):
        return self.data

    @property
    def demographic_criteria(self):
        return ['age']

    def partition_data_by_single_axis(self, data, criteria):
        if criteria == 'age':
            data_priv = data[data.age > 30]
            data_nonpriv = data[data.age <= 30]
            return data_priv, data_nonpriv
        else:
            raise ValueError(f"Unsupported: {criteria}")

    def partition_data_by_intersection(self, data, criteria1, criteria2):
        raise ValueError(f"Unsupported: {criteria1}, {criteria2}")
