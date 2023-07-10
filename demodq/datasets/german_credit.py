import numpy as np
import numpy as np
import pandas as pd

from ._base import Dataset


class GermanCreditDataset(Dataset):

    def __init__(self):
        columns = ['status', 'month', 'credit_history', 'purpose', 'credit_amount', 'savings',
            'employment', 'investment_as_income_percentage', 'personal_status',
            'other_debtors', 'residence_since', 'property', 'age',
            'installment_plans', 'housing', 'number_of_credits',
            'skill_level', 'people_liable_for', 'telephone',
            'foreign_worker', 'credit']

        self.data = pd.read_csv('data/german/german.data', sep=' ', header=None,
                                names=columns, na_values=['A65', 'A124'])

    @property
    def name(self):
        return 'german'

    @property
    def numerical_columns(self):
        return ['month', 'credit_amount', 'investment_as_income_percentage', 'residence_since', 'age',
                'number_of_credits', 'people_liable_for']

    @property
    def categorical_columns(self):
        return ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property',
                'installment_plans', 'housing', 'skill_level', 'telephone', 'foreign_worker', 'personal_status']

    @property
    def numerical_columns_for_prediction(self):
        return ['month', 'credit_amount', 'investment_as_income_percentage', 'residence_since',
                'number_of_credits', 'people_liable_for']

    @property
    def categorical_columns_for_prediction(self):
        return ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property',
                'installment_plans', 'housing', 'skill_level', 'telephone', 'foreign_worker']

    def extract_label_for_prediction_task(self, data):
        return np.array(data['credit'] == 1.0)

    def as_df(self):
        return self.data

    @property
    def demographic_criteria(self):
        return ['age']

    def partition_data_by(self, data, criteria):
        if criteria == 'age':
            data_priv = data[data.age > 25]
            data_nonpriv = data[data.age <= 25]
            return data_priv, data_nonpriv
        elif criteria == 'sex':
            data_priv = data[data.personal_status.isin(['A91', 'A93', 'A94'])]
            data_nonpriv = data[~data.personal_status.notin(['A91', 'A93', 'A94'])]
            return data_priv, data_nonpriv
        elif criteria == 'foreign_worker':
            data_priv = data[data.foreign_worker == 'A202']
            data_nonpriv = data[data.foreign_worker != 'A202']
            return data_priv, data_nonpriv
        else:
            raise ValueError(f"Unsupported: {criteria}")
