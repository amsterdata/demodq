from .folktables import FolktablesDataset
from .credit import CreditDataset
from .adult import AdultDataset
from .cardio import CardioDataset
from .german_credit import GermanCreditDataset


class Datasets:

    datasets_by_name = {
        'adult': AdultDataset,
        'folktables': FolktablesDataset,
        'heart': CardioDataset,
        'credit': CreditDataset,
        'german': GermanCreditDataset
    }

    @classmethod
    def load(cls, dataset_name):
        dataset_class = Datasets.datasets_by_name[dataset_name]
        return dataset_class()
