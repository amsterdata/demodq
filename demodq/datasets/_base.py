from abc import ABC, abstractmethod


class Dataset(ABC):

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def numerical_columns(self):
        pass

    @property
    @abstractmethod
    def categorical_columns(self):
        pass

    @property
    @abstractmethod
    def numerical_columns_for_prediction(self):
        pass

    @property
    @abstractmethod
    def categorical_columns_for_prediction(self):
        pass

    @abstractmethod
    def extract_label_for_prediction_task(self, data):
        pass

    @abstractmethod
    def as_df(self):
        pass

    @property
    @abstractmethod
    def demographic_criteria(self):
        pass

    def partition_by(self, criteria):
        return self.partition_data_by_single_axis(self.as_df(), criteria)

    @abstractmethod
    def partition_data_by_single_axis(self, data, criteria):
        pass

    @abstractmethod
    def partition_data_by_intersection(self, data, criteria1, criteria2):
        pass
