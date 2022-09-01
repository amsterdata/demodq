from .column_errors import detect_missing_values, detect_outliers_sd, detect_outliers_iqr
from .tuple_errors import detect_mislabeled_via_cleanlab, detect_outliers_via_if, detect_mislabeled_via_shapley

configs_sex = [
       {
              'dataset': "adult",
              'criteria': 'sex',
              'detector': detect_missing_values,
              'detector_name': 'missing-values',
              'columns': ['workclass', 'occupation', 'native-country']
       },
       {
              'dataset': "adult",
              'criteria': 'sex',
              'detector': detect_outliers_sd,
              'detector_name': 'outliers-sd',
              'columns': ['hours-per-week', 'capital-gain', 'capital-loss']
       },
       {
              'dataset': "adult",
              'criteria': 'sex',
              'detector': detect_outliers_iqr,
              'detector_name': 'outliers-iqr',
              'columns': ['hours-per-week', 'capital-gain', 'capital-loss']
       },
       {
              'dataset': "adult",
              'criteria': 'sex',
              'detector': detect_outliers_via_if,
              'detector_name': 'outliers-if'
       },
       {
              'dataset': "adult",
              'criteria': 'sex',
              'detector': detect_mislabeled_via_cleanlab,
              'detector_name': 'mislabeled-cl'
       },
       {
              'dataset': "adult",
              'criteria': 'sex',
              'detector': detect_mislabeled_via_shapley,
              'detector_name': 'mislabeled-sh'
       },

       {
              'dataset': "folktables",
              'criteria': 'sex',
              'detector': detect_missing_values,
              'detector_name': 'missing-values',
              'columns': ['COW', 'SCHL', 'OCCP', 'WKHP']
       },
       {
              'dataset': "folktables",
              'criteria': 'sex',
              'detector': detect_outliers_sd,
              'detector_name': 'outliers-sd',
              'columns': ['WKHP']
       },
       {
              'dataset': "folktables",
              'criteria': 'sex',
              'detector': detect_outliers_via_if,
              'detector_name': 'outliers-if'
       },
       {
              'dataset': "folktables",
              'criteria': 'sex',
              'detector': detect_mislabeled_via_cleanlab,
              'detector_name': 'mislabeled-cl'
       },
       {
              'dataset': "folktables",
              'criteria': 'sex',
              'detector': detect_mislabeled_via_shapley,
              'detector_name': 'mislabeled-sh'
       },

       {
              'dataset': "heart",
              'criteria': 'sex',
              'detector': detect_outliers_sd,
              'detector_name': 'outliers-sd',
              'columns': ['height', 'weight', 'ap_lo', 'gluc']
       },
       {
              'dataset': "heart",
              'criteria': 'sex',
              'detector': detect_outliers_iqr,
              'detector_name': 'outliers-iqr',
              'columns': ['height', 'weight', 'ap_hi', 'ap_lo', 'gluc']
       },
       {
              'dataset': "heart",
              'criteria': 'sex',
              'detector': detect_outliers_via_if,
              'detector_name': 'outliers-if'
       },
       {
              'dataset': "heart",
              'criteria': 'sex',
              'detector': detect_mislabeled_via_cleanlab,
              'detector_name': 'mislabeled-cl'
       },
       {
              'dataset': "heart",
              'criteria': 'sex',
              'detector': detect_mislabeled_via_shapley,
              'detector_name': 'mislabeled-sh'
       },

]

configs_race = [
       {
              'dataset': "adult",
              'criteria': 'race',
              'detector': detect_missing_values,
              'detector_name': 'missing-values',
              'columns': ['workclass', 'occupation', 'native-country']
       },
       {
              'dataset': "adult",
              'criteria': 'race',
              'detector': detect_outliers_sd,
              'detector_name': 'outliers-sd',
              'columns': ['hours-per-week', 'capital-loss']
       },
       {
              'dataset': "adult",
              'criteria': 'race',
              'detector': detect_outliers_iqr,
              'detector_name': 'outliers-iqr',
              'columns': ['age', 'hours-per-week', 'capital-gain', 'capital-loss']
       },
       {
              'dataset': "adult",
              'criteria': 'race',
              'detector': detect_outliers_via_if,
              'detector_name': 'outliers-if'
       },
       {
              'dataset': "adult",
              'criteria': 'race',
              'detector': detect_mislabeled_via_cleanlab,
              'detector_name': 'mislabeled-cl'
       },
       {
              'dataset': "adult",
              'criteria': 'race',
              'detector': detect_mislabeled_via_shapley,
              'detector_name': 'mislabeled-sh'
       },

       {
              'dataset': "folktables",
              'criteria': 'race',
              'detector': detect_missing_values,
              'detector_name': 'missing-values',
              'columns': ['COW', 'SCHL', 'OCCP', 'WKHP']
       },
       {
              'dataset': "folktables",
              'criteria': 'race',
              'detector': detect_outliers_via_if,
              'detector_name': 'outliers-if'
       },
       {
              'dataset': "folktables",
              'criteria': 'race',
              'detector': detect_mislabeled_via_cleanlab,
              'detector_name': 'mislabeled-cl'
       },
       {
              'dataset': "folktables",
              'criteria': 'race',
              'detector': detect_mislabeled_via_shapley,
              'detector_name': 'mislabeled-sh'
       },
]

configs_age = [
       {
       'dataset': "credit",
       'criteria': 'age',
       'detector': detect_missing_values,
       'detector_name': 'missing-values',
       'columns': ['MonthlyIncome', 'NumberOfDependents']
       },
       {
       'dataset': "credit",
       'criteria': 'age',
       'detector': detect_outliers_iqr,
       'detector_name':'outliers-iqr',
       'columns': ['RevolvingUtilizationOfUnsecuredLines', 'NumberOfTime30-59DaysPastDueNotWorse',
              'DebtRatio', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
              'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse']
       },
       {
       'dataset': "credit",
       'criteria': 'age',
       'detector': detect_outliers_sd,
       'detector_name': 'outliers-sd',
       'columns': ['NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
                   'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
                   'NumberOfDependents']
       },

       {
              'dataset': "german",
              'criteria': 'age',
              'detector': detect_missing_values,
              'detector_name': 'missing-values',
              'columns': ['property']
       },
       {
              'dataset': "german",
              'criteria': 'age',
              'detector': detect_outliers_iqr,
              'detector_name': 'outliers-iqr',
              'columns': ['people_liable_for']
       },
       {
              'dataset': "german",
              'criteria': 'age',
              'detector': detect_mislabeled_via_cleanlab,
              'detector_name': 'mislabeled-cl'
       },
       {
              'dataset': "german",
              'criteria': 'age',
              'detector': detect_mislabeled_via_shapley,
              'detector_name': 'mislabeled-sh'
       },
]
