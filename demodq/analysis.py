from scipy.stats import chi2_contingency


def is_disparate(num_priv, num_dirty_priv, num_nonpriv, num_dirty_nonpriv):
    if (num_dirty_priv > 0 or num_dirty_nonpriv > 0) and \
            (num_priv != num_dirty_priv and num_nonpriv != num_dirty_nonpriv):
        observations = [[num_dirty_nonpriv, num_nonpriv - num_dirty_nonpriv],
                        [num_dirty_priv, num_priv - num_dirty_priv]]

        # G-test for significant cooccurrence of being disadvantaged and having errors
        g, p, dof, expctd = chi2_contingency(observations, lambda_="log-likelihood")

        return p < 0.05
    else:
        return False


def detect_disparate_errors(dataset, criteria, columns, error_type, detector):
    data = dataset.as_df()

    for column in columns:

        dirty = detector(data, column)
        analyse_marked(error_type, criteria, dataset, dirty, column)


def analyse_marked(error_type, criteria, dataset, marked, column='*'):

    data_priv, data_nonpriv = dataset.partition_data_by_single_axis(dataset.as_df(), criteria)
    data_dirty_priv, data_dirty_nonpriv = dataset.partition_data_by_single_axis(marked, criteria)

    num_priv = len(data_priv)
    num_dirty_priv = len(data_dirty_priv)
    num_nonpriv = len(data_nonpriv)
    num_dirty_nonpriv = len(data_dirty_nonpriv)

    if is_disparate(num_priv, num_dirty_priv, num_nonpriv, num_dirty_nonpriv):
        print(f'%{dataset.name}: {num_priv} {num_dirty_priv} {num_nonpriv} {num_dirty_nonpriv}')
        print(f' & {error_type} & {column} & {round(num_dirty_priv * 100 / num_priv, 1)}\% & {round(num_dirty_nonpriv * 100 / num_nonpriv, 1)}\%\\\\')


