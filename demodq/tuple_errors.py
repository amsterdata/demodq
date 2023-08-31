import cleanlab
from numba import njit, prange
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def detect_mislabeled_via_shapley(data, dataset, seed):

    @njit(fastmath=True, parallel=True)
    def _compute_shapley_values(X_train, y_train, X_test, y_test, K):
        N = len(X_train)
        M = len(X_test)
        result = np.zeros(N, dtype=np.float32)

        for j in prange(M):
            score = np.zeros(N, dtype=np.float32)
            dist = np.zeros(N, dtype=np.float32)
            div_range = np.arange(1.0, N)
            div_min = np.minimum(div_range, K)
            for i in range(N):
                dist[i] = np.sqrt(np.sum(np.square(X_train[i] - X_test[j])))
            indices = np.argsort(dist)
            y_sorted = y_train[indices]
            eq_check = (y_sorted == y_test[j]) * 1.0
            diff = - 1 / K * (eq_check[1:] - eq_check[:-1])
            diff /= div_range
            diff *= div_min
            score[indices[:-1]] = diff
            score[indices[-1]] = eq_check[-1] / N
            score[indices] += np.sum(score[indices]) - np.cumsum(score[indices])
            result += score / M

        return result

    data['__seq_index'] = np.arange(len(data))

    numerical_encoder = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaled_numeric', StandardScaler())])
    categorical_encoder = Pipeline([('encode', OneHotEncoder(handle_unknown='ignore', sparse=False)),
                                    ('scale', StandardScaler())])

    encoder = ColumnTransformer(transformers=[
        ('categorical_features', categorical_encoder, dataset.categorical_columns),
        ('scaled_numeric', numerical_encoder, dataset.numerical_columns)], sparse_threshold=0.0)

    all_labels = dataset.extract_label_for_prediction_task(data)
    #print(f'label distribution: {np.sum(all_labels) / len(all_labels)} positive')

    # This is still a flaw, as we will ignore the 100 randomly chosen rows
    train_data, test_data = train_test_split(data, test_size=100, stratify=all_labels, random_state=seed)

    X_train = encoder.fit_transform(train_data)
    y_train = dataset.extract_label_for_prediction_task(train_data)
    X_test = encoder.transform(test_data)
    y_test = dataset.extract_label_for_prediction_task(test_data)

    shapley_values = _compute_shapley_values(X_train, np.squeeze(y_train), X_test, np.squeeze(y_test), K=10)

    potentially_mislabeled_indices = list(train_data[shapley_values < 0.0]['__seq_index'])

    return dataset.as_df().iloc[potentially_mislabeled_indices]


def detect_mislabeled_via_cleanlab(data, dataset, seed=None):

    numerical_encoder = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                                  ('scaled_numeric', StandardScaler())])

    encoder = ColumnTransformer(transformers=[
        ('categorical_features', OneHotEncoder(handle_unknown='ignore', sparse=False), dataset.categorical_columns),
        ('numerical_features', numerical_encoder, dataset.numerical_columns)])

    all_labels = dataset.extract_label_for_prediction_task(data)

    X = encoder.fit_transform(data)
    y = all_labels

    model = SGDClassifier(loss='log_loss').fit(X, y)
    cl = cleanlab.classification.CleanLearning(model, seed=seed)

    label_issues = cl.find_label_issues(X, y)
    label_issue_indexes = np.array(label_issues.is_label_issue)

    return data[label_issue_indexes]




def detect_outliers_via_if(data, dataset, seed):

    numerical_encoder = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaled_numeric', StandardScaler())])

    encoder = ColumnTransformer(transformers=[
        ('categorical_features', OneHotEncoder(handle_unknown='ignore'), dataset.categorical_columns),
        ('numerical_features',  numerical_encoder, dataset.numerical_columns)])

    X = encoder.fit_transform(data)

    identifiers = IsolationForest(random_state=seed, contamination=0.01).fit_predict(X)

    return data[identifiers == -1]

