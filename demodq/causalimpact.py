import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix


def train_and_evaluate(dataset, criteria, train, test):
    train_labels = dataset.extract_label_for_prediction_task(train)
    test_labels = dataset.extract_label_for_prediction_task(test)

    numerical_encoder = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaled_numeric', StandardScaler())])

    encoder = ColumnTransformer(transformers=[
        ('categorical_features', OneHotEncoder(handle_unknown='ignore', sparse=False),
         dataset.categorical_columns_for_prediction),
        ('numerical_features', numerical_encoder, dataset.numerical_columns_for_prediction)])

    pipeline = Pipeline([
        ('features', encoder),
        ('learner', SGDClassifier(loss='log_loss'))
    ])

    param_grid = {
        'learner__alpha': [0.0001, 0.001, 0.01, 0.1]
    }

    search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
    model = search.fit(train, train_labels)

    def bool_to_float(bool_arr):
        arr = []
        for bool_val in bool_arr:
            if bool_val:
                arr.append(1.0)
            else:
                arr.append(0.0)
        return np.array(arr)

    y_true = bool_to_float(test_labels)
    y_pred = model.predict_proba(test)[:, 1]

    accuracy = model.score(test, test_labels)
    auc = roc_auc_score(y_true, y_pred)

    priv, nonpriv = dataset.partition_data_by(test, criteria)

    priv_labels = dataset.extract_label_for_prediction_task(priv)
    nonpriv_labels = dataset.extract_label_for_prediction_task(nonpriv)

    priv_pred = model.predict(priv)
    nonpriv_pred = model.predict(nonpriv)

    tn_priv, fp_priv, fn_priv, tp_priv = confusion_matrix(priv_labels, priv_pred).ravel()
    tn_nonpriv, fp_nonpriv, fn_nonpriv, tp_nonpriv = confusion_matrix(nonpriv_labels, nonpriv_pred).ravel()

    return accuracy, auc, tn_priv, fp_priv, fn_priv, tp_priv, tn_nonpriv, fp_nonpriv, fn_nonpriv, tp_nonpriv


def statistical_parity(given_res):
    res = given_res.copy(deep=True)

    orig_p_priv = (res.orig_tp_priv + res.orig_fp_priv) / (
                res.orig_tp_priv + res.orig_fp_priv + res.orig_tn_priv + res.orig_fn_priv)
    orig_p_nonpriv = (res.orig_tp_nonpriv + res.orig_fp_nonpriv) / (
                res.orig_tp_nonpriv + res.orig_fp_nonpriv + res.orig_tn_nonpriv + res.orig_fn_nonpriv)

    orig_p = orig_p_priv - orig_p_nonpriv
    res['orig_p'] = orig_p

    clean_p_priv = (res.clean_tp_priv + res.clean_fp_priv) / (
                res.clean_tp_priv + res.clean_fp_priv + res.clean_tn_priv + res.clean_fn_priv)
    clean_p_nonpriv = (res.clean_tp_nonpriv + res.clean_fp_nonpriv) / (
                res.clean_tp_nonpriv + res.clean_fp_nonpriv + res.clean_tn_nonpriv + res.clean_fn_nonpriv)

    clean_p = clean_p_priv - clean_p_nonpriv
    res['clean_p'] = clean_p

    # Only look at unfair cases
    num_applicable = len(res[res.orig_p > 0])
    unfair_res = res[(res.orig_p > res.clean_p) & (res.orig_p > 0)]
    # Causal responsibility assumes that min value of fairness score is 0
    unfair_res.loc[:, 'clean_p'] = unfair_res['clean_p'].map(lambda x: max(x, 0.0))

    cresp = (unfair_res.orig_p - unfair_res.clean_p) / unfair_res.orig_p
    acc_diff = unfair_res.clean_accuracy - unfair_res.orig_accuracy
    auc_diff = unfair_res.clean_auc - unfair_res.orig_auc

    return len(res), len(unfair_res), num_applicable, orig_p, clean_p, cresp, acc_diff, auc_diff


def predictive_parity(given_res):
    res = given_res.copy(deep=True)

    orig_ppv_priv = (res.orig_tp_priv / (res.orig_tp_priv + res.orig_fp_priv))
    orig_ppv_nonpriv = (res.orig_tp_nonpriv / (res.orig_tp_nonpriv + res.orig_fp_nonpriv))

    orig_pp = orig_ppv_priv - orig_ppv_nonpriv
    res['orig_pp'] = orig_pp

    clean_ppv_priv = (res.clean_tp_priv / (res.clean_tp_priv + res.clean_fp_priv))
    clean_ppv_nonpriv = (res.clean_tp_nonpriv / (res.clean_tp_nonpriv + res.clean_fp_nonpriv))

    clean_pp = clean_ppv_priv - clean_ppv_nonpriv
    res['clean_pp'] = clean_pp

    # Only look at unfair cases
    num_applicable = len(res[res.orig_pp > 0])
    unfair_res = res[(res.orig_pp > res.clean_pp) & (res.orig_pp > 0)]
    # Causal responsibility assumes that min value of fairness score is 0
    unfair_res.loc[:, 'clean_pp'] = unfair_res['clean_pp'].map(lambda x: max(x, 0.0))
    cresp = (unfair_res.orig_pp - unfair_res.clean_pp) / unfair_res.orig_pp
    acc_diff = unfair_res.clean_accuracy - unfair_res.orig_accuracy
    auc_diff = unfair_res.clean_auc - unfair_res.orig_auc

    return len(res), len(unfair_res), num_applicable, orig_pp, clean_pp, cresp, acc_diff, auc_diff


def equal_opportunity(given_res):
    res = given_res.copy(deep=True)

    orig_eo = (res.orig_tp_priv / (res.orig_tp_priv + res.orig_fn_priv)) - \
              (res.orig_tp_nonpriv / (res.orig_tp_nonpriv + res.orig_fn_nonpriv))
    res['orig_eo'] = orig_eo

    clean_eo = (res.clean_tp_priv / (res.clean_tp_priv + res.clean_fn_priv)) - \
               (res.clean_tp_nonpriv / (res.clean_tp_nonpriv + res.clean_fn_nonpriv))
    res['clean_eo'] = clean_eo

    # Only look at unfair cases
    num_applicable = len(res[res.orig_eo > 0])
    unfair_res = res[(res.orig_eo > res.clean_eo) & (res.orig_eo > 0)]
    # Causal responsibility assumes that min value of fairness score is 0
    unfair_res.loc[:, 'clean_eo'] = unfair_res['clean_eo'].map(lambda x: max(x, 0.0))
    cresp = (unfair_res.orig_eo - unfair_res.clean_eo) / unfair_res.orig_eo
    acc_diff = unfair_res.clean_accuracy - unfair_res.orig_accuracy
    auc_diff = unfair_res.clean_auc - unfair_res.orig_auc

    return len(res), len(unfair_res), num_applicable, orig_eo, clean_eo, cresp, acc_diff, auc_diff
