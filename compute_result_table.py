from argparse import ArgumentParser
import json
import os

import numpy as np
from scipy.stats import ttest_rel


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--results-folder", required=True)
    parser.add_argument("--cardio-age-threshold",
                        choices=["age@45", "age@55"],
                        default="age@45")
    parser.add_argument("--intersectional-formulation",
                        choices=["pp_dd", "pagg_dd", "pp_dagg"])

    return parser.parse_args()


def load_data(folder, cardio_age_threshold):
    datasets = {}

    with open(f"{folder}/Credit_result.json") as f:
        datasets["credit"] = json.load(f)

    with open(f"{folder}/USCensus_result.json") as f:
        datasets["adult"] = json.load(f)

    with open(f"{folder}/ACSIncome_result.json") as f:
        datasets["folktables"] = json.load(f)

    with open(f"{folder}/Cardio_{cardio_age_threshold}_result.json") as f:
        datasets["heart"] = json.load(f)

    with open(f"{folder}/GermanCredit_result.json") as f:
        datasets["german"] = json.load(f)

    return datasets


def t_test(dirty, clean):
    """Comparing method"""
    def two_tailed_t_test(dirty, clean):
        n_d = len(dirty)
        n_c = len(clean)
        n = min(n_d, n_c)
        t, p = ttest_rel(clean[:n], dirty[:n])
        if np.isnan(t):
            t, p = 0, 1
        return {"t-stats":t, "p-value":p}

    def one_tailed_t_test(dirty, clean, direction):
        two_tail = two_tailed_t_test(dirty, clean)
        t, p_two = two_tail["t-stats"], two_tail["p-value"]
        if direction == "positive":
            if t > 0 :
                p = p_two * 0.5
            else:
                p = 1 - p_two * 0.5
        else:
            if t < 0:
                p = p_two * 0.5
            else:
                p = 1 - p_two * 0.5
        return {"t-stats":t, "p-value":p}

    result = {}
    result["two_tail"] = two_tailed_t_test(dirty, clean)
    result["one_tail_pos"] = one_tailed_t_test(dirty, clean, "positive")
    result["one_tail_neg"] = one_tailed_t_test(dirty, clean, "negative")
    return result


class EmptyGroupException(Exception):
    """
    Custom exception for when an intersectional group is empty,
    and therefore there are no stats in the stats JSON object
    with the corresponding dictionary key.
    """
    
    def __init__(self, repair, criteria, group):
        self.repair = repair
        self.criteria = "/".join(criteria)
        self.group = group
        super(EmptyGroupException, self).__init__(
            f"For cleaning method {repair} and criteria {criteria},"
            f" the group {group} is empty.")


def stat_value(stat_name, stats, cleaning_method, criteria, *groups):
    def _simple(stat_name, stats, cleaning_method, criteria, group):
        return stats[f"{cleaning_method}__{criteria}_{group}__{stat_name}"]

    def _intersectional(stat_name, stats, cleaning_method, criteria, *groups):
        def _prefix(cleaning_method, criteria, intersectional_group):
            first, second = intersectional_group.split("_")
            return f"{cleaning_method}__{criteria[0]}_{first}__{criteria[1]}_{second}"

        count = 0
        for g in groups:
            stat_key = f"{_prefix(cleaning_method, criteria, g)}__{stat_name}"
            try:
                count += stats[stat_key]
            except KeyError:
                raise EmptyGroupException(cleaning_method, criteria, g)
        return count

    if isinstance(criteria, str):
        return _simple(stat_name, stats, cleaning_method, criteria, *groups)
    else:
        return _intersectional(stat_name, stats, cleaning_method, criteria, *groups)


def define_priv_dis(intersectional_formulation):
    priv, dis = ["priv"], ["dis"]
    if intersectional_formulation == "pp_dd":
        priv = ["priv_priv"]
        dis = ["dis_dis"]
    elif intersectional_formulation == "pagg_dd":
        priv = ["priv_priv", "priv_dis", "dis_priv"]
        dis = ["dis_dis"]
    elif intersectional_formulation == "pp_dagg":
        priv = ["priv_priv"]
        dis = ["priv_dis", "dis_priv", "dis_dis"]
    return priv, dis


def compute_eo(stats, cleaning_method, criteria, flipped, priv, dis):
    if not flipped:
        priv_tp = stat_value("tp", stats, cleaning_method, criteria, *priv)
        priv_fn = stat_value("fn", stats, cleaning_method, criteria, *priv)
        dis_tp = stat_value("tp", stats, cleaning_method, criteria, *dis)
        dis_fn = stat_value("fn", stats, cleaning_method, criteria, *dis)

        return (priv_tp / (priv_tp + priv_fn)) - \
             (dis_tp / (dis_tp + dis_fn))
    else:
        priv_tn = stat_value("tn", stats, cleaning_method, criteria, *priv)
        priv_fp = stat_value("fp", stats, cleaning_method, criteria, *priv)
        dis_tn = stat_value("tn", stats, cleaning_method, criteria, *dis)
        dis_fp = stat_value("fp", stats, cleaning_method, criteria, *dis)

        return (priv_tn / (priv_tn + priv_fp)) - \
             (dis_tn / (dis_tn + dis_fp))


def compute_pp(stats, cleaning_method, criteria, flipped, priv, dis):
    if not flipped:
        priv_tp = stat_value("tp", stats, cleaning_method, criteria, *priv)
        priv_fp = stat_value("fp", stats, cleaning_method, criteria, *priv)
        dis_tp = stat_value("tp", stats, cleaning_method, criteria, *dis)
        dis_fp = stat_value("fp", stats, cleaning_method, criteria, *dis)

        return (priv_tp / (priv_tp + priv_fp)) - \
             (dis_tp / (dis_tp + dis_fp))
    else:
        priv_tn = stat_value("tn", stats, cleaning_method, criteria, *priv)
        priv_fn = stat_value("fn", stats, cleaning_method, criteria, *priv)
        dis_tn = stat_value("tn", stats, cleaning_method, criteria, *dis)
        dis_fn = stat_value("fn", stats, cleaning_method, criteria, *dis)

        return (priv_tn / (priv_tn + priv_fn)) - \
             (dis_tn / (dis_tn + dis_fn))


def compute_dp(stats, cleaning_method, criteria, flipped, priv, dis):
    if not flipped:
        priv_tp = stat_value("tp", stats, cleaning_method, criteria, *priv)
        priv_fp = stat_value("fp", stats, cleaning_method, criteria, *priv)
        priv_fn = stat_value("fn", stats, cleaning_method, criteria, *priv)
        priv_tn = stat_value("tn", stats, cleaning_method, criteria, *priv)
        dis_tp = stat_value("tp", stats, cleaning_method, criteria, *dis)
        dis_fp = stat_value("fp", stats, cleaning_method, criteria, *dis)
        dis_fn = stat_value("fn", stats, cleaning_method, criteria, *dis)
        dis_tn = stat_value("tn", stats, cleaning_method, criteria, *dis)

        return ((priv_tp + priv_fp) / (priv_tp + priv_fp + priv_fn + priv_tn)) - \
             ((dis_tp + dis_fp) / (dis_tp + dis_fp + dis_fn + dis_tn))
    else:
        priv_tn = stat_value("tn", stats, cleaning_method, criteria, *priv)
        priv_fn = stat_value("fn", stats, cleaning_method, criteria, *priv)
        priv_fp = stat_value("fp", stats, cleaning_method, criteria, *priv)
        priv_tp = stat_value("tp", stats, cleaning_method, criteria, *priv)
        dis_tn = stat_value("tn", stats, cleaning_method, criteria, *dis)
        dis_fn = stat_value("fn", stats, cleaning_method, criteria, *dis)
        dis_fp = stat_value("fp", stats, cleaning_method, criteria, *dis)
        dis_tp = stat_value("tp", stats, cleaning_method, criteria, *dis)

        return ((priv_tn + priv_fn) / (priv_tn + priv_fn + priv_fp + priv_tp)) - \
             ((dis_tn + dis_fn) / (dis_tn + dis_fn + dis_fp + dis_tp))


def evaluate_scores(dirty_scores, cleaning_scores, dirty_accs, cleaning_accs,
                    dataset_name, target_criteria, metric_name, model, error_type, log_file):
    if len(cleaning_scores) > 0:
        # bonferroni correction 
        alpha = 0.05 / len(cleaning_scores)

        for method, scores in cleaning_scores.items():
            test_results = t_test(dirty_scores, scores)

            repair_train, repair_clean = method
            test_repaired = repair_train == repair_clean

            difference = "insignificant"
            fairness_p = ""

            if test_results["two_tail"]["p-value"] < alpha:
                if test_results["one_tail_neg"]["p-value"] < alpha:
                    difference = "positive"
                    fairness_p = round(test_results["one_tail_neg"]["p-value"], 3)
                if test_results["one_tail_pos"]["p-value"] < alpha:
                    difference = "negative"
                    fairness_p = round(test_results["one_tail_pos"]["p-value"], 3)

            acc_test_results = t_test(dirty_accs, cleaning_accs[method])

            acc_difference = "insignificant"
            acc_p = ""

            if acc_test_results["two_tail"]["p-value"] < alpha:
                if acc_test_results["one_tail_neg"]["p-value"] < alpha:
                    acc_difference = "negative"
                    acc_p = round(acc_test_results["one_tail_neg"]["p-value"], 3)
                if acc_test_results["one_tail_pos"]["p-value"] < alpha:
                    acc_difference = "positive"
                    acc_p = round(acc_test_results["one_tail_pos"]["p-value"], 3)

            if error_type == "missing_values":
                repair_method = repair_train
                detection = ""
            elif error_type == "mislabel":
                detection, repair_method = repair_train.split("-")
            else:
                tokens = repair_train.split("_impute")
                detection = tokens[0].replace("clean_", "")
                repair_method = "impute" + tokens[1]

            if error_type == "mislabel" and (test_repaired or detection == "shapley"):
                continue

            criteria = target_criteria if isinstance(target_criteria, str) else "/".join(target_criteria)
            line = f"{dataset_name},{criteria},{metric_name},{model},{error_type},{detection},{repair_method},{test_repaired},{difference},{acc_difference}"

            if test_repaired or error_type == "mislabel":
                print(line)
                print(line, file=log_file)


def count(data, dataset_name, target_criteria, error_type, model, metric_name, scoring, log_file, flipped=False):
    dirty_scores = []
    dirty_accs = []

    cleaning_scores = {}
    cleaning_accs = {}

    dirty = "dirty"
    if error_type == "missing_values":
        dirty = "delete"

    for experiment, results in data.items():
        if error_type in experiment and model in experiment:
            train_method = experiment.split("/")[3]

            if train_method == dirty:
                # Missing values need special treatment, just deleting the corresponding rows from the test set
                # is not applicable in real-world scenarios, so we set a default way to treat the test data
                cleaning_method = "impute_mean_dummy" if error_type == "missing_values" else dirty

                score = scoring(results, cleaning_method, target_criteria, flipped)
                dirty_scores.append(score)
                dirty_accs.append(results[f"{cleaning_method}_test_acc"])

            else:
                for test_method in [dirty, train_method]:
                    approach = (train_method, test_method)

                    if approach not in cleaning_scores:
                        cleaning_scores[approach] = []

                    cleaning_method = "clean" if error_type == "mislabel" else test_method
                    scores = scoring(results, cleaning_method, target_criteria, flipped)
                    cleaning_scores[approach].append(scores)

                    if approach not in cleaning_accs:
                        cleaning_accs[approach] = []

                    if test_method == dirty:
                        cleaning_accs[approach].append(results[f"{dirty}_test_acc"])
                    elif error_type != "mislabel":
                        cleaning_accs[approach].append(results[f"{train_method}_test_acc"])

    evaluate_scores(dirty_scores, cleaning_scores, dirty_accs, cleaning_accs, 
                    dataset_name, target_criteria, metric_name, model, error_type, log_file)


def main():
    args = parse_args()

    data = load_data(args.results_folder, args.cardio_age_threshold)
    priv, dis = define_priv_dis(args.intersectional_formulation)

    errors = ["outliers", "missing_values", "mislabel"]
    models = ["logistic_regression", "knn_classification", "XGBoost"]
    metrics = [("equal_opportunity", lambda st, cl, cr, fl: compute_eo(st, cl, cr, fl, priv, dis)),
               ("predictive_parity", lambda st, cl, cr, fl: compute_pp(st, cl, cr, fl, priv, dis)),
               ("demographic_parity", lambda st, cl, cr, fl: compute_dp(st, cl, cr, fl, priv, dis))]

    if args.intersectional_formulation:
        results_filename = os.path.join(args.results_folder,
                                        f"cleanml_{args.cardio_age_threshold}_{args.intersectional_formulation}.csv")
        cases = [
            ("adult", ("sex", "race")),
            ("folktables", ("sex", "rac1p")),
            ("heart", ("gender", "age")),
            ("german", ("age", "sex")),
            # ("german", ("age", "foreign_worker")),
            # ("german", ("sex", "foreign_worker")),
        ]
    else:
        results_filename = os.path.join(args.results_folder, "cleanml.csv")
        cases = [
            ("adult", "sex"),
            ("adult", "race"),
            ("folktables", "sex"),
            ("folktables", "rac1p"),
            ("credit", "age"),
            ("german", "age"),
            ("heart", "gender"),
        ]

    with open(results_filename, "w") as log_file:
        print("dataset,criteria,metric,model,error,detection,repair,test_repaired,fairness_impact,accuracy_impact", file=log_file)

        for metric, scoring in metrics:
            for error in errors:
                for model in models:
                    for name, crit in cases:
                        flipped = (name in ["credit", "german"])
                        count(data[name], name, crit, error, model, metric, scoring, log_file, flipped=flipped)


if __name__ == "__main__":
    main()
