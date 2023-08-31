import duckdb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from demodq.datasets import Datasets
from demodq.column_errors import detect_missing_values, detect_outliers_sd, detect_outliers_iqr
from demodq.tuple_errors import detect_mislabeled_via_cleanlab, detect_outliers_via_if, detect_mislabeled_via_shapley
from demodq.analysis import is_disparate


def perc(frac):
    if frac == np.NAN: return frac
    return str(round(frac * 100, 1)) + '\%'


def _init():
    duckdb.sql("""CREATE TABLE groups(
        dataset VARCHAR,
        attr1 VARCHAR,
        attr2 VARCHAR,
        pp_count INTEGER,
        pd_count INTEGER,
        dp_count INTEGER,
        dd_count INTEGER,
        PRIMARY KEY(dataset, attr1, attr2));""")

    duckdb.sql("""CREATE TABLE errors(
        dataset VARCHAR,
        attr1 VARCHAR,
        attr2 VARCHAR,
        error_type VARCHAR,
        pp_dirty INTEGER,
        pd_dirty INTEGER,
        dp_dirty INTEGER,
        dd_dirty INTEGER,
        disparate BOOLEAN,
        PRIMARY KEY(dataset, attr1, attr2, error_type));""")


def _detect_missing_values(dataset_name, attr1, attr2, insert_groups=False):
    dataset = Datasets.load(dataset_name)

    data = dataset.as_df().copy(deep=True)
    data['id'] = range(len(data))

    dirty_slices = []

    for column in dataset.categorical_columns + dataset.numerical_columns:
        dirty_slices.append(detect_missing_values(data, column))

    dirty = pd.concat(dirty_slices).drop_duplicates(subset='id')

    data_priv_priv, data_priv_nonpriv, data_nonpriv_priv, data_nonpriv_nonpriv = \
        dataset.partition_data_by_intersection(data, attr1, attr2)
    dirty_priv_priv, dirty_priv_nonpriv, dirty_nonpriv_priv, dirty_nonpriv_nonpriv = \
        dataset.partition_data_by_intersection(dirty, attr1, attr2)

    if insert_groups:
        duckdb.sql(f"INSERT INTO groups VALUES ('{dataset_name}', '{attr1}', '{attr2}', "\
                f"{len(data_priv_priv)}, {len(data_priv_nonpriv)}, "\
                f"{len(data_nonpriv_priv)}, {len(data_nonpriv_nonpriv)})")

    disparate = is_disparate(len(data_priv_priv), len(dirty_priv_priv), len(data_nonpriv_nonpriv), len(dirty_nonpriv_nonpriv))

    duckdb.sql(f"INSERT INTO errors VALUES ('{dataset_name}', '{attr1}', '{attr2}', 'missing-values',"\
               f"{len(dirty_priv_priv)}, {len(dirty_priv_nonpriv)},"\
               f"{len(dirty_nonpriv_priv)}, {len(dirty_nonpriv_nonpriv)}, {disparate})")


def _detect_outliers(dataset_name, attr1, attr2):
    dataset = Datasets.load(dataset_name)

    data = dataset.as_df().copy(deep=True)
    data['id'] = range(len(data))

    for detector, name in [(detect_outliers_sd, 'sd'), (detect_outliers_iqr, 'iqr')]:
        dirty_slices = []

        for column in dataset.numerical_columns:
            dirty_slices.append(detector(data, column))

        dirty = pd.concat(dirty_slices).drop_duplicates(subset='id')

        data_priv_priv, _, _, data_nonpriv_nonpriv = \
            dataset.partition_data_by_intersection(data, attr1, attr2)
        dirty_priv_priv, dirty_priv_nonpriv, dirty_nonpriv_priv, dirty_nonpriv_nonpriv = \
            dataset.partition_data_by_intersection(dirty, attr1, attr2)

        disparate = is_disparate(len(data_priv_priv), len(dirty_priv_priv), len(data_nonpriv_nonpriv), len(dirty_nonpriv_nonpriv))

        duckdb.sql(f"INSERT INTO errors VALUES ('{dataset_name}', '{attr1}', '{attr2}', 'outliers-{name}',"\
                   f"{len(dirty_priv_priv)}, {len(dirty_priv_nonpriv)},"\
                   f"{len(dirty_nonpriv_priv)}, {len(dirty_nonpriv_nonpriv)}, {disparate})")

    # Isolation Forest
    dirty = detect_outliers_via_if(data, dataset, 1234)  # TODO: Try different seeds

    data_priv_priv, _, _, data_nonpriv_nonpriv = \
        dataset.partition_data_by_intersection(data, attr1, attr2)
    dirty_priv_priv, dirty_priv_nonpriv, dirty_nonpriv_priv, dirty_nonpriv_nonpriv = \
        dataset.partition_data_by_intersection(dirty, attr1, attr2)

    disparate = is_disparate(len(data_priv_priv), len(dirty_priv_priv), len(data_nonpriv_nonpriv), len(dirty_nonpriv_nonpriv))

    duckdb.sql(f"INSERT INTO errors VALUES ('{dataset_name}', '{attr1}', '{attr2}', 'outliers-if',"\
               f"{len(dirty_priv_priv)}, {len(dirty_priv_nonpriv)},"\
               f"{len(dirty_nonpriv_priv)}, {len(dirty_nonpriv_nonpriv)}, {disparate})")


def _detect_label_errors(dataset_name, attr1, attr2):
    dataset = Datasets.load(dataset_name)

    data = dataset.as_df().copy(deep=True)
    data['id'] = range(len(data))

    for detector, name in [(detect_mislabeled_via_cleanlab, 'cl'), (detect_mislabeled_via_shapley, 'shap')]:
        dirty = detector(data, dataset, 1234)  # TODO: Try different seeds
                                               # NOTE: For cleanlab, this seed is not even used!!!

        data_priv_priv, _, _, data_nonpriv_nonpriv = \
            dataset.partition_data_by_intersection(data, attr1, attr2)
        dirty_priv_priv, dirty_priv_nonpriv, dirty_nonpriv_priv, dirty_nonpriv_nonpriv = \
            dataset.partition_data_by_intersection(dirty, attr1, attr2)

        disparate = is_disparate(len(data_priv_priv), len(dirty_priv_priv), len(data_nonpriv_nonpriv), len(dirty_nonpriv_nonpriv))

        duckdb.sql(f"INSERT INTO errors VALUES ('{dataset_name}', '{attr1}', '{attr2}', 'mislabels-{name}',"\
                   f"{len(dirty_priv_priv)}, {len(dirty_priv_nonpriv)},"\
                   f"{len(dirty_nonpriv_priv)}, {len(dirty_nonpriv_nonpriv)}, {disparate})")


def _get_error_rates(filename):
    df = duckdb.sql("""SELECT
    e.dataset, e.attr1, e.attr2, e.error_type, e.disparate,

    pp_dirty, pp_count,
    IF(e.disparate, pp_dirty/pp_count, 0.0) AS pp_dirty__pp_count,
    pp_count/total_count AS pp_count__total_count,
    pp_dirty/total_dirty AS pp_dirty__total_dirty,

    pd_dirty, pd_count,
    IF(e.disparate, pd_dirty/pd_count, 0.0) AS pd_dirty__pd_count,
    pd_count/total_count AS pd_count__total_count,
    pd_dirty/total_dirty AS pd_dirty__total_dirty,

    dp_dirty, dp_count,
    IF(e.disparate, dp_dirty/dp_count, 0.0) AS dp_dirty__dp_count,
    dp_count/total_count AS dp_count__total_count,
    dp_dirty/total_dirty AS dp_dirty__total_dirty,

    dd_dirty, dd_count,
    IF(e.disparate, dd_dirty/dd_count, 0.0) AS dd_dirty__dd_count,
    dd_count/total_count AS dd_count__total_count,
    dd_dirty/total_dirty AS dd_dirty__total_dirty,

    (pp_dirty + pd_dirty + dp_dirty) AS pagg_dirty,
    (pp_count + pd_count + dp_count) AS pagg_count,
    IF(e.disparate, (pp_dirty + pd_dirty + dp_dirty) / (pp_count + pd_count + dp_count), 0.0) AS pagg_dirty__pagg_count,
    (pp_count + pd_count + dp_count) / total_count AS pagg_count__total_count,
    (pp_dirty + pd_dirty + dp_dirty) / total_dirty AS pagg_dirty__total_dirty,

    (pd_dirty + dp_dirty + dd_dirty) AS dagg_dirty,
    (pd_count + dp_count + dd_count) AS dagg_count,
    IF(e.disparate, (pd_dirty + dp_dirty + dd_dirty) / (pd_count + dp_count + dd_count), 0.0) AS dagg_dirty__dagg_count,
    (pd_count + dp_count + dd_count) / total_count AS dagg_count__total_count,
    (pd_dirty + dp_dirty + dd_dirty) / total_dirty AS dagg_dirty__total_dirty,
FROM (
    SELECT
        dataset, attr1, attr2, error_type, disparate,
        pp_dirty, pd_dirty, dp_dirty, dd_dirty,
        (pp_dirty + pd_dirty + dp_dirty + dd_dirty) AS total_dirty,
    FROM errors
) e
JOIN (
    SELECT
        dataset, attr1, attr2,
        pp_count, pd_count, dp_count, dd_count,
        (pp_count + pd_count + dp_count + dd_count) AS total_count,
    FROM groups
) g
ON g.dataset = e.dataset AND g.attr1 = e.attr1 AND g.attr2 = e.attr2""").df()
    df.to_csv(filename, index=False)
    return df

def _show_plot(cases, error_rates_df):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rc('text', usetex=True)
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'

    legend = ["int. priv", "int. dis"]
    query = """
    SELECT
        round(100*pp_dirty__pp_count, 1) AS priv_pct,
        pp_dirty AS priv_count,
        round(100*dd_dirty__dd_count, 1) AS dis_pct,
        dd_dirty AS dis_count,
    FROM error_rates_df
    WHERE error_type = '{}' AND attr2 != 'age@55'
    """
    error_rates = duckdb.sql("""
    SELECT
        dataset, attr1, attr2,
        round(100*pp_dirty__pp_count, 1) AS priv_pct,
        pp_dirty AS priv_count,
        round(100*dd_dirty__dd_count, 1) AS dis_pct,
        dd_dirty AS dis_count,
    FROM error_rates_df
    """).df()

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)

    xs = np.arange(len(cases))

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_xticks(xs)
        ax.set_xticklabels([f'{dataset}/{attr1}+{attr2.split("@")[0]}' for dataset, attr1, attr2 in cases], rotation=90)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        ax.set_ylim((0, 90))
        ax.set_xlim((-0.5, -0.5 + len(cases)))

        if ax != ax1:
            ax.get_yaxis().set_visible(False)

    color_priv = '#E9D5CA'
    color_dis = '#4D4C7D'

    datasets = tuple(duckdb.sql("SELECT dataset FROM error_rates").df().dataset[:4])
    print("datasets:", datasets == list(zip(*cases))[0])

    # --- missing values
    missing_values_df = duckdb.sql(query.format("missing-values")).df()
    # assert missing_values_df.dataset.tolist() == sorted_datasets, missing_values_df.dataset.tolist()
    ax1.bar(xs - 0.15, missing_values_df.priv_pct.tolist(), width=0.3, edgecolor='black', color=color_priv)
    ax1.bar(xs + 0.15, missing_values_df.dis_pct.tolist(), width=0.3, edgecolor='black', color=color_dis)
    ax1.set_title('missing values', fontsize=20)
    ax1.set_yticks([10, 20, 30, 40, 50, 60])
    ax1.set_ylabel('fraction of flagged\ntuples per group [\%]', fontsize=22)
    ax1.legend(legend, fontsize=16)

    xpos = -0.3
    for _, row in missing_values_df.iterrows():
        priv_pct, priv_count, dis_pct, dis_count = row.tolist()
        if priv_pct and priv_count:
            ax1.text(xpos, priv_pct + 5, str(int(priv_count)), fontsize=16, rotation=90)
        if dis_pct and dis_count:
            ax1.text(xpos + 0.35, dis_pct + 5, str(int(dis_count)), fontsize=16, rotation=90)
        xpos += 1

    # --- outliers-sd
    outliers_sd_df = duckdb.sql(query.format("outliers-sd")).df()
    # assert outliers_sd_df.dataset.tolist() == sorted_datasets, outliers_sd_df.dataset.tolist()
    ax2.bar(xs - 0.15, outliers_sd_df.priv_pct.tolist(), width=0.3, edgecolor='black', color=color_priv)
    ax2.bar(xs + 0.15, outliers_sd_df.dis_pct.tolist(), width=0.3, edgecolor='black', color=color_dis)
    ax2.set_title('outliers-sd\n(standard deviations)', fontsize=20)
    ax2.legend(legend, fontsize=16)

    xpos = -0.3
    for _, row in outliers_sd_df.iterrows():
        priv_pct, priv_count, dis_pct, dis_count = row.tolist()
        if priv_pct and priv_count:
            ax2.text(xpos, priv_pct + 5, str(int(priv_count)), fontsize=16, rotation=90)
        if dis_pct and dis_count:
            ax2.text(xpos + 0.35, dis_pct + 5, str(int(dis_count)), fontsize=16, rotation=90)
        xpos += 1

    # --- outliers-iqr
    outliers_iqr_df = duckdb.sql(query.format("outliers-iqr")).df()
    # assert outliers_iqr_df.dataset.tolist() == sorted_datasets, outliers_iqr_df.dataset.tolist()
    ax3.bar(xs - 0.15, outliers_iqr_df.priv_pct.tolist(), width=0.3, edgecolor='black', color=color_priv)
    ax3.bar(xs + 0.15, outliers_iqr_df.dis_pct.tolist(), width=0.3, edgecolor='black', color=color_dis)
    ax3.set_title('outliers-iqr\n(inter-quartile range)', fontsize=20)
    ax3.set_xlabel('dataset \& sensitive attribute', fontsize=22)
    ax3.legend(legend, fontsize=16)

    xpos = -0.3
    for _, row in outliers_iqr_df.iterrows():
        priv_pct, priv_count, dis_pct, dis_count = row.tolist()
        if priv_pct and priv_count:
            ax3.text(xpos, priv_pct + 5, str(int(priv_count)), fontsize=16, rotation=90)
        if dis_pct and dis_count:
            ax3.text(xpos + 0.35, dis_pct + 5, str(int(dis_count)), fontsize=16, rotation=90)
        xpos += 1

    # --- outliers-if
    outliers_if_df = duckdb.sql(query.format("outliers-if")).df()
    # assert outliers_if_df.dataset.tolist() == sorted_datasets, outliers_if_df.dataset.tolist()
    ax4.bar(xs - 0.15, outliers_if_df.priv_pct.tolist(), width=0.3, edgecolor='black', color=color_priv)
    ax4.bar(xs + 0.15, outliers_if_df.dis_pct.tolist(), width=0.3, edgecolor='black', color=color_dis)
    ax4.set_title('outliers-if\n(isolation forest)', fontsize=20)
    ax4.legend(legend, fontsize=16)

    xpos = -0.3
    for _, row in outliers_if_df.iterrows():
        priv_pct, priv_count, dis_pct, dis_count = row.tolist()
        if priv_pct and priv_count:
            ax4.text(xpos, priv_pct + 5, str(int(priv_count)), fontsize=16, rotation=90)
        if dis_pct and dis_count:
            ax4.text(xpos + 0.35, dis_pct + 5, str(int(dis_count)), fontsize=16, rotation=90)
        xpos += 1

    # --- mislabels-cl
    mislabels_cl_df = duckdb.sql(query.format("mislabels-cl")).df()
    # assert mislabels_cl_df.dataset.tolist() == sorted_datasets, mislabels_cl_df.dataset.tolist()
    ax5.bar(xs - 0.15, mislabels_cl_df.priv_pct.tolist(), width=0.3, edgecolor='black', color=color_priv)
    ax5.bar(xs + 0.15, mislabels_cl_df.dis_pct.tolist(), width=0.3, edgecolor='black', color=color_dis)
    ax5.set_title('label errors', fontsize=20)
    ax5.legend(legend, fontsize=16)

    xpos = -0.3
    for _, row in mislabels_cl_df.iterrows():
        priv_pct, priv_count, dis_pct, dis_count = row.tolist()
        if priv_pct and priv_count:
            ax5.text(xpos, priv_pct + 5, str(int(priv_count)), fontsize=16, rotation=90)
        if dis_pct and dis_count:
            ax5.text(xpos + 0.35, dis_pct + 5, str(int(dis_count)), fontsize=16, rotation=90)
        xpos += 1

    width, height = 16, 6
    fig.set_size_inches(width, height)
    plt.tight_layout()

    os.makedirs("figure", exist_ok=True)
    filename = os.path.join("figure", f"flagged-pp_dd-age@45-{width}x{height}-group_sizes.pdf")
    print(filename)
    plt.gcf().savefig(filename, dpi=300)
    # plt.show()


def main():
    cases = [('adult', 'sex', 'race'), ('folktables', 'sex', 'race'),
             ('german', 'age', 'sex'), ('heart', 'sex', 'age@45')]

    error_rates_filename = "error_rates_intersectional.csv"
    if os.path.exists(error_rates_filename):
        error_rates_df = pd.read_csv(error_rates_filename)

    else:
        _init()

        for dataset_name, attr1, attr2 in cases:
            _detect_missing_values(dataset_name, attr1, attr2, insert_groups=True)

        for dataset_name, attr1, attr2 in cases:
            _detect_outliers(dataset_name, attr1, attr2)

        for dataset_name, attr1, attr2 in cases:
            _detect_label_errors(dataset_name, attr1, attr2)

        error_rates_df = _get_error_rates(error_rates_filename)

    _show_plot(cases, error_rates_df)


if __name__ == "__main__":
    main()
