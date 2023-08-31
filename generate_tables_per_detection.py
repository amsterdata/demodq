from argparse import ArgumentParser
import math
import os

import duckdb
from tabulate import tabulate


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--results-filename", required=True)
    parser.add_argument("--save", action="store_true")

    return parser.parse_args()


def single(results, detection, fairness_impact, accuracy_impact):
    if not isinstance(detection, str) and math.isnan(detection):
        result_slice = results[(results.detection.isna()) & (results.fairness_impact == fairness_impact) & \
            (results.accuracy_impact == accuracy_impact)]
    else:
        result_slice = results[(results.detection == detection) & (results.fairness_impact == fairness_impact) & \
                (results.accuracy_impact == accuracy_impact)]

    if len(result_slice) > 0:
        return list(result_slice["count"])[0]
    else:
        return 0


def perc(count, total):
    if not total: return "nan"
    return str(round((count / total) * 100, 1)) + f"\% ({count})"


def make_tex_string(cpn, cpi, cpp, cin, cii, cip, cnn, cni, cnp):
    total = cpn + cpi + cpp + cin + cii + cip + cnn + cni + cnp

    return r"""\begin{tabular}{cl|ccc|r}
& & \multicolumn{3}{|c|}{\textbf{accuracy}} & \\
& & \textbf{worse} & \textbf{insignificant} & \textbf{better} & \\
\hline
\multirow{3}{*}{\rotatebox{}{\textbf{fair.}}} & \textbf{worse} & """ + " & ".join([
    perc(cnn, total),
    perc(cni, total),
    perc(cnp, total),
    perc(cnn + cni + cnp, total),
]) + r""" \\
& \textbf{insign.} & """ + " & ".join([
    perc(cin, total),
    perc(cii, total),
    perc(cip, total),
    perc(cin + cii + cip, total),
]) + r""" \\
& \textbf{better} & """ + " & ".join([
    perc(cpn, total),
    perc(cpi, total),
    perc(cpp, total),
    perc(cpn + cpi + cpp, total),
]) + r""" \\
\hline
&& """ + " & ".join([
    perc(cpn + cin + cnn, total),
    perc(cpi + cii + cni, total),
    perc(cpp + cip + cnp, total),
]) + r""" & \\
\end{tabular}"""


def main():
    args = parse_args()

    counts = duckdb.sql(f"""
        SELECT detection, fairness_impact, accuracy_impact, COUNT(*) as count
        FROM '{args.results_filename}'
        GROUP BY detection, fairness_impact, accuracy_impact
        ORDER BY detection, fairness_impact DESC, accuracy_impact DESC
    """).df()

    for detection in counts.detection.unique():
        print("%", detection)
        cpn = single(counts, detection, "positive", "negative")
        cpi = single(counts, detection, "positive", "insignificant")
        cpp = single(counts, detection, "positive", "positive")

        cin = single(counts, detection, "insignificant", "negative")
        cii = single(counts, detection, "insignificant", "insignificant")
        cip = single(counts, detection, "insignificant", "positive")

        cnn = single(counts, detection, "negative", "negative")
        cni = single(counts, detection, "negative", "insignificant")
        cnp = single(counts, detection, "negative", "positive")

        total = cpn + cpi + cpp + cin + cii + cip + cnn + cni + cnp

        tex_string = make_tex_string(cpn, cpi, cpp, cin, cii, cip, cnn, cni, cnp)
        # print(tex_string)

        table = [
            [
                "",
                "acc worse",
                "acc insig",
                "acc better",
                "",
            ],
            [
                "fair worse",
                perc(cnn, total),
                perc(cni, total),
                perc(cnp, total),
                perc(cnn + cni + cnp, total),
            ],
            [
                "fair insig",
                perc(cin, total),
                perc(cii, total),
                perc(cip, total),
                perc(cin + cii + cip, total),
            ],
            [
                "fair better",
                perc(cpn, total),
                perc(cpi, total),
                perc(cpp, total),
                perc(cpn + cpi + cpp, total),
            ],
            [
                "",
                perc(cpn + cin + cnn, total),
                perc(cpi + cii + cni, total),
                perc(cpp + cip + cnp, total),
                "",
            ],
        ]

        if args.save:
            results_folder = os.path.dirname(args.results_filename)
            tables_folder = os.path.join(results_folder, "tables_per_detection")
            os.makedirs(tables_folder, exist_ok=True)

            if os.path.basename(args.results_filename) == "cleanml.csv":
                tables_folder = os.path.join(tables_folder, "plain_binary")
            else:
                metadata = os.path.basename(args.results_filename).replace("cleanml_", "").replace(".csv", "")
                tables_folder = os.path.join(tables_folder, metadata)
            os.makedirs(tables_folder, exist_ok=True)

            tex_filename = os.path.join(tables_folder, f"{detection}.tex")
            with open(tex_filename, "w") as f:
                print(tex_string, file=f)

            txt_filename = os.path.join(tables_folder, f"{detection}.txt")
            with open(txt_filename, "w") as f:
                print(tabulate(table), file=f)


if __name__ == "__main__":
    main()
