demodq
===

Directory Structure
---

- [`cleanml-results`](cleanml-results) contains the model scores from every experimental condition.
- [`data`](data) contains the raw datasets.
- [`demodq`](demodq) is the code module written and used to run analysis.

Jupyter Notebooks:

- (**RQ1**) [`detect_errors-combined`](detect_errors-combined.ipynb) investigates error types for each dataset broken down by sensitive attributes.
- (**RQ1**) [`deep-dive-data-errors-mislabels`](deep-dive-data-errors-mislabels.ipynb) is an exploration of potentially mislabeled samples in the raw datasets.
- (**RQ2**) [`compute-result-table`](compute-result-table.ipynb) computes fairness metrics and statistical significance on the results in [`cleanml-results`](cleanml-results) and converts the raw result data structure into the result table in [`cleanml.csv`](cleanml.csv).
- (**RQ2**) [`cleanml-analysis`](cleanml-analysis.ipynb) generates the table in our paper that describes total case counts with negative, insignificant, and positive impact on fairness and on accuracy. It also examines how many experimental conditions had non-negative impact on fairness for each dataset and error type.
- (**RQ2**) [`cleanml-analysis-per-model`](cleanml-analysis-per-model.ipynb) groups all experiments by model type and error type and tallies the impact on fairness and on accuracy.
- (**RQ2**) [`cleanml-analysis-cleaning-type`](cleanml-analysis-cleaning-type.ipynb) counts cases with positive impact on fairness for each data cleaning method.
- (**RQ2**) [`cleanml-accuracies`](cleanml-accuracies.ipynb) identifies the best model types for each data error type with respect to model accuracy.

Setup
---

```shell
# Set up virtual env
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start Jupyter server
jupyter notebook
```
