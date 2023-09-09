"""Tasks running the results formatting (tables, figures)."""

import pandas as pd
import pytask

from EN.config import BLD, GROUPS, SRC
from EN.final.cache_deletion import delete_caches_in_directory
from EN.final.plot import table_produce
from EN.utilities import read_yaml


@pytask.mark.skip
@pytask.mark.depends_on(
    {
        "scripts": ["cache_deletion.py"],
        "cache": BLD / "python",
    },
)
@pytask.mark.task
# @pytask.mark.produces(
#    },
@pytask.mark.task()
def task_cache_deletion(depends_on):
    """Delete all caches after a run."""
    delete_caches_in_directory(depends_on["cache"])


import pickle

import pandas as pd

# pip install luigi
# pip install xhtml2pdf
from tabulate import tabulate


@pytask.mark.depends_on(
    {
        "scripts": ["cache_deletion.py"],
        "input1": BLD / "python" / "results" / "predict_to_save.pkl",
        "input2": BLD / "python" / "results" / "acc_scores_zer_shot.pkl",
    },
)
@pytask.mark.produces(
    {
        "training_results": BLD / "python" / "results" / "results.md",
        "zero_shot_results": BLD / "python" / "results" / "results2.md",
    },
)
def task_table_produce(depends_on, produces):
    """Produce the readme results to discuss."""
    with open(depends_on["input1"], "rb") as file:
        result1 = pickle.load(file)
    with open(depends_on["input1"], "rb") as file:
        result2 = pickle.load(file)
    result1, result2 = table_produce(result1, result2)

    markdown_table = tabulate(result1, headers="keys", tablefmt="pipe")
    markdown_table2 = tabulate(result2, headers="keys", tablefmt="pipe")

    # Save the Markdown table to a file
    with open(produces["training_results"], "w") as md_file:
        md_file.write(markdown_table)
    with open(produces["zero_shot_results"], "w") as md_file:
        md_file.write(markdown_table2)


for group in GROUPS:
    kwargs = {
        "group": group,
        "depends_on": {"predictions": BLD / "python" / "predictions" / f"{group}.csv"},
        "produces": BLD / "python" / "figures" / f"smoking_by_{group}.png",
    }

    @pytask.mark.skip
    @pytask.mark.depends_on(
        {
            "data_info": SRC / "data_management" / "data_info.yaml",
            "data": BLD / "python" / "data" / "data_clean.csv",
        },
    )
    @pytask.mark.task(id=group, kwargs=kwargs)
    def task_plot_results_by_age_python(depends_on, group, produces):
        """Plot the regression results by age (Python version)."""
        data_info = read_yaml(depends_on["data_info"])
        data = pd.read_csv(depends_on["data"])
        predictions = pd.read_csv(depends_on["predictions"])
        fig = plot_regression_by_age(data, data_info, predictions, group)
        fig.write_image(produces)


@pytask.mark.skip
@pytask.mark.depends_on(BLD / "python" / "models" / "model.pickle")
@pytask.mark.produces(BLD / "python" / "tables" / "estimation_results.tex")
def task_create_results_table_python(depends_on, produces):
    """Store a table in LaTeX format with the estimation results (Python version)."""
    model = load_model(depends_on)
    table = model.summary().as_latex()
    with open(produces, "w") as f:
        f.writelines(table)
