"""Tasks running the results formatting (tables, figures)."""

import pandas as pd
import pytask

from EN.config import BLD, GROUPS, SRC
from EN.final import plot_regression_by_age
from EN.final.cache_deletion import delete_caches_in_directory
from EN.utilities import read_yaml


@pytask.mark.skip
@pytask.mark.depends_on(
    {
        "folder": BLD / "python",
    },
)
@pytask.mark.task()
def task_cache_deletion(depends_on):
    """Delete all caches after a run."""
    delete_caches_in_directory(depends_on["folder"])


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
