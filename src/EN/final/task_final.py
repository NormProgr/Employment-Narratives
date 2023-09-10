"""Tasks running the results formatting (tables, figures)."""

import pickle

import pytask
from tabulate import tabulate

from EN.config import BLD
from EN.final.cache_deletion import delete_caches_in_directory
from EN.final.plot import table_produce


@pytask.mark.skip
@pytask.mark.depends_on(
    {
        "scripts": ["cache_deletion.py"],
        "cache": BLD / "python",
    },
)
@pytask.mark.produces(
    {},
)
@pytask.mark.task()
def task_cache_deletion(
    depends_on,
    produces,
):  # does not work unfortunately, had no time to finish it
    """Delete all caches after a run."""
    delete_caches_in_directory(depends_on["cache"])


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
    with open(depends_on["input2"], "rb") as file:
        result2 = pickle.load(file)
    result1, result2 = table_produce(result1, result2)

    markdown_table = tabulate(result1, headers="keys", tablefmt="pipe")
    markdown_table2 = tabulate(result2, headers="keys", tablefmt="pipe")

    # Save the Markdown table to a file
    with open(produces["training_results"], "w") as md_file:
        md_file.write(markdown_table)
    with open(produces["zero_shot_results"], "w") as md_file:
        md_file.write(markdown_table2)
