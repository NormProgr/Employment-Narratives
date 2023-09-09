"Functions that evaluate the zero-shot performance of the model."

import pandas as pd
from sklearn.metrics import accuracy_score


def calculate_accuracy_scores(hand_class, data):
    """Calculate accuracy scores for binary classification labels in merged DataFrames.

    Args:
        hand_class (pd.DataFrame): DataFrame containing manually labeled data.
        data (pd.DataFrame): DataFrame containing predicted labels and sequences.

    Returns:
        dict: A dictionary containing the following key-value pairs:
            - 'Class Accuracy': list of float, Accuracy scores for each label.
            - 'Mean Accuracy': float, Mean accuracy score across all labels.
            - 'Class Name': list of str, Names of the columns for which accuracy scores were calculated.
            - 'Count Ones': list of int, Count of '1's in each column.

    """
    hand_class = hand_class.rename(
        columns={
            "government intervention": "government intervention true",
            "labor demand": "labor demand true",
            "labor supply": "labor supply true",
        },
    )

    merged_df = hand_class.merge(data, on="sequence", how="inner")

    merged_df[
        [
            "government intervention",
            "labor demand",
            "labor supply",
        ]
    ] = pd.DataFrame(merged_df["label"].tolist(), index=merged_df.index)

    merged_df = merged_df.drop("label", axis=1)

    merged_df[["government intervention", "labor supply", "labor demand"]] = merged_df[
        ["government intervention", "labor supply", "labor demand"]
    ].applymap(_zero_one_transform)
    predicted_labels = merged_df[
        ["government intervention true", "labor demand true", "labor supply true"]
    ].fillna(0)

    true_labels = merged_df[["government intervention", "labor demand", "labor supply"]]

    accuracy_scores = [
        accuracy_score(true_labels[label], predicted_labels[label + " true"])
        for label in true_labels.columns
    ]

    mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    column_names = true_labels.columns.tolist()
    ones_count = true_labels.sum().tolist()

    result_dict = {
        "Class Accuracy": accuracy_scores,
        "Mean Accuracy": mean_accuracy,
        "Class Name": column_names,
        "Count Ones": ones_count,
    }

    return result_dict


def _zero_one_transform(x):
    """Transform a float to a binary value.

    Args:
        x (float): A float value.

    Returns:
        int: 1 if x > 0.5, else 0.

    """
    if x > 0.5:
        return 1
    else:
        return 0
