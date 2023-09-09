"Functions that evaluate the zero-shot performance of the model."

from sklearn.metrics import accuracy_score


def calculate_accuracy_scores(hand_class, data, threshold=0.5):
    """Calculate accuracy scores for binary classification labels in merged DataFrames.

    Args:
        hand_class (pd.DataFrame): DataFrame containing manually labeled data.
        data (pd.DataFrame): DataFrame containing predicted labels and sequences.
        threshold (float, optional): Threshold for binary classification (default is 0.5).

    Returns:
        tuple: A tuple containing:
            - accuracy_scores (list of float): Accuracy scores for each label.
            - mean_accuracy (float): Mean accuracy score across all labels.

    """
    hand_class = hand_class.rename(
        columns={
            "government intervention": "government intervention true",
            "labor demand": "labor demand true",
            "labor supply": "labor supply true",
        },
    )

    merged_df = hand_class.merge(data, on="sequence", how="inner")

    for label in ["government intervention", "labor demand", "labor supply"]:
        merged_df[label] = merged_df.apply(
            lambda row: 1 if any(val > threshold for val in row["label"]) else 0,
            axis=1,
        )

    # Drop the original 'label' column
    merged_df = merged_df.drop("label", axis=1)

    # Replace NaN values with zeros in the predicted_labels DataFrame
    predicted_labels = merged_df[
        ["government intervention true", "labor demand true", "labor supply true"]
    ].fillna(0)

    true_labels = merged_df[["government intervention", "labor demand", "labor supply"]]

    accuracy_scores = [
        accuracy_score(true_labels[label], predicted_labels[label + " true"])
        for label in true_labels.columns
    ]

    mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)

    return accuracy_scores, mean_accuracy
