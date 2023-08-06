"""Not replicaple part.

Preprocess data to get the source data.

"""

df_1 = pd.read_csv("C:/aaapro/Data/CNN_Articels_clean_raw.csv")
df_2 = pd.read_csv("C:/aaapro/Data/CNN_Articels_clean_2_raw.csv")

if not set(df_1.columns) == set(df_2.columns):
    raise ValueError("Both datasets must have the same columns.")


merged_dataset = pd.concat([df_1, df_2], axis=0)

merged_dataset.to_csv("merged_dataset.csv", index=False)
