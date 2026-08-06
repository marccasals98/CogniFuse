import pandas as pd

dataset_path = "/home/usuaris/veussd/marc.casals/datasets/WAB_samples/labels.csv"

df = pd.read_csv(dataset_path)

print(df.head())
print(df["DX_Pilar"].unique())