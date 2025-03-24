import pandas as pd

# Load dataset
df = pd.read_csv("sample_100%.csv")

# Create smaller dataset (e.g., 10% of the original)
df_30 = df.sample(frac=0.1, random_state=1)

# Save the smaller dataset
df_30.to_csv("sample_10%.csv", index=False)

# Create smaller dataset (e.g., 50% of the original)
df_60 = df.sample(frac=0.5, random_state=1)

# Save the smaller dataset
df_60.to_csv("sample_50%.csv", index=False) 