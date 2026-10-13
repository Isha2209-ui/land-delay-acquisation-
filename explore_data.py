import pandas as pd  # Used to work with the dataset in table form


# Read the CSV file and store it in a DataFrame
df = pd.read_csv("SIH26017_MoSPI_1477_projects.csv")


print("\n--- DATASET SHAPE ---")
print(df.shape)  # Shows the number of rows and columns


print("\n--- COLUMNS ---")
print(df.columns.tolist())  # Shows all column names in the dataset


print("\n--- FIRST 5 PROJECTS ---")
print(df.head())  # Displays the first five rows to check the data


print("\n--- DATA TYPES ---")
print(df.dtypes)  # Shows the type of data stored in each column


print("\n--- MISSING VALUES ---")
print(df.isnull().sum())  # Counts missing values in every column


print("\n--- TARGET DISTRIBUTION ---")
print(df["is_delayed"].value_counts())  # Counts delayed and non-delayed projects


print("\n--- TARGET PERCENTAGES ---")
print(df["is_delayed"].value_counts(normalize=True) * 100)  # Shows the target values as percentages