import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Task 1: Load and Explore the Dataset
# Load the Iris dataset
try:
    # Load the dataset from a CSV file or directly using sklearn
    from sklearn.datasets import load_iris
    iris_data = load_iris()
    
    # Create a DataFrame from the Iris dataset
    iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    iris_df['species'] = iris_data.target
    
    # Map numerical species to actual names
    iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
except FileNotFoundError:
    print("The dataset file was not found.")
    exit()

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(iris_df.head())

# Explore the structure of the dataset
print("\nData types and missing values:")
print(iris_df.info())
print("\nMissing values in each column:")
print(iris_df.isnull().sum())

# Clean the dataset (in this case, there are no missing values)
# If there were missing values, you could fill or drop them
# iris_df.fillna(method='ffill', inplace=True)  # Example of filling missing values

# Task 2: Basic Data Analysis
# Compute basic statistics
print("\nBasic statistics of the numerical columns:")
print(iris_df.describe())

# Group by species and compute the mean of numerical columns
grouped_means = iris_df.groupby('species').mean()
print("\nMean values grouped by species:")
print(grouped_means)

# Identify interesting patterns
print("\nInteresting findings:")
print("The average petal length of 'virginica' is significantly higher than that of 'setosa'.")

# Task 3: Data Visualization
# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Line chart (using a hypothetical time-series)
# For demonstration, we will create a line chart of petal length over an index
plt.figure(figsize=(10, 6))
plt.plot(iris_df.index, iris_df['petal length (cm)'], marker='o', linestyle='-')
plt.title('Petal Length Over Index')
plt.xlabel('Index')
plt.ylabel('Petal Length (cm)')
plt.grid()
plt.show()

# Bar chart showing average petal length per species
plt.figure(figsize=(10, 6))
sns.barplot(x=grouped_means.index, y=grouped_means['petal length (cm)'])
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# Histogram of petal length
plt.figure(figsize=(10, 6))
sns.histplot(iris_df['petal length (cm)'], bins=10, kde=True)
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of sepal length vs petal length
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris_df, x='sepal length (cm)', y='petal length (cm)', hue='species', style='species', s=100)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()