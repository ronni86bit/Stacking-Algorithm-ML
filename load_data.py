import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def load_iris_data():
    # Load the Iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target  # Add target labels
    
    return df

if __name__ == "__main__":
    df = load_iris_data()
    
    # Display basic info
    print(df.head())  # Show first 5 rows
    print("\nDataset Shape:", df.shape)
    print("\nClass Distribution:\n", df['target'].value_counts())

    # Visualize class distribution
    sns.pairplot(df, hue="target", diag_kind="kde")
    plt.show()

