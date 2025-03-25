import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
def load_iris_data():
    
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target 
    
    return df

if __name__ == "__main__":
    df = load_iris_data()
    print(df.head())  
    print("\nDataset Shape:", df.shape)
    print("\nClass Distribution:\n", df['target'].value_counts())
    sns.pairplot(df, hue="target", diag_kind="kde")
    plt.show()

