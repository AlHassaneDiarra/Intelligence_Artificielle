import matplotlib.pyplot as plt
import pandas as pd

def visualize_coefficients(model, feature_names):
    # Importance des caractéristiques
    coefficients = pd.Series(model.coef_[0], index=feature_names)
    coefficients.plot(kind='barh', color='skyblue')
    plt.title("Importance des caractéristiques")
    plt.xlabel("Impact")
    plt.ylabel("Caractéristiques")
    plt.show()
