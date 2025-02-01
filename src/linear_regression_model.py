from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

def train_model(X, y):
    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Création et entraînement du modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédictions (valeurs continues)
    y_pred = model.predict(X_test)

    # Application d'un seuil pour convertir les prédictions continues en 0 ou 1
    y_pred_class = np.where(y_pred > 0.5, 1, 0)

    # Calcul des métriques d'évaluation
    accuracy = np.mean(y_pred_class == y_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Score de fidélité en pourcentage
    fidelity_scores = model.predict(X_test) * 100  # Convertir les prédictions en pourcentage

    # Affichage des résultats de la prédiction de fidélité pour quelques clients
    print("\nQuelques scores de fidélité (en pourcentage) pour les clients testés :")
    for i in range(5):
        print(f"Client {i + 1}: {fidelity_scores[i]:.2f}% fidélité")

    # Calcul de la courbe ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Affichage de la courbe ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Ligne d'équilibre
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs ')
    plt.ylabel('Taux de Vrais Positifs ')
    plt.title('Courbe ROC')
    plt.legend(loc='lower right')
    plt.show()

    # Visualisation de l'importance des caractéristiques
    visualize_coefficients(model, X.columns)

    # Rapport des résultats
    report = f"Accuracy: {accuracy:.2f}\nMSE: {mse:.2f}\nR2: {r2:.2f}\nAUC: {roc_auc:.2f}"

    return model, accuracy, report

def visualize_coefficients(model, feature_names):
    # Importance des coefficients
    coefficients = np.abs(model.coef_)
    coefficients = pd.Series(coefficients, index=feature_names)
    coefficients.sort_values(ascending=False).plot(kind='barh', color='skyblue')
    plt.title("Importance des caractéristiques")
    plt.xlabel("Impact")
    plt.ylabel("Caractéristiques")
    plt.show()
