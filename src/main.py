from data_preprocessing import preprocess_data
from linear_regression_model import train_model

def main():
    # Chemins des fichiers
    train_path = "../data/customer_churn_dataset-training-master.csv"
    test_path = "../data/customer_churn_dataset-testing-master.csv"

    # Préparation des données
    X, y = preprocess_data(train_path, test_path)
    print("Données préparées avec succès.")

    # Vérification des valeurs manquantes
    if X.isnull().values.any() or y.isnull().values.any():
        print("Les données contiennent encore des valeurs manquantes !")
        return

    # Entraînement du modèle
    model, accuracy, report = train_model(X, y)
    print(f"Précision du modèle : {accuracy:.2f}")
    print("Rapport de régression :\n", report)

if __name__ == "__main__":
    main()
