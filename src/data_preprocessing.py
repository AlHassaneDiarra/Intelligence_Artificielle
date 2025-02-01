import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(train_path, test_path):
    # Chargement des données
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Concaténation des deux ensembles
    data = pd.concat([train_data, test_data], ignore_index=True)

    # Suppression des colonnes inutiles
    data = data.drop(columns=['CustomerID', 'Last Interaction', 'Subscription Type', 'Contract Length'])

    # Conversion de la colonne 'Gender' en variables numériques
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

    # Gestion des valeurs manquantes (remplacement par la moyenne)
    data.fillna(data.mean(), inplace=True)

    # Normalisation des colonnes numériques
    scaler = MinMaxScaler()
    data[['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend']] = scaler.fit_transform(
        data[['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend']]
    )

    # Conversion de la colonne cible 'Churn' en entier
    data['Churn'] = data['Churn'].astype(int)

    # Séparation des caractéristiques (X) et de la cible (y)
    X = data.drop(columns=['Churn'])
    y = data['Churn']

    return X, y
