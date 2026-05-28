#run_models
# Importations de base
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import time
from datetime import datetime

# VOS Modèles scikit-learn
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Outils statistiques pour RandomizedSearchCV
from scipy.stats import randint, loguniform

# Charger les deux fichiers
full_features = pd.read_csv("final_sequence_dataset_features.csv")
df_labels = pd.read_csv("Dataset_gbm.csv")

# Certaines mutations apparaissent plusieurs fois dans les datasets : on enlève les doublons 
df_features_clean = full_features.drop_duplicates(subset=['UniProt ID', 'Mutation'])
df_labels_clean = df_labels.drop_duplicates(subset=['UniProt ID', 'Mutation'])
print(f"Duplicates in the features : {full_features.duplicated(subset=['UniProt ID', 'Mutation']).sum()}")
print(f"Duplicates in the labels : {df_labels.duplicated(subset=['UniProt ID', 'Mutation']).sum()}")

#  Fusionner les DataFrames (Inner Join)
# Cela ne garde que les mutations présentes dans les deux fichiers
df_final = pd.merge(df_features_clean, 
                    df_labels_clean[['UniProt ID', 'Mutation', 'Class']], 
                    on=['UniProt ID', 'Mutation'], 
                    how='inner')

# Vérification
print(f"Features originales : {len(full_features)}")
print(f"Après fusion avec labels : {len(df_final)}")
print(f"Répartition des classes :\n{df_final['Class'].value_counts()}")


# Separate the full dataset into 3 subdatasets 
# 0 = Helix (Alpha), 1 = Strand (Beta), 2 = Coil
df_alpha = df_final[df_final['secondary_structure'] == 0].copy()
df_beta  = df_final[df_final['secondary_structure'] == 1].copy()
df_coil  = df_final[df_final['secondary_structure'] == 2].copy()

# Show lengths of each subdataset 
print(f"Total mutations : {len(df_final)}")
print("-" * 30)
print(f"Alpha-Helix (0) : {len(df_alpha)} mutations")
print(f"Beta-Strand (1) : {len(df_beta)} mutations")
print(f"Coil (2)        : {len(df_coil)} mutations")

def prepare_data(df):
    # 1. On isole la variable cible
    y = df['Class'].map({'Driver': 1, 'Passenger':0})

    # 2. On retire UNIQUEMENT les vraies métadonnées inutiles
    colonnes_a_supprimer = ['UniProt ID', 'Gene Name', 'Mutation', 'Wild', 'Mut', 'Pos', 'Class', 'secondary_structure', 'nM', 'Mc', 'tri',
                            'n_M', 'M_c', 'n__M', 'M__c']
    X = df.drop(columns=colonnes_a_supprimer)
    
    return X, y

def run_screening(X_train, y_train , subclass_name):
    
    # Different models to be tested   
    models = {
        "SVM": SVC(probability=True, random_state=42), # probability=True est OBLIGATOIRE pour l'AUC
        "KNN": KNeighborsClassifier(),
        "MLP": MLPClassifier(random_state=42, max_iter=1000), # max_iter élevé pour éviter que le MLP s'arrête trop tôt
        "DNN": MLPClassifier(
            random_state=42, 
            max_iter=1000, 
            early_stopping=True, # Indispensable pour gagner du temps sur les gros réseaux
            validation_fraction=0.1,
            n_iter_no_change=10
        )
    }
    
    results = []

    for name, model in models.items():
        
        # 1. Capture du temps au démarrage
        start_time = time.time()
        current_time = datetime.now().strftime("%H:%M:%S")
        print("="*50)
        print(f"[{current_time}] Début de l'optimisation pour le modèle : {name}")
        print("="*50)
        print(f"[{subclass_name}] Optimisation of {name}...")
        # RandomizedSearch with 10-fold CV and 10 iterations (meaning 10 hyperparameters set are tested)
        search = RandomizedSearchCV(
            model, param_grids[name],
            n_iter=10, cv=10, 
            scoring='roc_auc',
            n_jobs=-1, random_state=42
        )
        search.fit(X_train, y_train)

        results.append({
            "Subclass": subclass_name,
            "Model": name,
            "Best_Params": search.best_params_,
            "AUC": search.best_score_
            })
        
        # 3. Calcul de la durée à la fin
        end_time = time.time()
        elapsed_minutes = (end_time - start_time) / 60
    
        print(f"\n {name} terminé avec succès !")
        print(f" Temps d'exécution total pour {name} : {elapsed_minutes:.2f} minutes.\n")

    return pd.DataFrame(results)

# Generate data for each subclass
X_alpha, y_alpha = prepare_data(df_alpha)
X_beta, y_beta = prepare_data(df_beta)
X_coil, y_coil = prepare_data(df_coil)

# Generate the grid of hyperparameters for each Ensemble tree classifier and logistic regression 
# Grille d'hyperparamètres pour VOS modèles
param_grids = {
    "SVM": {
        # L'article utilise un noyau RBF et C=10. On encadre cette valeur.
        "C": [0.1, 1, 10, 50, 100], 
        "gamma": ['scale', 'auto', 0.01, 0.1, 1], # Taille du "rayon" d'influence des vecteurs de support
        "kernel": ['rbf'] # On force le RBF comme dans l'article [cite: 108, 156]
    },
    "KNN": {
        "n_neighbors": randint(3, 30), # Teste un nombre de voisins aléatoire entre 3 et 30
        "weights": ['uniform', 'distance'], # Donne le même poids à tous, ou plus de poids aux voisins proches
        "metric": ['euclidean', 'manhattan'] # Différentes façons de calculer la distance
    },
    "MLP": {
        # Différentes architectures : 1 couche de 50, 1 de 100, ou 2 couches (50 puis 25 neurones)
        "hidden_layer_sizes": [(50,), (100,), (50, 25), (100, 50)],
        "activation": ['relu', 'tanh'],
        "solver": ['adam', 'sgd'],
        "alpha": loguniform(1e-5, 1e-1), # Taux de régularisation (recherche logarithmique)
        "learning_rate": ['constant', 'adaptive']
    },
    "DNN": {
        "hidden_layer_sizes": [
            (256, 128, 64),       # Profond
            (512, 256),            # Large
            (128, 128, 128, 128),  # Très profond
            (512, 512)             # Très large
        ],
        "activation": ['relu'],
        "solver": ['adam'],
        "alpha": loguniform(1e-4, 1e-2),
        "learning_rate_init": [0.001, 0.0001]
    }
}

#split the data intro train and test set
X_train_alpha_raw, X_test_alpha_raw, y_train_alpha, y_test_alpha = train_test_split(X_alpha, y_alpha, test_size=0.2, random_state=42)
X_train_beta_raw, X_test_beta_raw, y_train_beta, y_test_beta = train_test_split(X_beta, y_beta, test_size=0.2, random_state=42)
X_train_coil_raw, X_test_coil_raw, y_train_coil, y_test_coil = train_test_split(X_coil, y_coil, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_alpha = scaler.fit_transform(X_train_alpha_raw)
X_test_alpha = scaler.transform(X_test_alpha_raw)

scaler = StandardScaler()
X_train_beta = scaler.fit_transform(X_train_beta_raw)
X_test_beta = scaler.transform(X_test_beta_raw)

scaler = StandardScaler()
X_train_coil = scaler.fit_transform(X_train_coil_raw)
X_test_coil = scaler.transform(X_test_coil_raw)

print("Train/test splits created.")
print(f"Alpha  — train: {len(y_train_alpha)}, test: {len(y_test_alpha)}")
print(f"Beta   — train: {len(y_train_beta)},  test: {len(y_test_beta)}")
print(f"Coil   — train: {len(y_train_coil)},  test: {len(y_test_coil)}")
 
# Run main loop of models screening
results_alpha = run_screening(X_train_alpha, y_train_alpha, "Alpha")
results_beta = run_screening(X_train_beta, y_train_beta, "Beta")
results_coil = run_screening(X_train_coil, y_train_coil, "Coil")

# Combine results for comparison
full_screening_report = pd.concat([results_alpha, results_beta, results_coil], ignore_index=True)
full_screening_report.to_csv("screening_report.csv", index=False)

# Print best models 
print(full_screening_report.sort_values(by="AUC", ascending=False))
# À la toute fin du script
total_end_time = datetime.now().strftime("%H:%M:%S")
print(f"🏁 Exécution globale du script terminée à {total_end_time}.")