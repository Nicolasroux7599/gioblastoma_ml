#feature_selection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

def print_time(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

print_time("Démarrage du script...")

# ==========================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==========================================
print_time("Chargement des fichiers CSV...")
full_features = pd.read_csv("final_sequence_dataset_features.csv")
df_labels = pd.read_csv("Dataset_gbm.csv")

df_features_clean = full_features.drop_duplicates(subset=['UniProt ID', 'Mutation'])
df_labels_clean = df_labels.drop_duplicates(subset=['UniProt ID', 'Mutation'])

df_final = pd.merge(df_features_clean, 
                    df_labels_clean[['UniProt ID', 'Mutation', 'Class']], 
                    on=['UniProt ID', 'Mutation'], 
                    how='inner')

structures_dict = {
    'Alpha': df_final[df_final['secondary_structure'] == 0].copy(),
    'Beta': df_final[df_final['secondary_structure'] == 1].copy(),
    'Coil': df_final[df_final['secondary_structure'] == 2].copy()
}

# ==========================================
# 2. DÉFINITION DES MODÈLES ET HYPERPARAMÈTRES
# ==========================================
models_config = {
    'Alpha': {
        "SVM": SVC(C=1, kernel='rbf', gamma='scale'),
        "KNN": KNeighborsClassifier(metric='manhattan', n_neighbors=21, weights='uniform', n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', solver='sgd', alpha=0.05678201970293124, learning_rate='adaptive'),
        "DL": MLPClassifier(activation = 'relu', alpha= np.float64(0.00010994335574766199), hidden_layer_sizes= (512, 256), learning_rate_init= 0.0001, solver= 'adam')
    },
    'Beta': {
        "SVM": SVC(kernel='rbf', gamma='scale', C=1), # Ex: Hyperparamètres de l'article
        "KNN": KNeighborsClassifier(metric='manhattan', n_neighbors=23, weights='uniform', n_jobs=-1),
        "MLP": MLPClassifier(activation='tanh', alpha=np.float64(0.05678201970293124), hidden_layer_sizes=(100,), learning_rate='adaptive', solver='sgd'),
        "DL": MLPClassifier(activation = 'relu', alpha= np.float64(0.00013066739238053285), hidden_layer_sizes= (512, 512), learning_rate_init= 0.001, solver= 'adam')
    },
    'Coil': {
        "SVM": SVC(kernel='rbf', gamma='scale', C=1),
        "KNN": KNeighborsClassifier(metric='manhattan', n_neighbors=21, weights='uniform', n_jobs=-1), # À remplacer par tes vrais paramètres Coil
        "MLP": MLPClassifier(activation='relu', alpha=np.float64(0.002950706670790532), hidden_layer_sizes=(100,), learning_rate='adaptive', solver='adam'),
        "DL": MLPClassifier(activation = 'relu', alpha= np.float64(0.00010994335574766199), hidden_layer_sizes= (512, 256), learning_rate_init= 0.0001, solver= 'adam')
    }
}

feature_steps = list(range(5, 155, 5))
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_names = list(models_config['Alpha'].keys())

all_results = {struct: {model: {'mean': [], 'std': []} for model in model_names} for struct in structures_dict}

# ==========================================
# 3. ÉVALUATION ET SÉLECTION DE FEATURES
# ==========================================
for struct_name, df_subset in structures_dict.items():
    print_time(f"Début du traitement de la structure : {struct_name}")
    
    y = df_subset['Class'].map({'Driver': 1, 'Passenger': 0})
    
    cols_to_drop = [
        'UniProt ID', 'Gene Name', 'Mutation', 'Wild', 'Mut', 'Pos', 'Class', 
        'secondary_structure', 'tri', 'n_M', 'M_c', 'n__M', 'M__c', 'nM', 'Mc'
    ]
    or_float_cols = [c for c in df_subset.columns if 'odds_rat' in c or 'odds_ra' in c or 'odds_ratio' in c]
    cols_to_drop.extend(or_float_cols)
    
    X = df_subset.drop(columns=[c for c in cols_to_drop if c in df_subset.columns])
    
    # ---------------------------------------------------------
    # NOUVEAU : Split Train/Test (80% Train, 20% Test)
    # Stratification sur y pour conserver la proportion de classes
    # ---------------------------------------------------------
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardisation ajustée sur le Train set uniquement
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=X.columns) # Test set prêt pour une future évaluation finale
    
    # Calcul des importances sur le Train set uniquement
    forest = ExtraTreesClassifier(n_estimators=250, random_state=42, n_jobs=-1)
    forest.fit(X_train_scaled, y_train)
    importances = pd.Series(forest.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    for n_features in tqdm(feature_steps, desc=f"Testing feature counts ({struct_name})"):
        selected_cols = importances.head(n_features).index
        X_train_sel = X_train_scaled[selected_cols]
        # Note: X_test_sel = X_test_scaled[selected_cols] serait utilisé ici si on voulait prédire sur le test set
        
        for model_name, model_instance in models_config[struct_name].items():
            # La validation croisée est exécutée uniquement sur la portion Train
            cv_scores = cross_val_score(model_instance, X_train_sel, y_train, cv=cv_strategy, scoring='roc_auc', n_jobs=-1)
            all_results[struct_name][model_name]['mean'].append(float(cv_scores.mean()))
            all_results[struct_name][model_name]['std'].append(float(cv_scores.std()))

    print_time(f"Fin du traitement de la structure : {struct_name}")

# ==========================================
# 3.5 SAUVEGARDE DE SÉCURITÉ DES DONNÉES
# ==========================================
print_time("Sauvegarde des données brutes en JSON...")
with open('raw_results_feature_selection.json', 'w') as f:
    json.dump(all_results, f, indent=4)

# ==========================================
# 4. CRÉATION DES GRAPHIQUES
# ==========================================
print_time("Génération des graphiques...")
fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=False)
colors = {'SVM': 'tab:red', 'KNN': 'tab:green', 'MLP': 'tab:blue', 'DL': 'tab:orange'}
markers = {'SVM': 'o', 'KNN': 'o', 'MLP': 'o', 'DL': 'o'}

for ax, struct_name in zip(axes, structures_dict.keys()):
    for model_name in model_names:
        means = np.array(all_results[struct_name][model_name]['mean'])
        stds = np.array(all_results[struct_name][model_name]['std'])
        
        ax.plot(feature_steps, means, label=model_name, 
                color=colors[model_name], marker=markers[model_name], markersize=4, lw=1.5)
        
        ax.fill_between(feature_steps, means - stds, means + stds, 
                        color=colors[model_name], alpha=0.15)
        
        max_idx = np.argmax(means)
        max_x = feature_steps[max_idx]
        max_y = means[max_idx]
        ax.scatter(max_x, max_y, marker='*', s=200, color=colors[model_name], 
                   edgecolor='black', zorder=5)

    ax.set_title(f"{struct_name} — MDI feature selection (Train set)\n($\\star$ = automatic idxmax peak)")
    ax.set_xlabel("Number of features")
    ax.set_ylabel("5-fold CV AUC")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

print_time("Sauvegarde du graphique en PDF...")
plt.savefig("feature_selection_results.pdf", format="pdf", bbox_inches="tight")
plt.savefig("feature_selection_results.png", dpi=300, bbox_inches="tight")
print_time("Script terminé avec succès !")