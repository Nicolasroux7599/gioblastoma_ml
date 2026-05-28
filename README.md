# INFO-F439 : METHODS IN BIOFORMATIC

The aim of this project is to reimplement the methods of a scientific paper\
The paper is :\
Pandey, M., Anoosha, P., Yesudhas, D. and Gromiha, M. M. (2022), ‘Identification of potential
driver mutations in glioblastoma using machine learning’, Briefings in Bioinformatics 23(6).
doi: 10.1093/bib/bbac451.

## Abstract 
  Glioblastoma multiforme (GBM) is one of the most lethal primary brain tumours, with a
median survival of less than two years. A critical challenge in therapeutic development is the
identification of driver mutations, defined as variants that confer a selective growth advantage to
tumour cells, among the thousands of somatic mutations observed across patients. Pandey et al.
(2022) addressed this by developing GBMDriver, a GBM-specific machine learning classifier that
discriminates driver from passenger mutations using protein sequence-derived features, stratified
by predicted secondary structure, achieving an AUC of 0.87 on a blind test set. In this work, we
re-implement GBMDriver from the published description, critically assess its methodological
choices, and extend its interpretability. Our pipeline reproduces the core architectural choices :
secondary-structure stratification, sequence-only features, and ensemble classifiers, while also
introducing three methodological extensions: (i) a systematic comparison of MDI, permutation
importance, and SHAP-based feature rankings to guide feature selection; (ii) an optimal
decision threshold derived by maximising balanced accuracy on out-of-fold predictions, rather
than the fixed 0.5 threshold used in the original work; and (iii) a SHAP beeswarm analysis for
the β-strand subclass, enabling directional interpretation of feature contributions. The final
models: AdaBoost (n = 35 features) for α-helix, RandomForest (n = 25 features) for β-strand,
and AdaBoost (n = 40 features) for coil achieve overall AUC values of 0.68, 0.79, and 0.72
respectively on the held-out test set (overall AUC = 0.73), compared to 0.89, 0.87, and 0.85 in
the original article. The performance gap is primarily attributable to the absence of PSSM
and conservation features, which were identified as the most discriminative descriptors in the
original work but could not be computed within our computational constraints. SHAP analysis
confirms the dominant role of relative solvent accessibility (RSA) across all subclasses, and
reveals interpretable contributions from substitution matrices and thermodynamic stability
features, consistent with the known biophysics of driver mutations.

### 
All files are in the data directory except for the NetSurf CSV files, which were too large to be pushed.
