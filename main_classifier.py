def get_messidor_dataset():
    import pandas as pd
    from scipy.io import arff
    data = arff.loadarff('messidor_features.arff')
    dataset = pd.DataFrame(data[0])
    X = dataset.iloc[:, 0: 19].values
    y = dataset.iloc[:, 19].values

    return [X,y]

if __name__ == "__main__":
    import classifier_generator as cg
    
    N_FOLDS = 3
    E_BEST_SCORE = 'roc_auc'
    DIPLAY_ALL_FEATURES = True
    DIPLAY_KPCA_FEATURES = True
    DIPLAY_LDA_FEATURES = True
    DIPLAY_SELK_FEATURES = True
    
    # Importamos o conjunto de dados
    X, y = get_messidor_dataset()
    energy_functions = cg.get_energy_functions()
    classifiers = cg.get_classifiers()
    
    # Feature Scaling
    X_sc = cg.feature_scaling(X)
    
    # Testamos todos os classificadores com todas as features
    if DIPLAY_ALL_FEATURES:
        print('---------------------------')
        print('**** ALL ****')
        print('---------------------------')
        # Tanto os classificadores como o sistema de voto
        # Avaliação ao conjunto de dados inteiro apenas com feature scaling
        cg.full_model_evaluation(energy_functions, X_sc, y, 'all_classifiers', N_FOLDS, open_file=True, display_metrics=False)
        cg.full_voting_evaluation(energy_functions, classifiers, X_sc, y, E_BEST_SCORE, 'all_classifiers', N_FOLDS, True, True, True, open_file=True, display_metrics=True)
        
        
    # Testamos os clacificadores com um conjunto PCA - Principal component Analisis
    if DIPLAY_KPCA_FEATURES:
        print('---------------------------')
        print('**** KPCA ****')
        print('---------------------------')
        # Escolhemos o componente como 3 para os podermos visualizar
        X_pca = cg.get_kpca_components(X_sc,3)
        
        cg.full_model_evaluation(energy_functions, X_pca, y, 'kpca_classifiers', N_FOLDS, open_file=True, display_metrics=False)
        cg.full_voting_evaluation(energy_functions, classifiers, X_pca, y, E_BEST_SCORE, 'kpca_classifiers', N_FOLDS, True, True, True, open_file=True, display_metrics=True)
        #cg.binary_3d_plot(X_pca, y_test, y_pred)
        
     
    # Testamos os clacificadores com um conjunto LDA - Linear Discriminant Analysis
    if DIPLAY_LDA_FEATURES:
        print('---------------------------')
        print('**** LDA ****')
        print('---------------------------')
        # Escolhemos o componente como 3 para os podermos visualizar
        X_lda = cg.get_lda_components(X_sc, y,3)
        
        cg.full_model_evaluation(energy_functions, X_lda, y, 'lda_classifiers', N_FOLDS, open_file=True, display_metrics=True)
        cg.full_voting_evaluation(energy_functions, classifiers, X_lda, y, E_BEST_SCORE, 'lda_classifiers', N_FOLDS, True, True, True, open_file=True, display_metrics=True)
        #cg.binary_3d_plot(X_pca, y_test, y_pred)
        
        
    # Testamos os clacificadores com um conjunto Select K Atributes
    if DIPLAY_SELK_FEATURES:
        print('---------------------------')
        print('**** K Best ****')
        print('---------------------------')
        
        # Calculamos o kbest antes de fazermos a nova feature scaling dado que a função chi2 não admite valore negativos
        X_kbest, scores2 = cg.get_chi2_kbest_components(X, y, 5)
        X_sc = cg.feature_scaling(X_kbest)
        
        cg.full_model_evaluation(energy_functions, X_sc, y, 'kbest_classifiers', N_FOLDS, open_file=True, display_metrics=True)
        cg.full_voting_evaluation(energy_functions, classifiers, X_sc, y, E_BEST_SCORE, 'kbest_classifiers', N_FOLDS, True, True, True, open_file=True, display_metrics=True)
        