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
    from sklearn.externals import joblib
    
    DIPLAY_AUC_SDIM = False
    DIPLAY_ROC_CURVE = False
    
    # Importamos o conjunto de dados
    X, y = get_messidor_dataset()

    # Escalamos os atributos
    X = cg.feature_scaling(X)
    
    # O melhor resultado foi obtido em Gaussian Process tendo em conta todos os atributos
    classifier_file = 'all_classifiers/gaussian_process_accuracy.pkl'
    classifier = joblib.load(classifier_file) 
    
    # Dividimos o dataset e guardamos 20% para testes
    X, X_gtest, y, y_gtest = cg.split_dataset(X, y, test_size=.2)
    
    if DIPLAY_AUC_SDIM:
        cg.acc_sample_dim(classifier, X, X_gtest, y, y_gtest)
    
    if DIPLAY_ROC_CURVE:
        classifier.fit(X, y)        
        y_probas = classifier.predict_proba(X_gtest)
        cg.print_binary_roc_curve(y_gtest, y_probas)