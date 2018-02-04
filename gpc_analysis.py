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
    
    DIPLAY_AUC_SDIM = True
    DIPLAY_ROC_CURVE = True
    DISPLAY_2KBEST = True
    DISPLAY_3KBEST = True
    DISPLAY_MA_VS_NS = True
    DISPLAY_PRBA_HIST = True
    
    # Importamos o conjunto de dados
    X, y = get_messidor_dataset()

    # Escalamos os atributos
    X_sc = cg.feature_scaling(X)
    
    # O melhor resultado foi obtido em Gaussian Process tendo em conta todos os atributos
    classifier_file = 'all_classifiers/gaussian_process_accuracy.pkl'
    classifier = joblib.load(classifier_file) 
    
    # Dividimos o dataset e guardamos 20% para testes
    X_train, X_test, y_train, y_test = cg.split_dataset(X_sc, y, test_size=.2)
    
    if DIPLAY_AUC_SDIM:
        cg.acc_sample_dim(classifier, X_train, X_test, y_train, y_test)
    
    if DIPLAY_ROC_CURVE:
        classifier.fit(X_train, y_train)        
        y_probas = classifier.predict_proba(X_test)
        cg.print_binary_roc_curve(y_test, y_probas)
        
    if DISPLAY_2KBEST:
        
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF
        classifier = GaussianProcessClassifier(kernel =   1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)))
        
        X_kbest, scores = cg.get_chi2_kbest_components(X, y, 2)
        X_sc = cg.feature_scaling(X_kbest)
        classifier.fit(X_sc, y)       

        cg.print_classifier_2d_analysis(classifier, X_sc, y)
        
    if DISPLAY_3KBEST:
        
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF
        classifier = GaussianProcessClassifier(kernel =   1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)))
        
        X_kbest, scores = cg.get_chi2_kbest_components(X, y, 3)
        X_sc = cg.feature_scaling(X_kbest)
        classifier.fit(X_sc, y)       

        cg.binary_3d_plot(X_kbest, y, classifier.predict(X_sc))


    if DISPLAY_MA_VS_NS:

        import numpy as np

        x_ma = X_sc[:,2:8]
        x_nex = X_sc[:,8:16]
    
        X_ma_pca = cg.get_pca_components(x_ma,y, 1)
        X_nex_pca = cg.get_pca_components(x_nex,y, 1)
        
        X_ma_nex = np.zeros((x_ma.shape[0],2))
        
        X_ma_nex[:,0] = X_ma_pca[:,0] 
        X_ma_nex[:,1] = X_nex_pca[:,0] 
        
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF
        classifier = GaussianProcessClassifier(kernel =   1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)))
        
        classifier.fit(X_ma_nex, y)       
    
        cg.print_classifier_2d_analysis(classifier, X_ma_nex, y, x_label = 'PCA MAs', y_label= 'PCA exsudados')

    if DISPLAY_PRBA_HIST:
        classifier.fit(X_train, y_train)        
        y_probas = classifier.predict_proba(X_test)
        cg.print_proba_histogram(y_probas)