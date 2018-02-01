def specificity(estimator, X, y_test):
    from sklearn.metrics import confusion_matrix
    y_pred = estimator.predict(X)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    # SPC = TN/N = TN/(TN+FP)
    return tn/(tn+fp)

# Implementamos uma grid search de modo a encontrar os melhores parâmetros para o classificados
def grid_search(classifier, parameters, scoring, X_train, y_train, n_folds):
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = scoring, cv = n_folds, n_jobs = -1)
    grid_search = grid_search.fit(X_train, y_train)
    print('Best Params: ')
    print(grid_search.best_params_)
    print('-------------------------------')
    return grid_search.best_params_
    
# Mostramos as métricas do classificador
def kfold_algorithm_metrics(classifier, X_train, y_train, n_folds):
    # Applying k-Fold Cross Validation
    from sklearn.model_selection import cross_validate
    
    score_evaluate = {'accuracy':'accuracy','recall':'recall','precision':'precision','specificity':specificity,'f1':'f1', 'roc_auc':'roc_auc'}
    cv_results = cross_validate(estimator = classifier, X = X_train, y = y_train, cv = n_folds, scoring=score_evaluate)
    
    print('Metrics: ')
    print('Accuracy: ' + str(cv_results['test_accuracy'].mean()))
    print('Recall: ' + str(cv_results['test_recall'].mean()))
    print('Precion: ' + str(cv_results['test_precision'].mean()))
    print('Specificity: ' + str(cv_results['test_specificity'].mean()))
    print('F-Measure: ' + str(cv_results['test_f1'].mean()))
    print('ROC: ' + str(cv_results['test_roc_auc'].mean()))
    print('-------------------------------')

def test_classifier_energy_function(classifier_class, energy_functions, parameters, classifier_name, X_train, y_train, folder_name, n_folds, open_file = True, display_metrics = False):
    from sklearn.externals import joblib
    import os.path
    for en_fn in energy_functions:
        print('Energy Function: '  + en_fn['title'])
        print('-------------------------------')
        classifier_file = folder_name + '/'+classifier_name+'_'+str( en_fn['scoring'])+ '.pkl'
        # Se já optimizamos o classificador, podemos saltar esse passo
        if os.path.isfile(classifier_file) and open_file:
            classifier = joblib.load(classifier_file) 
        else:
            if parameters is not None:
                best_parameters = grid_search(classifier_class(), parameters, en_fn['scoring'], X_train, y_train, n_folds)
                classifier=classifier_class(**best_parameters)
            else:
                classifier=classifier_class()
            joblib.dump(classifier, classifier_file)
        
        if display_metrics:
            kfold_algorithm_metrics(classifier, X_train, y_train, n_folds)
        
def full_model_evaluation(energy_functions, X_train, y_train, folder_name, n_folds, open_file = True, display_metrics = False):
    import numpy as np
    print('-------------------------------')
    # DecisionTree
    from sklearn.tree import DecisionTreeClassifier
    print('Classifier: DecisionTreeClassifier')
    print('-------------------------------')
    # Parametros para o teste da decision tree
    parameters = [{'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'min_samples_split': np.arange(2,12), 'min_samples_leaf':np.arange(1,12), 
               'min_weight_fraction_leaf': np.arange(.0,.6,.1), 'max_features': ['auto', 'sqrt','log2'], 'random_state':[0], 'min_impurity_decrease': np.arange(0,6), 'presort':[True, False]}]
    test_classifier_energy_function(DecisionTreeClassifier, energy_functions, parameters, 'decision_tree', X_train, y_train, folder_name, n_folds, open_file, display_metrics)

    
    # K-NN 
    from sklearn.neighbors import KNeighborsClassifier
    print('Classifier: KNN')
    print('-------------------------------')
    # Faltam testar algumas métricas ('wminkowski', 'seuclidean', 'mahalanobis'), mas estas requerem mais alguns parametros de entrada
    parameters = [{'n_neighbors':np.arange(3,8), 'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                   'weights':['uniform', 'distance'], 'leaf_size': np.arange(20,50,5), 'n_jobs':[-1]}]
    test_classifier_energy_function(KNeighborsClassifier, energy_functions, parameters, 'knn', X_train, y_train, folder_name, n_folds, open_file, display_metrics)
    
    # AdaBoost
    from sklearn.ensemble import AdaBoostClassifier
    print('Classifier: AdaBoost')
    print('-------------------------------')
    parameters = [{'random_state':[0],'algorithm':['SAMME', 'SAMME.R'], 'learning_rate':np.arange(1,10), 'n_estimators':np.arange(20,70,5)}]
    test_classifier_energy_function(AdaBoostClassifier, energy_functions, parameters, 'adaboost', X_train, y_train, folder_name, n_folds, open_file, display_metrics)
    
    
    # Multilayer Perceptron
    from sklearn.neural_network import MLPClassifier
    print('Classifier: Multilayer Perceptron')
    print('-------------------------------')
    # Como é mais intensio para o processador, neste caso primeiro vemos qual o solver vs activation
    #parameters = [{'activation':['identity', 'logistic', 'tanh', 'relu'], 'random_state':[0], 'solver': ['lbfgs', 'sgd', 'adam']}]
    # Nos testes anteriores o que melhor funcionou foi o adam e o relu, como tal, vamos afinar apenas para estes
    # Vamos refinar esta pesquisa desta vez para detetar as layers e o learning rate
    #parameters = [{'hidden_layer_sizes': [(5,2),(10,2),(20,4),(10,4)], 'activation':['relu'], 'random_state':[0], 'solver': ['adam'], 
    #                                      'learning_rate' : ['constant', 'invscaling', 'adaptive'], 'max_iter':[500]}]
    # Desta segunda iteração podemos manter as layers 20,4 e 10,4 e o learning rate como constant
    # Nesta última iteração vamos otimizar o alpha e o epsilon
    parameters = [{'hidden_layer_sizes': [(20,4),(10,4)], 'activation':['relu'], 'random_state':[0], 'solver': ['adam'], 
                                          'learning_rate' : ['constant'], 'max_iter':[500],'alpha':[0.001, 0.0001, 0.00001], 'epsilon': [1e-6,1e-7,1e-8]}]    
    test_classifier_energy_function(MLPClassifier, energy_functions, parameters, 'multilayer_perceptron', X_train, y_train, folder_name, n_folds, open_file, display_metrics)
    
    
    # Naive Bayes
    from sklearn.naive_bayes import GaussianNB
    print('Classifier: Naive Bayes')
    print('-------------------------------')
    test_classifier_energy_function(GaussianNB, energy_functions, None, 'naive_bayes', X_train, y_train, folder_name, n_folds, open_file, display_metrics)
    
    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    print('Classifier: Random Forest')
    print('-------------------------------')
    parameters = [{ 'n_jobs':[-1], 'random_state':[0], 'n_estimators':[10,100,1000], 'criterion': ['gini', 'entropy'], 'max_features':['auto'], 'min_samples_leaf':np.arange(1,12)}]
    test_classifier_energy_function(RandomForestClassifier, energy_functions, parameters, 'random_forest', X_train, y_train, folder_name, n_folds, open_file, display_metrics)
      
    # SVM
    from sklearn.svm import SVC
    print('Classifier: SVM')
    print('-------------------------------')
    parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear','poly'], 'probability':[True]},
                   {'C': [1, 10, 100, 1000], 'kernel': ['rbf','sigmoid'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'probability':[True]}]
    test_classifier_energy_function(SVC, energy_functions, parameters, 'svm', X_train, y_train, folder_name, n_folds, open_file, display_metrics)
    
    # GaussianProcessClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    print('Classifier: Gaussian Process Classifier')
    print('-------------------------------')
    # Os parametros a testar reprensentam isotropic e anisotropic, respetivamente
    # O anisotropico deve ter o mesmo número de dimensões que de parametros passados ao rbf
    parameters = [{'kernel':[1.0 * RBF([1.0]),1.0 * RBF(np.ones(len(X_train[0])))]}]
    test_classifier_energy_function(GaussianProcessClassifier, energy_functions, parameters, 'gaussian_process', X_train, y_train, folder_name, n_folds, open_file, display_metrics)
    
    
def get_energy_functions():
    return [{'title':'Sensitivity', 'scoring':'recall'},
                        {'title':'Accuracy', 'scoring':'accuracy'},
                        {'title':'F-Measure', 'scoring':'f1'}]

def split_dataset(X, y, test_size=.2):
    #X_train, X_test, y_train, y_test = split_dataset(X, y)
    from sklearn.model_selection  import train_test_split
    return train_test_split(X, y, test_size = test_size, random_state = 0)

def feature_scaling(X, X_test=None):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_fc = sc.fit_transform(X)
    if X_test is not None:
        X_fc = [X_fc,sc.transform(X_test)]
    
    return X_fc

def get_classifiers():
    return [{'title':'Decision Tree', 'name':'decision_tree'},
                   {'title':'KNN', 'name':'knn'},
                   {'title':'AdaBoost', 'name':'adaboost'},
                   {'title':'Multilayer Perceptron', 'name':'multilayer_perceptron'},
                   {'title':'Naive Bayes', 'name':'naive_bayes'},
                   {'title':'Random Forest', 'name':'random_forest'},
                   {'title':'SVM', 'name':'svm'},
                   {'title':'Gaussian Process Classifier', 'name':'gaussian_process'}]    

def score_classifier(classifier, X_train, y_train, scoring, n_folds):
    from sklearn.model_selection import cross_val_score    
    cv_score = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = n_folds, scoring=scoring)
    return cv_score.mean()
    
def generate_estimators_tuple(estimators):
    esti_tuple = []
    for estimator in estimators:
        esti_tuple.append((estimator['classifier_name'],estimator['classifier']))
    return esti_tuple

# Começamos com todos os classificadores e vamos eliminando os que têm pior classificação
# Continuamos a eliminar até que não se encontre melhoria no scoring do sistema de votação
# Utilizamos o scoring ebest para decidir se continuamos na procura ou não,
# Utilizamos o scoring para saber qual o melhor algoritmo tendo em conta a função de energia em estudo
def estimators_backward_search(estimators, X_train, y_train, ebest, n_folds):
    from sklearn.ensemble import VotingClassifier
    esti_clone = estimators[:]
    esti_act = esti_clone[:]
    
    eclf = VotingClassifier(estimators=generate_estimators_tuple(esti_clone), voting='soft')
    best_score = score_classifier(eclf, X_train, y_train, ebest, n_folds)
    score_act = best_score
    
    while score_act>=best_score:  
        esti_clone = esti_act[:]
        best_score = score_act
        rm_esti = min(esti_act, key=lambda e : e['scoring'])
        esti_act.remove(rm_esti)
        if len(esti_act)>0:
            eclf = VotingClassifier(estimators=generate_estimators_tuple(esti_act), voting='soft')
            score_act = score_classifier(eclf, X_train, y_train, ebest, n_folds)
        else:
            break
        
    return esti_clone


# Começamos com o melhor classificador e vamos adicionando os seguintes melhores até que não melhoremos a classificação
# Utilizamos o scoring ebest para decidir se continuamos na procura ou não,
# Utilizamos o scoring para saber qual o melhor algoritmo tendo em conta a função de energia em estudo
def estimators_forward_search(estimators, X_train, y_train, ebest, n_folds):
    from sklearn.ensemble import VotingClassifier
    esti_parcial = estimators[:]
    # Selecionamos o algoritmo a adicionar
    # E eliminamos da lista de algoritmos disponiveis uma vez que já está adicionado
    add_max = max(esti_parcial, key=lambda e : e['scoring'])
    esti_parcial.remove(add_max)
    esti_act = [add_max]
    esti_clone = [add_max]
    
    eclf = VotingClassifier(estimators=generate_estimators_tuple(esti_act), voting='soft')
    best_score = score_classifier(eclf, X_train, y_train, ebest, n_folds)
    score_act = best_score
    
    while score_act>=best_score:  
        esti_clone = esti_act[:]
        best_score = score_act
        
        add_max = max(esti_parcial, key=lambda e : e['scoring'])
        esti_parcial.remove(add_max)
        esti_act.append(add_max)
        
        eclf = VotingClassifier(estimators=generate_estimators_tuple(esti_act), voting='soft')
        score_act = score_classifier(eclf, X_train, y_train, ebest, n_folds)
        
    return esti_clone    

def get_estimators_efn(classifiers, en_fn, estimators, folder_name, X_train, y_train, n_folds):
    from sklearn.externals import joblib
    # Apenas inicilizamos os estimadores a primeira vez que chamamos esta função
    if len(estimators)==0:
        for classifier_info in classifiers:
                classifier_name = classifier_info['name']+'_'+en_fn['scoring']
                classifier_file = folder_name+'/'+classifier_name+'.pkl'
                classifier = joblib.load(classifier_file)
                estimators.append({'classifier_name':classifier_name,'classifier':classifier,
                                   'scoring':score_classifier(classifier, X_train, y_train, en_fn['scoring'], n_folds)})
    

def full_voting_evaluation(energy_functions, classifiers, X_train, y_train, e_best, folder_name, n_folds, voting_all, voting_back, voting_forw, open_file=True, display_metrics = False):
    
    from sklearn.ensemble import VotingClassifier
    import os.path
    from sklearn.externals import joblib
    
    for en_fn in energy_functions:
        print('-------------------------------')
        print('Energy Function: '  + en_fn['title'])
        print('-------------------------------')
        
        # Inicializamos os estimators dado que se trata de uma nova funão de energia
        estimators = []

        if voting_all:
            print('-------------------------------')
            print('Classifier: Voting All')
            print('-------------------------------')
            
            # VOTING para todo o conjunto de estimators            
            # A opção soft é preferivel no que toca à utilização de classificadores bem calibrados
            classifier_file = folder_name + '/voting_all_'+str( en_fn['scoring'])+ '.pkl'
            # Se já optimizamos o classificador, podemos saltar esse passo
            if os.path.isfile(classifier_file) and open_file:
                eclf = joblib.load(classifier_file) 
            else:
                get_estimators_efn(classifiers, en_fn, estimators, folder_name, X_train, y_train, n_folds)
                eclf = VotingClassifier(estimators=generate_estimators_tuple(estimators), voting='soft')
                joblib.dump(eclf, classifier_file)
            if display_metrics:
                kfold_algorithm_metrics(eclf, X_train, y_train, n_folds)
    
        if voting_back:
            print('-------------------------------')
            print('Classifier: Voting Backward Search')
            print('-------------------------------')
            classifier_file = folder_name + '/voting_back_'+str( en_fn['scoring'])+ '.pkl'
            # Se já optimizamos o classificador, podemos saltar esse passo
            if os.path.isfile(classifier_file) and open_file:
                eclf = joblib.load(classifier_file) 
            else:
                get_estimators_efn(classifiers, en_fn, estimators, folder_name, X_train, y_train, n_folds)
                backward_estimators = estimators_backward_search(estimators, X_train, y_train, e_best, n_folds)
                print('Classifiers')
                print('-------------------------------')
                print([d['classifier_name'] for d in backward_estimators])
                print('-------------------------------')
                eclf = VotingClassifier(estimators=generate_estimators_tuple(backward_estimators), voting='soft')
                joblib.dump(eclf, classifier_file)
            if display_metrics:
                kfold_algorithm_metrics(eclf, X_train, y_train, n_folds)
            
            
        if voting_forw:
            print('-------------------------------')
            print('Classifier: Voting Forward Search')
            print('-------------------------------')
            classifier_file = folder_name + '/voting_forw_'+str( en_fn['scoring'])+ '.pkl'
            # Se já optimizamos o classificador, podemos saltar esse passo
            if os.path.isfile(classifier_file) and open_file:
                eclf = joblib.load(classifier_file) 
            else:
                get_estimators_efn(classifiers, en_fn, estimators, folder_name, X_train, y_train, n_folds)
                forward_estimators = estimators_forward_search(estimators, X_train, y_train, e_best, n_folds)
                print('Classifiers')
                print('-------------------------------')
                print([d['classifier_name'] for d in forward_estimators])
                print('-------------------------------')
                eclf = VotingClassifier(estimators=generate_estimators_tuple(forward_estimators), voting='soft')
                joblib.dump(eclf, classifier_file)
            if display_metrics:
                kfold_algorithm_metrics(eclf, X_train, y_train, n_folds)
            
def binary_3d_plot(X, y_train, y_pred):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    y_hit = y_pred == y_train
    y_pred_hit = y_pred[y_hit==True]
    y_pred_nhit = y_pred[y_hit==False]
    
    X_hit = X[y_hit==True]
    X_nhit = X[y_hit==False]
    
    # Pintamos o acerto a verde, e a classe 1 com o triangulo
    ax.scatter(X_hit[y_pred_hit==1,0], X_hit[y_pred_hit==1,1], X_hit[y_pred_hit==1,2], c='g', marker='^')
    
    # Os que foram mal classificados pintamos a vermelho
    ax.scatter(X_nhit[y_pred_nhit==1,0], X_nhit[y_pred_nhit==1,1], X_nhit[y_pred_nhit==1,2], c='r', marker='^')
    
    # Pintamos o acerto a verde, e a classe 0 com o circulo
    ax.scatter(X_hit[y_pred_hit==0,0], X_hit[y_pred_hit==0,1], X_hit[y_pred_hit==0,2], c='b', marker='o')
    
    # Os que foram mal classificados pintamos a vermelho
    ax.scatter(X_nhit[y_pred_nhit==0,0], X_nhit[y_pred_nhit==0,1], X_nhit[y_pred_nhit==0,2], c='r', marker='^')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()
    
def get_pca_components(X_train, n_components):
    # Applying PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X_train)

def get_kpca_components(X_train, n_components):
    # Applying kPCA
    from sklearn.decomposition import KernelPCA
    kpca = KernelPCA(n_components=n_components, kernel = 'rbf')
    return kpca.fit_transform(X_train)

def get_lda_components(X_train, y_train, n_components):
    # Applying LDA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    lda = LDA(n_components = n_components)
    return lda.fit_transform(X_train, y_train)

def get_chi2_kbest_components(X_train, y_train, n_components):
    # Applying kBest with chi2
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    kbest = SelectKBest(chi2, k=n_components).fit(X_train, y_train)
    X_kbest = kbest.transform(X_train)
    scores = kbest.scores_
    
    return [X_kbest, scores]