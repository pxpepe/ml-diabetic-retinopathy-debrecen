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
    from sklearn.metrics import accuracy_score
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Importamos o conjunto de dados
    X, y = get_messidor_dataset()

    # Escalamos os atributos
    X = cg.feature_scaling(X)
    
    # O melhor resultado foi obtido em Gaussian Process tendo em conta todos os atributos
    classifier_file = 'all_classifiers/gaussian_process_accuracy.pkl'
    classifier = joblib.load(classifier_file) 
    
    # Dividimos o dataset e guardamos 20% para testes
    X, X_gtest, y, y_gtest = cg.split_dataset(X, y, test_size=.2)
    
    acc_score_curve = []

    for i in np.arange(0.95, 0.14, -0.1):
        
        X_train, X_test, y_train, y_test = cg.split_dataset(X, y, test_size=i)
        
        classifier.fit(X_train, y_train)
    
        y_train_pred = classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)
        y_gtest_pred = classifier.predict(X_gtest)
    
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy =  accuracy_score(y_test, y_test_pred)
        gtest_accuracy =  accuracy_score(y_gtest, y_gtest_pred)
        
        
        acc_score_curve.append({'test_size':i, 'train_acc': train_accuracy, 'test_acc':test_accuracy, 'gtest_acc':gtest_accuracy})
        
    
    test_size = [asc_act['test_size'] for asc_act in acc_score_curve]
    train_size = np.ones(len(test_size))-test_size
    train_acc = [asc_act['train_acc'] for asc_act in acc_score_curve]
    test_acc = [asc_act['test_acc'] for asc_act in acc_score_curve]
    gtest_acc = [asc_act['gtest_acc'] for asc_act in acc_score_curve]
    
    plt.figure()
    lw = 2
    plt.plot(train_size, train_acc, color='darkorange',lw=2,label='Subconjunto de treino')
    plt.plot(train_size, test_acc, color='navy',lw=2,label='Subconjunto de teste')
    plt.plot(train_size, gtest_acc, color='g',lw=2,label='Conjunto de teste')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tamanho em % do cojunto de treino')
    plt.ylabel('Exatidão')
    plt.title('Curva de aprendizagem em função da dimensão da amostra')
    plt.legend(loc="lower right")
    plt.show()
    