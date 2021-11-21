import pandas
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def ChangeValues(tabela):
    tabela["sex"].replace({0: 'female', 1: 'male'}, inplace=True)
    tabela["cpt"].replace({1: 'atypical angina', 2: 'typical angina', 3: 'asymptomatic', 4: 'nonanginal pain'}, inplace=True)
    tabela["fbs"].replace({0: '< 120 mg/dl', 1: '> 120 mg/dl'}, inplace=True)
    tabela["res"].replace({0: 'normal', 1: 'having ST-T', 2: 'hypertrophy'}, inplace=True)
    tabela["eia"].replace({0: 'no', 1: 'yes'}, inplace=True)
    tabela["pes"].replace({1: 'up sloping', 2: 'flat', 3: 'down sloping'}, inplace=True)
    tabela["pes"].replace({1: 'up sloping', 2: 'flat', 3: 'down sloping'}, inplace=True)
    tabela["tha"].replace({3: 'normal', 6: 'fixed defect', 7: 'reversible defect'}, inplace=True)
    tabela["target"].replace({2: 1, 3: 1, 4: 1}, inplace=True)
    return tabela


def CreateTrainAndTestSamples(tabela):
    train, test = train_test_split(tabela, test_size=0.2, random_state=0)
    Y_train = train['target']
    Y_test = test['target']
    X_train = train.drop(['target'], axis=1)
    X_test = test.drop(['target'], axis=1)
    return X_train, Y_train, X_test, Y_test


def CalculateAccAndErrors(Y_test, Y_predict):
    acc = accuracy_score(Y_test, Y_predict)
    mae = mean_absolute_error(Y_test, Y_predict)
    rmse = math.sqrt(mean_squared_error(Y_test, Y_predict))
    return round(acc, 3), round(mae, 3), round(rmse, 3)


def CalculateSensAndSpec(Y_test, Y_predicted):
    TN, FP, FN, TP = confusion_matrix(Y_test, Y_predicted).ravel()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    return round(sensitivity, 3), round(specificity, 3)


def plotRoc(fpr, tpr, classifier):
    a = 1


def allTheWork(X_train, Y_train, X_test, Y_test, model):
    model.fit(X_train, Y_train)
    Y_predicted = model.predict(X_test)
    acc, mae, rmse = CalculateAccAndErrors(Y_test, Y_predicted)
    print(f'acc:{acc}, mae:{mae}, rmse:{rmse}')
    sensitivity, specificity = CalculateSensAndSpec(Y_test, Y_predicted)
    print(f'sensitivity:{sensitivity}, specificity:{specificity}')
    Y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_prob)
    plotRoc(fpr, tpr, "")
    auc_result = round(auc(fpr, tpr), 3)
    print(f'auc:{auc_result}')
    print('------------------')


def DecisionTree(X_train, Y_train, X_test, Y_test):
    print('Decision tree')
    model = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
    allTheWork(X_train, Y_train, X_test, Y_test, model)


def RandomForest(X_train, Y_train, X_test, Y_test):
    print('Random Forest')
    model = RandomForestClassifier(max_depth=4)
    allTheWork(X_train, Y_train, X_test, Y_test, model)


def SVM(X_train, Y_train, X_test, Y_test):
    print('SVM')
    model = svm.SVC(kernel='linear', random_state=0, probability=True)
    allTheWork(X_train, Y_train, X_test, Y_test, model)


def KNN(X_train, Y_train, X_test, Y_test):
    print('KNN')
    model = KNeighborsClassifier(n_neighbors=20)
    allTheWork(X_train, Y_train, X_test, Y_test, model)


def NeuralNetworks(X_train, Y_train, X_test, Y_test):
    print('Neural Networks')
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25,), random_state=0)
    allTheWork(X_train, Y_train, X_test, Y_test, model)


if __name__ == '__main__':
    table = pandas.read_csv("data.csv")
    table = ChangeValues(table)
    dummy_table = pandas.get_dummies(table)
    x_train, y_train, x_test, y_test = CreateTrainAndTestSamples(dummy_table)
    DecisionTree(x_train, y_train, x_test, y_test)
    RandomForest(x_train, y_train, x_test, y_test)
    SVM(x_train, y_train, x_test, y_test)
    KNN(x_train, y_train, x_test, y_test)
    NeuralNetworks(x_train, y_train, x_test, y_test)
