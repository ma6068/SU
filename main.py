import pandas
import math
import os
import matplotlib.pyplot as plt
from subprocess import call
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from IPython.display import Image

f = open("results.txt", "w")


def ChangeValues(tabela):
    tabela["sex"].replace({0: 'female', 1: 'male'}, inplace=True)
    tabela["cpt"].replace({1: 'atypical angina', 2: 'typical angina', 3: 'asymptomatic', 4: 'nonanginal pain'},
                          inplace=True)
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


def PlotRocCurve(fpr, tpr, classifier):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    plt.title(f'{classifier} - ROC curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.savefig(f'images/Roc_{classifier}.png')


def AllTheWork(X_train, Y_train, X_test, Y_test, model, classifier):
    model.fit(X_train, Y_train)
    Y_predicted = model.predict(X_test)
    acc, mae, rmse = CalculateAccAndErrors(Y_test, Y_predicted)
    f.write(f'acc:{acc}, mae:{mae}, rmse:{rmse}\n')
    sensitivity, specificity = CalculateSensAndSpec(Y_test, Y_predicted)
    f.write(f'sensitivity:{sensitivity}, specificity:{specificity}\n')
    Y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_prob)
    PlotRocCurve(fpr, tpr, classifier)
    auc_result = round(auc(fpr, tpr), 3)
    f.write(f'auc:{auc_result}\n')
    f.write('--------------------\n')


def PlotTree(model, X_train, Y_train, filename):
    Y_train_str = Y_train.astype('str')
    Y_train_str[Y_train_str == '0'] = 'healthy'
    Y_train_str[Y_train_str == '1'] = 'sick'
    class_names = Y_train_str.values
    feature_names = [i for i in X_train.columns]
    _ = export_graphviz(model, out_file='dot/decision_tree.dot',
                        class_names=class_names,
                        feature_names=feature_names,
                        rounded=True, proportion=True,
                        label='root', precision=2,
                        filled=True)
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
    call(['dot', '-Tpng', 'dot/decision_tree.dot', '-o', 'images/' + filename, '-Gdpi=600'])
    Image(filename='images/' + filename)


def DecisionTree(X_train, Y_train, X_test, Y_test):
    f.write('Decision tree\n')
    model = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
    AllTheWork(X_train, Y_train, X_test, Y_test, model, "Decision_tree")
    PlotTree(model, X_train, Y_train, "decision_tree.png")


def RandomForest(X_train, Y_train, X_test, Y_test):
    f.write('Random Forest\n')
    model = RandomForestClassifier(max_depth=4)
    AllTheWork(X_train, Y_train, X_test, Y_test, model, "Random_forest")


def SVM(X_train, Y_train, X_test, Y_test):
    f.write('SVM\n')
    model = svm.SVC(kernel='linear', random_state=0, probability=True)
    AllTheWork(X_train, Y_train, X_test, Y_test, model, "SVM")


def KNN(X_train, Y_train, X_test, Y_test):
    f.write('KNN\n')
    model = KNeighborsClassifier(n_neighbors=20)
    AllTheWork(X_train, Y_train, X_test, Y_test, model, "KNN")


def NeuralNetworks(X_train, Y_train, X_test, Y_test):
    f.write('Neural Networks\n')
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25,), random_state=0)
    AllTheWork(X_train, Y_train, X_test, Y_test, model, "Neutral_Networks")


if __name__ == '__main__':
    table = pandas.read_csv("data/data.csv")
    table = ChangeValues(table)
    dummy_table = pandas.get_dummies(table)
    x_train, y_train, x_test, y_test = CreateTrainAndTestSamples(dummy_table)
    DecisionTree(x_train, y_train, x_test, y_test)
    RandomForest(x_train, y_train, x_test, y_test)
    SVM(x_train, y_train, x_test, y_test)
    KNN(x_train, y_train, x_test, y_test)
    NeuralNetworks(x_train, y_train, x_test, y_test)
    f.close()
