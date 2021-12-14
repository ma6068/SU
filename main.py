import pandas
import math
import os
import random
import statistics
import matplotlib.pyplot as plt
from subprocess import call
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


def PlotRocCurve(fpr, tpr, classifier, index):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    plt.title(f'{classifier} - ROC curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.savefig(f'images/Roc_{classifier}_{index}.png')


def CalculateStandardDeviation(Y_test, Y_predicted):
    razliki = []
    yt = Y_test.values.tolist()
    yp = list(Y_predicted)
    for a in range(len(yt)):
        if yt[a] == yp[a]:
            razliki.append(1)
        else:
            razliki.append(0)
    sd = statistics.stdev(razliki)
    sd = sd / math.sqrt(len(razliki))
    return round(sd, 3)


def AllTheWork(X_train, Y_train, X_test, Y_test, model, classifier, index):
    f.write(f'Iteracija: {index}\n')
    model.fit(X_train, Y_train)
    Y_predicted = model.predict(X_test)
    acc, mae, rmse = CalculateAccAndErrors(Y_test, Y_predicted)
    sd = CalculateStandardDeviation(Y_test, Y_predicted)
    f.write(f'acc:{acc}, sd:{sd}, mae:{mae}, rmse:{rmse}\n')
    sensitivity, specificity = CalculateSensAndSpec(Y_test, Y_predicted)
    f.write(f'sensitivity:{sensitivity}, specificity:{specificity}\n')
    Y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_prob)
    PlotRocCurve(fpr, tpr, classifier, index)
    auc_result = round(auc(fpr, tpr), 3)
    f.write(f'auc:{auc_result}\n')
    f.write('--------------------\n')
    acc_table.append((acc, classifier))
    sd_table.append((sd, classifier))
    sensitivity_table.append((sensitivity, classifier))
    specificity_table.append((specificity, classifier))
    auc_table.append((auc_result, classifier))


def PlotTree(model, X_train, Y_train, filename, index):
    Y_train_str = Y_train.astype('str')
    Y_train_str[Y_train_str == '0'] = 'healthy'
    Y_train_str[Y_train_str == '1'] = 'sick'
    class_names = Y_train_str.values
    feature_names = [i for i in X_train.columns]
    _ = export_graphviz(model, out_file='dot/' + filename + '.dot',
                        class_names=class_names,
                        feature_names=feature_names,
                        rounded=True, proportion=True,
                        label='root', precision=2,
                        filled=True)
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
    call(['dot', '-Tpng', 'dot/decision_tree.dot', '-o', 'images/' + filename + '_' + str(index) + '.png', '-Gdpi=600'])
    Image(filename='images/' + filename + '_' + str(index) + '.png')


def DecisionTree(X_train, Y_train, X_test, Y_test, index):
    f.write('Decision tree\n')
    model = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
    AllTheWork(X_train, Y_train, X_test, Y_test, model, "Decision_tree", index)
    PlotTree(model, X_train, Y_train, "decision_tree", index)


def RandomForest(X_train, Y_train, X_test, Y_test, index):
    f.write('Random Forest\n')
    model = RandomForestClassifier(max_depth=4)
    AllTheWork(X_train, Y_train, X_test, Y_test, model, "Random_forest", index)
    estimator = model.estimators_[1]
    PlotTree(estimator, X_train, Y_train, "random_forest_tree", index)


def SVM(X_train, Y_train, X_test, Y_test, index):
    f.write('SVM\n')
    model = svm.SVC(kernel='linear', random_state=0, probability=True)
    AllTheWork(X_train, Y_train, X_test, Y_test, model, "SVM", index)


def KNN(X_train, Y_train, X_test, Y_test, index):
    f.write('KNN\n')
    model = KNeighborsClassifier(n_neighbors=20)
    AllTheWork(X_train, Y_train, X_test, Y_test, model, "KNN", index)


def NeuralNetworks(X_train, Y_train, X_test, Y_test, index):
    f.write('Neural Networks\n')
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25,), random_state=0)
    AllTheWork(X_train, Y_train, X_test, Y_test, model, "Neutral_Networks", index)


def CreateTrainAndTest(tabela, index, brTestni, redovi):
    X_train = tabela.copy()
    X_test = tabela.copy()
    for a in range(len(tabela)):
        # spaga vo testni => izbrisi go od ucni
        if index * brTestni <= a < index * brTestni + brTestni:
            X_train.drop(labels=redovi[a], inplace=True)
        # spaga vo ucni => izbrisi go od testni
        else:
            X_test.drop(labels=redovi[a], inplace=True)
    Y_train = X_train['target'].copy()
    Y_test = X_test['target'].copy()
    X_train.drop(['target'], axis=1, inplace=True)
    X_test.drop(['target'], axis=1, inplace=True)
    return X_train, Y_train, X_test, Y_test


def IzracunajPovprecje(tabela):
    dt = 0
    rf = 0
    sv = 0
    kn = 0
    nn = 0
    counter = 0
    for terka in tabela:
        if terka[1] == 'Decision_tree':
            dt += terka[0]
            counter = counter + 1
        elif terka[1] == 'Random_forest':
            rf += terka[0]
        elif terka[1] == 'SVM':
            sv += terka[0]
        elif terka[1] == 'KNN':
            kn += terka[0]
        elif terka[1] == 'Neutral_Networks':
            nn += terka[0]
    dt = dt / counter
    rf = rf / counter
    sv = sv / counter
    kn = kn / counter
    nn = nn / counter
    return round(dt, 3), round(rf, 3), round(sv, 3), round(kn, 3), round(nn, 3)


def IzracunajIIspisiPovprecje():
    f.write('Average results\n')
    dt, rf, sv, kn, nn = IzracunajPovprecje(acc_table)
    f.write(f'Accuracy:\n')
    f.write(f'DT:{dt}  RF:{rf}  SVM:{sv}  KNN:{kn}  NN:{nn}\n')
    dt, rf, sv, kn, nn = IzracunajPovprecje(sd_table)
    f.write(f'Standard deviation:\n')
    f.write(f'DT:{dt}  RF:{rf}  SVM:{sv}  KNN:{kn}  NN:{nn}\n')
    dt, rf, sv, kn, nn = IzracunajPovprecje(sensitivity_table)
    f.write(f'Sensitivity:\n')
    f.write(f'DT: {dt}  RF:{rf}  SVM:{sv}  KNN:{kn}  NN:{nn}\n')
    dt, rf, sv, kn, nn = IzracunajPovprecje(specificity_table)
    f.write(f'Specificity:\n')
    f.write(f'DT: {dt}  RF:{rf}  SVM:{sv}  KNN:{kn}  NN:{nn}\n')
    dt, rf, sv, kn, nn = IzracunajPovprecje(auc_table)
    f.write(f'AUC score:\n')
    f.write(f'DT: {dt}  RF:{rf}  SVM:{sv}  KNN:{kn}  NN:{nn}\n')


acc_table = []
sd_table = []
sensitivity_table = []
specificity_table = []
auc_table = []


if __name__ == '__main__':
    # citame podatoci i gi spremame za obrabotka
    table = pandas.read_csv("data/data.csv")
    table = ChangeValues(table)
    dummy_table = pandas.get_dummies(table)
    # imame k-kratno proveruvanje, gi delime podatocite na ucni(80%) i testni(20%)
    redovi = list(range(0, len(dummy_table)))
    random.Random(0).shuffle(redovi)
    brojUcni = math.floor(len(redovi) * 0.8)
    brojTestni = len(redovi) - brojUcni
    brojIteracii = math.ceil(len(dummy_table) / brojTestni)
    for i in range(brojIteracii):
        x_train, y_train, x_test, y_test = CreateTrainAndTest(dummy_table, i, brojTestni, redovi)
        DecisionTree(x_train, y_train, x_test, y_test, i)
        RandomForest(x_train, y_train, x_test, y_test, i)
        SVM(x_train, y_train, x_test, y_test, i)
        KNN(x_train, y_train, x_test, y_test, i)
        NeuralNetworks(x_train, y_train, x_test, y_test, i)
    IzracunajIIspisiPovprecje()
    f.close()
