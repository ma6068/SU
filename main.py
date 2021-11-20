import pandas
import math
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
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


def CalculateErrors(Y_test, Y_predict):
    acc = accuracy_score(Y_test, Y_predict)
    mae = mean_absolute_error(Y_test, Y_predict)
    rmse = math.sqrt(mean_squared_error(Y_test, Y_predict))
    return round(acc, 3), round(mae, 3), round(rmse, 3)


def DecisionTree(X_train, Y_train, X_test, Y_test):
    model = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
    model.fit(X_train, Y_train)
    Y_predicted = model.predict(X_test)
    acc, mae, rmse = CalculateErrors(Y_test, Y_predicted)
    print(acc, mae, rmse)


def RandomForest(X_train, Y_train, X_test, Y_test):
    model = RandomForestClassifier(max_depth=4)
    model.fit(X_train, Y_train)
    Y_predicted = model.predict(X_test)
    acc, mae, rmse = CalculateErrors(Y_test, Y_predicted)
    print(acc, mae, rmse)


def SVM(X_train, Y_train, X_test, Y_test):
    model = svm.SVC(kernel='linear', random_state=0, probability=True)
    model.fit(X_train, Y_train)
    Y_predicted = model.predict(X_test)
    acc, mae, rmse = CalculateErrors(Y_test, Y_predicted)
    print(acc, mae, rmse)


def KNN(X_train, Y_train, X_test, Y_test):
    model = KNeighborsClassifier(n_neighbors=20)
    model.fit(X_train, Y_train)
    Y_predicted = model.predict(X_test)
    acc, mae, rmse = CalculateErrors(Y_test, Y_predicted)
    print(acc, mae, rmse)


def NeuralNetworks(X_train, Y_train, X_test, Y_test):
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25,), random_state=0)
    model.fit(X_train, Y_train)
    Y_predicted = model.predict(X_test)
    acc, mae, rmse = CalculateErrors(Y_test, Y_predicted)
    print(acc, mae, rmse)


if __name__ == '__main__':
    table = pandas.read_csv("data.csv")
    table = ChangeValues(table)
    dummy_table = pandas.get_dummies(table)
    x_train, y_train, x_test, y_test = CreateTrainAndTestSamples(dummy_table)
    # DecisionTree(x_train, y_train, x_test, y_test)
    # RandomForest(x_train, y_train, x_test, y_test)
    # SVM(x_train, y_train, x_test, y_test)
    # KNN(x_train, y_train, x_test, y_test)
    # NeuralNetworks(x_train, y_train, x_test, y_test)