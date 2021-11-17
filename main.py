import pandas


def changeValues(tabela):
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


if __name__ == '__main__':
    table = pandas.read_csv("data.csv")
    table = changeValues(table)
    print(table)
