def readData():
    with open('cleveland_data') as cl:
        lines = cl.readlines()
    return lines


def writeData(lines):
    with open('data.csv', 'w') as f:
        f.write('age,sex,cpt,rbp,sch,fbs,res,mhr,eia,opk,pes,vca,tha,target\n')
        for line in lines:
            data = line.split(',')
            if '?' not in data:
                for i in range(len(data)):
                    if i != 13:
                        f.write(data[i] + ',')
                    else:
                        f.write(data[i])


if __name__ == '__main__':
    lines_data = readData()
    writeData(lines_data)