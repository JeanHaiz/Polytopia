import json
from ortools.sat.python import cp_model


MAX_CITY_LEVEL = 10


def createDic():
    global dic
    dic = {}


def createModel():
    global model
    model = cp_model.CpModel()


def __loadFile(filename):
    with open(filename, 'r') as file:
        return json.load(file)


def getValues():
    return __loadFile('values.json')


def getTechs():
    return __loadFile('tech_tree.json')


def getTribes():
    return __loadFile('tribes.json')


def getRuins():
    return __loadFile('ruin.json')


def getCost():
    return __loadFile('cost.json')


def getPopulation():
    return __loadFile('population.json')


createDic()
createModel()
