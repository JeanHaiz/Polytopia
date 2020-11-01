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
    with open("json" + "/" + filename, 'r') as file:
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


def getNextTurn(turn):
    if turn is None:
        return "start"
    elif turn == "start":
        return "t0"
    else:
        return "t" + str(int(turn[1:]) + 1)


def getPastTurn(turn):
    if turn == "start" or turn is None:
        return None
    elif turn == "t0":
        return "start"
    else:
        return "t" + str(int(turn[1:]) - 1)


def name(turn, var):
    return turn + "_" + var


def get(turn, var):
    return dic[name(turn, var)]


def diff(turn, var):
    past_turn = getPastTurn(turn)
    if past_turn is not None:
        return get(turn, var) - get(past_turn, var)
    else:
        return get(turn, var)


def getTechList():

    def rec(d):
        if len(d['allows']) == 0:
            return [d['name']]
        else:
            return [u for t in d['allows'] for u in rec(t)] + [d['name']]

    techs = getTechs()
    return [u for t in techs for u in rec(t)]


createDic()
createModel()
