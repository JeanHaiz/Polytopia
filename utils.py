from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import pandas as pd

from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import _SumArray


class DataFrameSolutionAggregator(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__pd_variables = []
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        values = []
        for v in self.__variables:
            values.append(self.Value(v))
        self.__pd_variables.append(values)

    def get_data_frame(self):
        df = pd.DataFrame(self.__pd_variables, columns=self.__variables)
        return df.loc[:, (df != 0).any(axis=0)]

    def solution_count(self):
        return self.__solution_count


def __loadFile(filename):
    with open(filename, 'r') as file:
        return json.load(file)


def getValues():
    return __loadFile('values.json')


def getTechs():
    return __loadFile('tech_tree.json')


def getTribes():
    return __loadFile('tribes.json')


def addAllVars(model, turn):
    values = getValues()
    eq_dict = {}
    sym_dict = {}
    for category in values.keys():
        for element, score in values[category].items():
            name = turn + "_" + element
            # replace 10 with max value of elements, static or dynamic
            sym_dict[name] = model.NewIntVar(0, 30, name)
            eq_dict[name] = score * sym_dict[name]
    return eq_dict, sym_dict


def addAllTechs(model, turn):
    techs = getTechs()
    tech_dict = {}
    for t1 in techs:
        k1 = turn + "_" + t1['name']
        tech_dict[k1] = model.NewBoolVar(k1)
        for t2 in t1['allows']:
            k2 = turn + "_" + t2['name']
            tech_dict[k2] = model.NewBoolVar(k2)
            for t3 in t2['allows']:
                k3 = turn + "_" + t3['name']
                tech_dict[k3] = model.NewBoolVar(k3)
    return tech_dict


def addAllTribes(model, turn, sym_dict, tech_dict):
    tribes = getTribes()
    tribe_dict = {}
    for t1 in tribes:
        k1 = turn + "_" + t1['name']
        tribe_dict[k1] = model.NewBoolVar(k1)
        if t1['tech'] != '':
            model.Add(tech_dict[turn + "_" + t1['tech']] == 1).OnlyEnforceIf(tribe_dict[k1])
    model.Add(_SumArray(tribe_dict.values()) == 1)
    return tribe_dict


def addScore(model, turn):
    score_dict = {
        turn + "_" + "raw_score": model.NewIntVar(0, 200000, turn + "_" + "raw_score"),
        # turn + "_" + "delta_raw_score" : model.NewIntVar(0, 200000, turn + "_" + "delta_raw_score"),
        turn + "_" + "full_score": model.NewIntVar(0, 200000, turn + "_" + "full_score"),
        # turn + "_" + "delta_full_score" : model.NewIntVar(0, 200000, turn + "_" + "delta_full_score"),
    }
    return score_dict


def setTribe(model, turn, tribe, tribe_dict):
    model.Add(tribe_dict[turn + "_" + tribe] == 1)


def setRawScore(model, turn, score_dict, raw_score):
    model.Add(score_dict[turn + "_" + 'raw_score'] == raw_score)


def setDeltaRawScore(model, turn, score_dict, delta_raw_score):
    model.Add(score_dict[turn + "_" + 'delta_raw_score'] == delta_raw_score)


def setFullScore(model, turn, score_dict, full_score):
    model.Add(score_dict[turn + "_" + 'full_score'] == full_score)


def setDeltaFullScore(model, turn, score_dict, delta_full_score):
    model.Add(score_dict[turn + "_" + 'delta_full_score'] == delta_full_score)


def buildCityConstrains(model, turn, sym_dict):
    for i in range(1, 10):  # TODO change bounds
        model.Add(sym_dict[turn + "_" + "level-%d" % i] >= sym_dict[turn + "_" + "level-%d" % (i + 1)])


def buildGiantConstraints(model, turn, sym_dict):
    pass
    # model.Add()


def buildCityPopConstraint(model, turn, sym_dict):
    model.Add(sym_dict[turn + "_" + 'population'] >= _SumArray(
        [i * sym_dict[turn + "_" + "level-%d" % i] for i in range(2, 11)]))


def buildUnitTrainByCityConstraint(model, turn, sym_dict):
    model.Add(sym_dict[turn + "_" + 'level-1'] >= _SumArray(
        [sym_dict[turn + "_" + u] for u in getValues()['units'].keys()]))


def buildTribeConstraints(model, turn, tribe_dict, tech_dict):
    tribes = getTribes()
    for t1 in tribes:
        k1 = turn + "_" + t1['name']
        if t1['tech'] != '':
            model.Add(tech_dict[turn + "_" + t1['tech']] == 1).OnlyEnforceIf(tribe_dict[k1])
    model.Add(_SumArray(tribe_dict.values()) == 1)


def buildTechTreeConstraints(model, turn, tech_dict):
    techs = getTechs()
    for t1 in techs:
        k1 = turn + "_" + t1['name']
        for t2 in t1['allows']:
            k2 = turn + "_" + t2['name']
            model.Add(tech_dict[k2] == 0).OnlyEnforceIf(tech_dict[k1].Not())
            for t3 in t2['allows']:
                k3 = turn + "_" + t3['name']
                model.Add(tech_dict[k3] == 0).OnlyEnforceIf(tech_dict[k2].Not())


def buildTechUnitsConstraints(model, turn, tech_dict, sym_dict):
    techs = getTechs()
    for t1 in techs:
        k1 = turn + "_" + t1['name']
        for u in t1['units']:
            model.Add(sym_dict[turn + "_" + u] == 0).OnlyEnforceIf(tech_dict[k1].Not())
        for t2 in t1['allows']:
            k2 = turn + "_" + t2['name']
            for u in t2['units']:
                model.Add(sym_dict[turn + "_" + u] == 0).OnlyEnforceIf(tech_dict[k2].Not())
            for t3 in t2['allows']:
                k3 = turn + "_" + t3['name']
                for u in t3['units']:
                    model.Add(sym_dict[turn + "_" + u] == 0).OnlyEnforceIf(tech_dict[k3].Not())


def linkTechAndTech(model, turn, tech_dict, sym_dict):
    techs = getTechs()
    model.Add(sym_dict[turn + "_" + 'tier-1'] == _SumArray([tech_dict[turn + "_" + t['name']] for t in techs]))
    model.Add(sym_dict[turn + "_" + 'tier-2'] == _SumArray(
        [tech_dict[turn + "_" + t['name']] for u in techs for t in u['allows']]))
    model.Add(sym_dict[turn + "_" + 'tier-2'] == _SumArray(
        [tech_dict[turn + "_" + t['name']] for v in techs for u in v['allows'] for t in u['allows']]))


def buildFullScoreConstraint(model, turn, eq_dict, score_dict):
    values = getValues()
    cat_eq_dict = {}
    for category in getValues().keys():
        cat_eq_dict[turn + "_" + category] = _SumArray(
            [eq_dict[turn + "_" + element] for element in values[category].keys()])
    model.Add(_SumArray(
        [cat_eq_dict[cat] for cat in cat_eq_dict.keys()]) == score_dict[turn + "_" + 'full_score'])


def buildMaxUnitConstraint(model, turn, sym_dict):
    values = getValues()
    model.Add(_SumArray([sym_dict[turn + "_" + e] for e in values['units'].keys()]) < 20)


def buildVisionConstraint(model, turn, eq_dict, score_dict):
    model.Add(eq_dict[turn + "_" + 'revealed'] ==
              (score_dict[turn + "_" + 'full_score'] - score_dict[turn + "_" + 'raw_score']))


def buildAllConstraints(model, turn, sym_dict, eq_dict, tech_dict, tribe_dict, score_dict):
    buildVisionConstraint(model, turn, eq_dict, score_dict)
    buildFullScoreConstraint(model, turn, eq_dict, score_dict)
    buildTechUnitsConstraints(model, turn, tech_dict, sym_dict)
    buildTechTreeConstraints(model, turn, tech_dict)
    # buildMaxUnitConstraint(model, turn, sym_dict)
    buildCityConstrains(model, turn, sym_dict)
    buildTribeConstraints(model, turn, tribe_dict, tech_dict)
    linkTechAndTech(model, turn, tech_dict, sym_dict)
    buildCityPopConstraint(model, turn, sym_dict)
    buildUnitTrainByCityConstraint(model, turn, sym_dict)
