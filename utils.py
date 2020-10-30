from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import pandas as pd

from ortools.sat.python import cp_model


MAX_CITY_LEVEL = 10


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
        df = pd.DataFrame(self.__pd_variables, columns=[str(s) for s in self.__variables])
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


def getRuins():
    return __loadFile('ruin.json')


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


def addAllVars(model, turn):
    values = getValues()
    eq_dict = {}
    sym_dict = {}
    for category in values.keys():
        for element, score in values[category].items():
            name = turn + "_" + element
            # replace 10 with max value of elements, static or dynamic
            sym_dict[name] = model.NewIntVar(0, 50, name)
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
    return tribe_dict


def addScore(model, turn):
    score_dict = {
        turn + "_" + "raw_score": model.NewIntVar(0, 200000, turn + "_" + "raw_score"),
        # turn + "_" + "delta_raw_score" : model.NewIntVar(0, 200000, turn + "_" + "delta_raw_score"),
        turn + "_" + "full_score": model.NewIntVar(0, 200000, turn + "_" + "full_score"),
        # turn + "_" + "delta_full_score" : model.NewIntVar(0, 200000, turn + "_" + "delta_full_score"),
        # turn + "_" + "computed_full_score": model.NewIntVar(0, 200000, turn + "_" + "computed_full_score")
        # turn + "_" + "computed_raw_score": model.NewIntVar(0, 200000, turn + "_" + "computed_raw_score"),
    }
    return score_dict


def addSpecial(model, turn):
    special_dict = {}
    # turn + "_" + "whale" : model.NewIntVar(0, 20, turn + "_" + "whale"),
    special = getRuins()
    for s in special:
        k = turn + "_" + s['name']
        special_dict[k] = model.NewIntVar(0, 20, k)
        for r in s['output']:
            k2 = k + "_" + r['name']
            special_dict[k2] = model.NewIntVar(0, 20, k2)
    return special_dict


def addCityUpgrade(model, turn):
    city_upgrade = {}
    city_eq = {}
    city_upgrade[turn + "_" + "level-2_spt"] = model.NewIntVar(0, 20, turn + "_" + "level-2_spt")
    city_upgrade[turn + "_" + "level-2_explorer"] = model.NewIntVar(0, 20, turn + "_" + "level-2_explorer")
    city_upgrade[turn + "_" + "level-4_border_growth"] = model.NewIntVar(0, 20, turn + "_" + "level-4_border_growth")
    city_upgrade[turn + "_" + "level-4_population"] = model.NewIntVar(0, 20, turn + "_" + "level-4_population")
    for i in range(5, MAX_CITY_LEVEL + 1):
        city_upgrade[turn + "_" + "level-" + str(i) + "_giant"] = \
            model.NewIntVar(0, 20, turn + "_" + "level-" + str(i) + "_giant")
        city_upgrade[turn + "_" + "level-" + str(i) + "_garden"] = \
            model.NewIntVar(0, 20, turn + "_" + "level-" + str(i) + "_garden")

    for i in range(1, MAX_CITY_LEVEL + 1):
        city_upgrade[turn + "_" + "level-" + str(i) + "_count"] = \
            model.NewIntVar(0, 20, turn + "_" + "level-" + str(i) + "_count")

    city_upgrade[turn + "_" + "pop_min"] = model.NewIntVar(0, 40, turn + "_" + "pop_min")
    city_upgrade[turn + "_" + "pop_max"] = model.NewIntVar(0, 40, turn + "_" + "pop_max")

    return city_upgrade, city_eq


def buildCityUpgradeConstraints(model, turn, dictionnaries):
    # diff city level with previous turn + capture == consequences for each level
    # - level-2: spt or explorer
    # - level-3: stars or nothing, so >=
    # - level-4: border growth or pop
    # - level-5+: giant or garden
    model.Add(dictionnaries[turn]['symbols'][turn + "_" + "level-2"] ==
              (dictionnaries[turn]['city'][turn + "_" + "level-2_spt"] +
              dictionnaries[turn]['city'][turn + "_" + "level-2_explorer"]))

    # Add Ruin explorer
    model.Add(
        dictionnaries[turn]['city'][turn + "_" + "level-2_explorer"] +
        dictionnaries[turn]['special'][turn + "_" + "ruin" + "_" + "explorer"] ==
        dictionnaries[turn]['symbols'][turn + "_" + "explorer"])

    # Add level-3 stars

    model.Add(dictionnaries[turn]['symbols'][turn + "_" + "level-4"] ==
              dictionnaries[turn]['city'][turn + "_" + "level-4_border_growth"] +
              dictionnaries[turn]['city'][turn + "_" + "level-4_population"])

    for i in range(5, MAX_CITY_LEVEL + 1):
        model.Add(dictionnaries[turn]['symbols'][turn + "_" + "level-" + str(i)] ==
                  dictionnaries[turn]['city'][turn + "_" + "level-" + str(i) + "_giant"] +
                  dictionnaries[turn]['city'][turn + "_" + "level-" + str(i) + "_garden"])


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
    for i in range(1, MAX_CITY_LEVEL):
        model.Add(sym_dict[turn + "_" + "level-" + str(i)] >= sym_dict[turn + "_" + "level-" + str(i + 1)])


def buildGiantConstraints(model, turn, dictionnaries):
    # Add conversion
    model.Add(
        dictionnaries[turn]['symbols'][turn + "_" + "giant"] ==
        sum(dictionnaries[turn]['city'][turn + "_" + "level-" + str(i) + "_giant"]
            for i in range(5, MAX_CITY_LEVEL + 1)) +
        dictionnaries[turn]['special'][turn + "_" + "ruin_giant"])
    # giants >= city upgrade (lvl 5+) + ruin_giant + convert_giant


def buildUnitTrainByCityConstraint(model, turn, dictionnaries):
    # Add conversion, giant on lvl_up, ruin
    past_turn = getPastTurn(turn)
    if past_turn is not None:
        model.Add(
            dictionnaries[turn]['symbols'][turn + "_" + 'level-1'] +
            dictionnaries[turn]['special'][turn + "_" + 'ruin' + "_" + "giant"] +
            sum(dictionnaries[turn]['city'][turn + "_" + "level-" + str(i) + "_giant"] -
                dictionnaries[past_turn]['city'][past_turn + "_" + "level-" + str(i) + "_giant"]
                for i in range(5, MAX_CITY_LEVEL + 1)) >= (
                sum(dictionnaries[turn]['symbols'][turn + "_" + u] for u in getValues()['units'].keys()) -
                sum(dictionnaries[past_turn]['symbols'][past_turn + "_" + u] for u in getValues()['units'].keys())
            )
        )


def buildTribeConstraints(model, turn, tribe_dict, tech_dict):
    tribes = getTribes()
    for t1 in tribes:
        k1 = turn + "_" + t1['name']
        if t1['tech'] != '':
            model.Add(tech_dict[turn + "_" + t1['tech']] == 1).OnlyEnforceIf(tribe_dict[k1])
    model.Add(sum(tribe_dict.values()) == 1)


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
    model.Add(sym_dict[turn + "_" + 'tier-1'] == sum([tech_dict[turn + "_" + t['name']] for t in techs]))
    model.Add(sym_dict[turn + "_" + 'tier-2'] == sum(
        [tech_dict[turn + "_" + t['name']] for u in techs for t in u['allows']]))
    model.Add(sym_dict[turn + "_" + 'tier-3'] == sum(
        [tech_dict[turn + "_" + t['name']] for v in techs for u in v['allows'] for t in u['allows']]))


def buildFullScoreConstraint(model, turn, dictionnaries):
    eq_dict = dictionnaries[turn]["equations"]
    score_dict = dictionnaries[turn]["scores"]
    values = getValues()
    cat_eq_dict = {}
    for category in getValues().keys():
        cat_eq_dict[turn + "_" + category] = sum(
            [eq_dict[turn + "_" + element] for element in values[category].keys()])
    # print(sum([cat_eq_dict[cat] for cat in cat_eq_dict.keys()]))
    model.Add(sum([cat_eq_dict[cat] for cat in cat_eq_dict.keys()]) == score_dict[turn + "_" + 'full_score'])
    # score_dict[turn + "_" + 'computed_full_score'])
    # model.Add(score_dict[turn + "_" + 'computed_full_score'] == score_dict[turn + "_" + 'full_score'])


def buildClaimingCityConstraint(model, turn, sym_dict):
    # event to other tribe
    pass


def buildMaxUnitConstraint(model, turn, dictionnaries):
    values = getValues()
    past_turn = getPastTurn(turn)
    # add conversion
    if past_turn is not None:
        model.Add(
            sum(dictionnaries[turn]["city"][turn + "_" + "level-" + str(i) + "_count"] * (i + 1)
                for i in range(1, MAX_CITY_LEVEL + 1)) >=
            sum(dictionnaries[turn]["symbols"][turn + "_" + e] for e in values['units'].keys()))


def buildVisionConstraint(model, turn, eq_dict, score_dict):
    model.Add(eq_dict[turn + "_" + 'revealed'] ==
              (score_dict[turn + "_" + 'full_score'] - score_dict[turn + "_" + 'raw_score']))


def buildSpecialConstraints(model, turn, dictionnaries):
    for r in getRuins():
        ruins = [dictionnaries[turn]['special'][turn + "_" + r['name'] + "_" + s['name']] for s in r['output']]
        model.Add(dictionnaries[turn]['special'][turn + "_" + r['name']] == sum(ruins))


def buildPopulationConstraints(model, turn, dictionnaries):
    # Add meet tribe pop
    past_turn = getPastTurn(turn)
    model.Add(
        dictionnaries[turn]['symbols'][turn + "_" + 'population'] - (
            dictionnaries[past_turn]['symbols'][past_turn + "_" + 'population'] if past_turn is not None else 0
        ) >= dictionnaries[turn]['special'][turn + "_" + "ruin" + "_" + "population"] +
        dictionnaries[turn]['city'][turn + "_" + "level-4_population"] - ((
            dictionnaries[past_turn]['city'][past_turn + "_" + "level-4_population"] +
            dictionnaries[past_turn]['special'][past_turn + "_" + "ruin" + "_" + "population"]
        ) if past_turn is not None else 0))
    # pop >= ruin_pop + lvl-4_pop + tribe_meet_pop + ...


def buildClaimedConstraints(model, turn, dictionnaries):
    past_turn = getPastTurn(turn)
    if past_turn is not None:

        # x = lvl-1 cst
        # y = claimed cst
        # x => y
        # x.Not() or y
        # not x => not y or y
        # z = bool
        # z => x and z => y
        # not z => not x, not z => not y

        lvl_1_diff = (dictionnaries[turn]['symbols'][turn + "_" + "level-1"] -
                      dictionnaries[past_turn]['symbols'][past_turn + "_" + "level-1"])

        lvl_4_diff = (dictionnaries[turn]['city'][turn + "_" + "level-4_border_growth"] -
                      dictionnaries[past_turn]['city'][past_turn + "_" + "level-4_border_growth"])

        claimed_diff = (dictionnaries[turn]['symbols'][turn + "_" + "claimed"] -
                        dictionnaries[past_turn]['symbols'][past_turn + "_" + "claimed"])

        lvl_1_cst = dictionnaries[turn]['symbols'][turn + "_" + 'lvl_1_cst']
        claimed_cst = dictionnaries[turn]['symbols'][turn + "_" + 'claimed_cst']

        model.Add((lvl_1_diff + lvl_4_diff) != 0).OnlyEnforceIf(lvl_1_cst)
        model.Add((lvl_1_diff == 0) and (lvl_4_diff == 0)).OnlyEnforceIf(lvl_1_cst.Not())
        model.Add(claimed_diff == 0).OnlyEnforceIf(claimed_cst)
        model.Add(claimed_diff != 0).OnlyEnforceIf(claimed_cst.Not())
        model.Add(lvl_1_cst + claimed_cst >= 1)

        model.Add(lvl_1_diff <= sum(
            dictionnaries[past_turn]['symbols'][past_turn + "_" + u] for u in getValues()['units'].keys()))


def buildCityPopConstraints(model, turn, dictionnaries):
    if turn != "start":
        for i in range(1, MAX_CITY_LEVEL):
            model.Add((dictionnaries[turn]['city'][turn + "_" + "level-" + str(i) + "_count"] +
                       dictionnaries[turn]['symbols'][turn + "_" + "level-" + str(i + 1)]) ==
                      dictionnaries[turn]['symbols'][turn + "_" + "level-" + str(i)])
        model.Add(dictionnaries[turn]['city'][turn + "_" + "level-" + str(MAX_CITY_LEVEL) + "_count"] ==
                  dictionnaries[turn]['symbols'][turn + "_" + "level-" + str(MAX_CITY_LEVEL)])

    model.Add(
        dictionnaries[turn]['city'][turn + "_" + "pop_min"] ==
        sum(
            dictionnaries[turn]['city'][turn + "_" + "level-" + str(i) + "_count"] *
            int(i * (i + 1) / 2 - 1) for i in range(1, MAX_CITY_LEVEL + 1)
        )
    )
    model.Add(
        dictionnaries[turn]['symbols'][turn + "_" + "population"] >=
        dictionnaries[turn]['city'][turn + "_" + "pop_min"]
    )

    model.Add(
        dictionnaries[turn]['city'][turn + "_" + "pop_max"] ==
        sum(
            dictionnaries[turn]['city'][turn + "_" + "level-" + str(i) + "_count"] *
            int((i + 1) * (i + 2) / 2 - 2) for i in range(1, MAX_CITY_LEVEL + 1)
        )
    )
    model.Add(
        dictionnaries[turn]['symbols'][turn + "_" + "population"] <=
        dictionnaries[turn]['city'][turn + "_" + "pop_max"]
    )


def buildRuinLinkConstraints(model, turn, dictionnaries):
    past_turn = getPastTurn(turn)
    if past_turn is not None:
        model.Add(
            (
                dictionnaries[turn]['special'][turn + "_" + "ruin" + "_" + "tech"] -
                dictionnaries[past_turn]['special'][past_turn + "_" + "ruin" + "_" + "tech"]
            ) <= (
                sum(dictionnaries[turn]['symbols'][turn + "_" + "tier-" + str(i)] for i in range(1, 4)) -
                sum(dictionnaries[past_turn]['symbols'][past_turn + "_" + "tier-" + str(i)] for i in range(1, 4))
            )
        )
        model.Add(
            3 * (
                dictionnaries[turn]['special'][turn + "_" + "ruin" + "_" + "population"] -
                dictionnaries[past_turn]['special'][past_turn + "_" + "ruin" + "_" + "population"]
            ) <= (
                dictionnaries[turn]['symbols'][turn + "_" + "population"] -
                dictionnaries[past_turn]['symbols'][past_turn + "_" + "population"]
            )
        )
        model.Add(
            sum(
                dictionnaries[turn]['special'][turn + "_" + "ruin" + "_" + e['name']] for e in getRuins()[0]['output']
            ) <= sum(
                dictionnaries[past_turn]["symbols"][past_turn + "_" + e] for e in getValues()['units']
            )
        )
    else:
        model.Add(
            dictionnaries[turn]['special'][turn + "_" + "ruin" + "_" + "tech"] <=
            sum(dictionnaries[turn]['symbols'][turn + "_" + "tier-" + str(i)] for i in range(1, 4))
        )
        model.Add(
            3 * dictionnaries[turn]['special'][turn + "_" + "ruin" + "_" + "population"] <=
            dictionnaries[turn]['symbols'][turn + "_" + "population"]
        )
        model.Add(
            sum(
                dictionnaries[turn]['special'][turn + "_" + "ruin" + "_" + e['name']] for e in getRuins()[0]['output']
            ) == 0
        )
    model.Add(dictionnaries[turn]['special'][turn + "_" + "whale"] == 0).OnlyEnforceIf(
        dictionnaries[turn]['technologies'][turn + "_" + 'whaling'].Not()
    )


def buildAllConstraints(model, turn, dictionnaries):
    eq_dict = dictionnaries[turn]["equations"]
    score_dict = dictionnaries[turn]["scores"]
    tech_dict = dictionnaries[turn]["technologies"]
    sym_dict = dictionnaries[turn]["symbols"]
    tribe_dict = dictionnaries[turn]["tribes"]
    # city_eq_dict = dictionnaries[turn]["city_equations"]
    buildVisionConstraint(model, turn, eq_dict, score_dict)
    buildFullScoreConstraint(model, turn, dictionnaries)
    buildTechUnitsConstraints(model, turn, tech_dict, sym_dict)
    buildTechTreeConstraints(model, turn, tech_dict)
    buildCityConstrains(model, turn, sym_dict)
    buildTribeConstraints(model, turn, tribe_dict, tech_dict)
    linkTechAndTech(model, turn, tech_dict, sym_dict)
    buildCityPopConstraints(model, turn, dictionnaries)
    buildUnitTrainByCityConstraint(model, turn, dictionnaries)
    buildClaimedConstraints(model, turn, dictionnaries)
    buildCityUpgradeConstraints(model, turn, dictionnaries)
    buildRuinLinkConstraints(model, turn, dictionnaries)

    buildGiantConstraints(model, turn, dictionnaries)
    buildMaxUnitConstraint(model, turn, dictionnaries)
    buildSpecialConstraints(model, turn, dictionnaries)
    buildPopulationConstraints(model, turn, dictionnaries)


def setObjective(model, turn, dictionnaries):
    # min ruin events
    # min tribe meeting
    pass


def linkTurns(model, turn, dictionnaries):
    past_turn = getPastTurn(turn)
    if past_turn is not None:
        for dic_name, dic in dictionnaries[past_turn].items():
            for var_name, var in dic.items():
                model.Add(var <= dictionnaries[turn][dic_name][turn + "_" + "_".join(var_name.split("_")[1:])])


def buildDictionnaries(model, turn):
    dictionnaries = {}
    dictionnaries['equations'], dictionnaries['symbols'] = addAllVars(model, turn)
    dictionnaries['technologies'] = addAllTechs(model, turn)
    dictionnaries['tribes'] = addAllTribes(model, turn, dictionnaries['symbols'], dictionnaries['technologies'])
    dictionnaries['scores'] = addScore(model, turn)
    if turn != "start":
        dictionnaries['symbols'][turn + "_" + 'claimed_cst'] = model.NewBoolVar(turn + "_" + 'claimed_cst')
        dictionnaries['symbols'][turn + "_" + 'lvl_1_cst'] = model.NewBoolVar(turn + "_" + 'lvl_1_cst')
    dictionnaries['special'] = addSpecial(model, turn)
    dictionnaries["city"], dictionnaries["city_equations"] = addCityUpgrade(model, turn)
    return dictionnaries


def addInitialState(model, turn, dictionnaries):
    if turn == "start":
        model.Add(dictionnaries[turn]['symbols'][turn + "_" + "claimed"] == 9)  # allow for some params
        model.Add(dictionnaries[turn]['symbols'][turn + "_" + "revealed"] == 25)
        model.Add(dictionnaries[turn]['symbols'][turn + "_" + "level-1"] == 1)
        model.Add(dictionnaries[turn]['symbols'][turn + "_" + "population"] == 0)
        model.Add(dictionnaries[turn]['special'][turn + "_" + "ruin_stars"] == 0)
        model.Add(dictionnaries[turn]['special'][turn + "_" + "whale_stars"] == 0)

        for i in range(2, MAX_CITY_LEVEL + 1):
            model.Add(dictionnaries[turn]['symbols'][turn + "_" + "level-" + str(i)] == 0)
        for tribe in getTribes():
            k1 = turn + "_" + tribe['name']
            model.Add(dictionnaries[turn]['symbols'][turn + "_" + tribe['unit']] == 1
                      ).OnlyEnforceIf(dictionnaries[turn]['tribes'][k1])
            for u in getValues()['units']:
                if u != tribe['unit']:
                    model.Add(dictionnaries[turn]['symbols'][turn + "_" + u] == 0
                              ).OnlyEnforceIf(dictionnaries[turn]['tribes'][k1])
                else:
                    model.Add(dictionnaries[turn]['symbols'][turn + "_" + u] == 1
                              ).OnlyEnforceIf(dictionnaries[turn]['tribes'][k1])
            for name, tech in dictionnaries[turn]['technologies'].items():
                if name != turn + "_" + tribe['tech']:
                    model.Add(dictionnaries[turn]['technologies'][name] == 0
                              ).OnlyEnforceIf(dictionnaries[turn]['tribes'][k1])


def addSolution(model, turn, dictionnaries):
    model.Add(dictionnaries["start"]['symbols']["start_monument"] == 0)
    model.Add(dictionnaries["start"]['city']["start_level-2_count"] == 0)
    model.Add(dictionnaries["start"]['city']["start_level-5_count"] == 0)
    model.Add(dictionnaries["t0"]['symbols']["t0_level-1"] == 1)
    model.Add(dictionnaries["t0"]['symbols']["t0_level-2"] == 1)
    model.Add(dictionnaries["t0"]['symbols']["t0_level-3"] == 0)
    model.Add(dictionnaries["t0"]['city']["t0_level-1_count"] == 0)
    model.Add(dictionnaries["t0"]['city']["t0_level-2_count"] == 1)
    model.Add(dictionnaries["t0"]['city']["t0_level-3_count"] == 0)
    model.Add(dictionnaries["t0"]['city']["t0_level-4_count"] == 0)
    model.Add(dictionnaries["t0"]['city']["t0_level-5_count"] == 0)
    model.Add(dictionnaries["t0"]['symbols']["t0_population"] == 2)
    model.Add(dictionnaries["t0"]['symbols']["t0_claimed"] == 9)
    model.Add(dictionnaries["t0"]['symbols']["t0_tier-1"] == 1)
    model.Add(dictionnaries["t0"]['symbols']["t0_tier-2"] == 0)
    model.Add(dictionnaries["t0"]['symbols']["t0_monument"] == 0)
    model.Add(dictionnaries["t0"]['symbols']["t0_giant"] == 0)
    model.Add(dictionnaries["t0"]['symbols']["t0_warrior"] == 1)
    model.Add(dictionnaries["t0"]['symbols']["t0_explorer"] == 0)


def printValueList(df):
    count = 1
    for i in range(len(df.columns)):
        c = list(df.columns)[i]
        print(c, len(df[c].unique()), df[c].unique())
        count *= len(df[c].unique())
    return count


def printVariants(df):
    count = 1
    variants = []
    for i in range(len(df.columns)):
        c = list(df.columns)[i]
        u = df[c].unique()
        if len(u) > 1:
            print(c, len(u), u)
            count *= len(u)
            variants.append((c, u))
    return variants


def getTechList():

    def rec(tech_dict):
        if len(tech_dict['allows']) == 0:
            return [tech_dict['name']]
        else:
            return [u for t in tech_dict['allows'] for u in rec(t)] + [tech_dict['name']]

    techs = getTechs()
    return [u for t in techs for u in rec(t)]


def getScenarioRegex():
    scenario_regex = []
    for u in list(getValues()['units'].keys()) +\
            ['tier-' + str(i) for i in range(1, 4)] +\
            ['level-' + str(i) + "_count" for i in range(1, MAX_CITY_LEVEL)] +\
            list(map(lambda t: t['name'], getTribes())) +\
            getTechList() +\
            ['{1}population', 'claimed', 'revealed', 'level-2_spt', 'level-4_border_growth',
                'level-4_population', r'ruin([_a-z]*)']:
        scenario_regex.append(rf"(t[0-9]*_{u})")
    return scenario_regex


def chooseBestSolution(df):
    for star_column in list(filter(lambda s: 'stars' in s, df.columns)):
        min_star = min(df[star_column])
        df = df[df[star_column] == min_star]
    for ruin_giant_column in list(filter(lambda s: 'ruin_giant' in s, df.columns)):
        min_ruin_giant = min(df[ruin_giant_column])
        df = df[df[ruin_giant_column] == min_ruin_giant]
    return df
