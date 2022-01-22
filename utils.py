from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from file_utils import getValues, getTribes, getRuins
from file_utils import MAX_CITY_LEVEL, dic, model, getTechList
from var_utils import addTechs, addVars, addTribes, addScore, addSpecial, addPop, addCity, addStars


def setTribe(turn, tribe):
    model.Add(dic[turn + "_" + tribe] == 1)


def setRawScore(turn, raw_score):
    model.Add(dic[turn + "_" + 'raw_score'] == raw_score)


def setDeltaRawScore(turn, delta_raw_score):
    model.Add(dic[turn + "_" + 'delta_raw_score'] == delta_raw_score)


def setFullScore(turn, full_score):
    model.Add(dic[turn + "_" + 'full_score'] == full_score)


def setDeltaFullScore(turn, delta_full_score):
    model.Add(dic[turn + "_" + 'delta_full_score'] == delta_full_score)


def buildDictionnaries(turn):
    dic.update(addVars(turn))
    dic.update(addTechs(turn))
    dic.update(addTribes(turn))
    dic.update(addScore(turn))
    if turn != "start":
        dic[turn + "_" + 'claimed_cst'] = model.NewBoolVar(turn + "_" + 'claimed_cst')
        dic[turn + "_" + 'lvl_1_cst'] = model.NewBoolVar(turn + "_" + 'lvl_1_cst')
    dic.update(addSpecial(turn))
    dic.update(addCity(turn))
    dic.update(addPop(turn))
    dic.update(addStars(turn))


def addInitialState(turn):
    if turn == "start":
        model.Add(dic[turn + "_" + "claimed"] == 9)  # allow for some params
        model.Add(dic[turn + "_" + "revealed"] == 25)
        model.Add(dic[turn + "_" + "level-1"] == 1)
        model.Add(dic[turn + "_" + "population"] == 0)
        model.Add(dic[turn + "_" + "ruin_stars"] == 0)
        model.Add(dic[turn + "_" + "whale_stars"] == 0)

        model.Add(
            dic[turn + "_" + "stars"] == 3
        )
        # model.Add(
        #     dic[turn + "_" + "spt"] == 4
        # ).OnlyEnforceIf(dic[turn + "_" + 'Luxidoor'])
        # model.Add(
        #     dic[turn + "_" + "spt"] == 2
        # ).OnlyEnforceIf(dic[turn + "_" + 'Luxidoor'].Not())

        for i in range(2, MAX_CITY_LEVEL + 1):
            model.Add(dic[turn + "_" + "level-" + str(i)] == 0)
        for tribe in getTribes():
            k1 = turn + "_" + tribe['name']
            model.Add(dic[turn + "_" + tribe['unit']] == 1
                      ).OnlyEnforceIf(dic[k1])
            for u in getValues()['units']:
                if u != tribe['unit']:
                    model.Add(dic[turn + "_" + u] == 0).OnlyEnforceIf(dic[k1])
                    if u != 'giant':
                        # model.Add(dic[turn + "_trained_" + u] == 0).OnlyEnforceIf(dic[k1])
                        # model.Add(dic[turn + "_killed_" + u] == 0).OnlyEnforceIf(dic[k1])
                        pass
                else:
                    model.Add(dic[turn + "_" + u] == 1).OnlyEnforceIf(dic[k1])
                    if u != 'giant':
                        # model.Add(dic[turn + "_trained_" + u] == 1).OnlyEnforceIf(dic[k1])
                        # model.Add(dic[turn + "_killed_" + u] == 0).OnlyEnforceIf(dic[k1])
                        pass
            for tech_name in getTechList():
                if tech_name != tribe['tech']:
                    model.Add(dic[turn + "_" + tech_name] == 0
                              ).OnlyEnforceIf(dic[k1])
        model.Add(sum(dic[turn + "_" + "ruin" + "_" + e['name']] for e in getRuins()[0]['output']) == 0)


def addSolution(turn):
    model.Add(dic["start_monument"] == 0)
    model.Add(dic["start_level-2_count"] == 0)
    model.Add(dic["start_level-5_count"] == 0)
    model.Add(dic["t0_level-1"] == 1)
    model.Add(dic["t0_level-2"] == 1)
    model.Add(dic["t0_level-3"] == 0)
    model.Add(dic["t0_level-1_count"] == 0)
    model.Add(dic["t0_level-2_count"] == 1)
    model.Add(dic["t0_level-3_count"] == 0)
    model.Add(dic["t0_level-4_count"] == 0)
    model.Add(dic["t0_level-5_count"] == 0)
    model.Add(dic["t0_population"] == 2)
    model.Add(dic["t0_claimed"] == 9)
    model.Add(dic["t0_tier-1"] == 1)
    model.Add(dic["t0_tier-2"] == 0)
    model.Add(dic["t0_monument"] == 0)
    model.Add(dic["t0_giant"] == 0)
    model.Add(dic["t0_warrior"] == 1)
    model.Add(dic["t0_explorer"] == 0)


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


def dropColumns(df):
    res = df
    for col in df.columns:
        if len(df[col].unique()) == 1:
            res = res.drop(col, axis=1)
    return res


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
    for star_column in list(filter(lambda s: 'whale_stars' in s or 'ruin_stars' in s, df.columns)):
        min_star = min(df[star_column])
        df = df[df[star_column] == min_star]
    for ruin_giant_column in list(filter(lambda s: 'ruin_giant' in s, df.columns)):
        min_ruin_giant = min(df[ruin_giant_column])
        df = df[df[ruin_giant_column] == min_ruin_giant]
    return df


def setObjective(turn):
    # min ruin events
    # min tribe meeting
    pass


def printScenario(full_df):
    regex = getScenarioRegex()
    selected_columns = [u for u in full_df.columns if sum(re.search(r, u) is not None for r in regex) != 0]
    # print(selected_columns)
    df = full_df[selected_columns]
    identical_columns = list(filter(lambda x: len(df[x].unique()) == 1, selected_columns))
    different_columns = list(filter(lambda x: len(df[x].unique()) != 1, selected_columns))
    turns = list(set(map(lambda u: u.split("_")[0], selected_columns)))
    for turn_i in sorted(turns):
        turn_string = ("Turn " + turn_i[1:] + ":" if turn_i.startswith("t") else "Start:")
        print(turn_string)
        print("  Common:")
        for element in identical_columns:
            # print(df[element].iloc[0])
            if element.startswith(turn_i) and df[element].iloc[0] != 0:
                print("   " + " ".join(element.split("_")[1:]) + ": " + str(df[element].iloc[0]))
        if sum(e.startswith(turn_i) for e in different_columns) > 0:
            for i, row in df.iterrows():
                print(" Scenario " + str(i) + ":")
                print("  Specific:")
                for element in different_columns:
                    if element.startswith(turn_i):
                        print("   " + " ".join(element.split("_")[1:]) + ": " + str(row[element]))
