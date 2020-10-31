from file_utils import getValues, getTechs, getTribes, getRuins, getPopulation, getCost
from file_utils import MAX_CITY_LEVEL, dic, model, getTechList, name, getPastTurn


def buildCityConstrains(turn):
    for i in range(1, MAX_CITY_LEVEL):
        model.Add(dic[turn + "_" + "level-" + str(i)] >= dic[turn + "_" + "level-" + str(i + 1)])


def buildGiantConstraints(turn):
    # Add conversion
    model.Add(
        dic[turn + "_" + "giant"] ==
        sum(dic[turn + "_" + "level-" + str(i) + "_giant"]
            for i in range(5, MAX_CITY_LEVEL + 1)) +
        dic[turn + "_" + "ruin_giant"])
    # giants >= city upgrade (lvl 5+) + ruin_giant + convert_giant


def buildUnitTrainByCityConstraint(turn):
    # Add conversion, giant on lvl_up, ruin
    past_turn = getPastTurn(turn)
    if past_turn is not None:
        model.Add(
            dic[turn + "_" + 'level-1'] +
            dic[turn + "_" + 'ruin' + "_" + "giant"] +
            sum(dic[turn + "_" + "level-" + str(i) + "_giant"] -
                dic[past_turn + "_" + "level-" + str(i) + "_giant"]
                for i in range(5, MAX_CITY_LEVEL + 1)) >= (
                sum(dic[turn + "_" + u] for u in getValues()['units'].keys()) -
                sum(dic[past_turn + "_" + u] for u in getValues()['units'].keys())
            )
        )


def buildTribeConstraints(turn):
    tribes = getTribes()
    for t1 in tribes:
        k1 = turn + "_" + t1['name']
        if t1['tech'] != '':
            model.Add(dic[turn + "_" + t1['tech']] == 1).OnlyEnforceIf(dic[k1])
    model.Add(sum(dic[turn + "_" + t['name']] for t in getTribes()) == 1)


def buildTechTreeConstraints(turn):
    techs = getTechs()
    for t1 in techs:
        k1 = turn + "_" + t1['name']
        for t2 in t1['allows']:
            k2 = turn + "_" + t2['name']
            model.Add(dic[k2] == 0).OnlyEnforceIf(dic[k1].Not())
            for t3 in t2['allows']:
                k3 = turn + "_" + t3['name']
                model.Add(dic[k3] == 0).OnlyEnforceIf(dic[k2].Not())


def buildTechUnitsConstraints(turn):
    techs = getTechs()
    for t1 in techs:
        k1 = turn + "_" + t1['name']
        for u in t1['units']:
            model.Add(dic[turn + "_" + u] == 0).OnlyEnforceIf(dic[k1].Not())
        for t2 in t1['allows']:
            k2 = turn + "_" + t2['name']
            for u in t2['units']:
                model.Add(dic[turn + "_" + u] == 0).OnlyEnforceIf(dic[k2].Not())
            for t3 in t2['allows']:
                k3 = turn + "_" + t3['name']
                for u in t3['units']:
                    model.Add(dic[turn + "_" + u] == 0).OnlyEnforceIf(dic[k3].Not())


def linkTechAndTech(turn):
    techs = getTechs()
    model.Add(dic[turn + "_" + 'tier-1'] == sum([dic[turn + "_" + t['name']] for t in techs]))
    model.Add(dic[turn + "_" + 'tier-2'] == sum(
        [dic[turn + "_" + t['name']] for u in techs for t in u['allows']]))
    model.Add(dic[turn + "_" + 'tier-3'] == sum(
        [dic[turn + "_" + t['name']] for v in techs for u in v['allows'] for t in u['allows']]))


def buildFullScoreConstraint(turn):
    values = getValues()
    cat_eq_dict = {}
    for category in getValues().keys():
        cat_eq_dict[turn + "_" + category] = sum(
            [dic[turn + "_" + element] * score for element, score in values[category].items()])
    model.Add(sum([cat_eq_dict[cat] for cat in cat_eq_dict.keys()]) == dic[turn + "_" + 'full_score'])


def buildClaimingCityConstraint(turn):
    # event to other tribe
    pass


def buildMaxUnitConstraint(turn):
    values = getValues()
    past_turn = getPastTurn(turn)
    # add conversion
    if past_turn is not None:
        model.Add(
            sum(dic[turn + "_" + "level-" + str(i) + "_count"] * (i + 1)
                for i in range(1, MAX_CITY_LEVEL + 1)) >=
            sum(dic[turn + "_" + e] for e in values['units'].keys()))


def buildVisionConstraint(turn):
    model.Add(dic[turn + "_" + 'revealed'] * 5 ==
              (dic[turn + "_" + 'full_score'] - dic[turn + "_" + 'raw_score']))


def buildSpecialConstraints(turn):
    for r in getRuins():
        ruins = [dic[turn + "_" + r['name'] + "_" + s['name']] for s in r['output']]
        model.Add(dic[turn + "_" + r['name']] == sum(ruins))


def buildPopulationConstraints(turn):
    # Add meet tribe pop
    past_turn = getPastTurn(turn)
    model.Add(
        dic[turn + "_" + 'population'] - (
            dic[past_turn + "_" + 'population'] if past_turn is not None else 0
        ) >= dic[turn + "_" + "ruin" + "_" + "population"] +
        dic[turn + "_" + "level-4_population"] - ((
            dic[past_turn + "_" + "level-4_population"] +
            dic[past_turn + "_" + "ruin" + "_" + "population"]
        ) if past_turn is not None else 0))
    # pop >= ruin_pop + lvl-4_pop + tribe_meet_pop + ...


def buildClaimedConstraints(turn):
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

        lvl_1_diff = (dic[turn + "_" + "level-1"] -
                      dic[past_turn + "_" + "level-1"])

        lvl_4_diff = (dic[turn + "_" + "level-4_border_growth"] -
                      dic[past_turn + "_" + "level-4_border_growth"])

        claimed_diff = (dic[turn + "_" + "claimed"] -
                        dic[past_turn + "_" + "claimed"])

        lvl_1_cst = dic[turn + "_" + 'lvl_1_cst']
        claimed_cst = dic[turn + "_" + 'claimed_cst']

        model.Add((lvl_1_diff + lvl_4_diff) != 0).OnlyEnforceIf(lvl_1_cst)
        model.Add((lvl_1_diff == 0) and (lvl_4_diff == 0)).OnlyEnforceIf(lvl_1_cst.Not())
        model.Add(claimed_diff == 0).OnlyEnforceIf(claimed_cst)
        model.Add(claimed_diff != 0).OnlyEnforceIf(claimed_cst.Not())
        model.Add(lvl_1_cst + claimed_cst >= 1)

        model.Add(lvl_1_diff <= sum(
            dic[past_turn + "_" + u] for u in getValues()['units'].keys()))


def buildCityPopConstraints(turn):
    if turn != "start":
        for i in range(1, MAX_CITY_LEVEL):
            model.Add((dic[turn + "_" + "level-" + str(i) + "_count"] +
                       dic[turn + "_" + "level-" + str(i + 1)]) ==
                      dic[turn + "_" + "level-" + str(i)])
        model.Add(dic[turn + "_" + "level-" + str(MAX_CITY_LEVEL) + "_count"] ==
                  dic[turn + "_" + "level-" + str(MAX_CITY_LEVEL)])

    model.Add(
        dic[turn + "_" + "pop_min"] ==
        sum(
            dic[turn + "_" + "level-" + str(i) + "_count"] *
            int(i * (i + 1) / 2 - 1) for i in range(1, MAX_CITY_LEVEL + 1)
        )
    )
    model.Add(
        dic[turn + "_" + "population"] >=
        dic[turn + "_" + "pop_min"]
    )

    model.Add(
        dic[turn + "_" + "pop_max"] ==
        sum(
            dic[turn + "_" + "level-" + str(i) + "_count"] *
            int((i + 1) * (i + 2) / 2 - 2) for i in range(1, MAX_CITY_LEVEL + 1)
        )
    )
    model.Add(
        dic[turn + "_" + "population"] <=
        dic[turn + "_" + "pop_max"]
    )


def buildRuinLinkConstraints(turn):
    past_turn = getPastTurn(turn)
    if past_turn is not None:
        model.Add(
            (
                dic[turn + "_" + "ruin" + "_" + "tech"] -
                dic[past_turn + "_" + "ruin" + "_" + "tech"]
            ) <= (
                sum(dic[turn + "_" + "tier-" + str(i)] for i in range(1, 4)) -
                sum(dic[past_turn + "_" + "tier-" + str(i)] for i in range(1, 4))
            )
        )
        model.Add(
            3 * (
                dic[turn + "_" + "ruin" + "_" + "population"] -
                dic[past_turn + "_" + "ruin" + "_" + "population"]
            ) <= (
                dic[turn + "_" + "population"] -
                dic[past_turn + "_" + "population"]
            )
        )
        model.Add(
            sum(
                dic[turn + "_" + "ruin" + "_" + e['name']] for e in getRuins()[0]['output']
            ) <= sum(
                dic[past_turn + "_" + e] for e in getValues()['units']
            )
        )
    else:
        model.Add(
            dic[turn + "_" + "ruin" + "_" + "tech"] <=
            sum(dic[turn + "_" + "tier-" + str(i)] for i in range(1, 4))
        )
        model.Add(
            3 * dic[turn + "_" + "ruin" + "_" + "population"] <=
            dic[turn + "_" + "population"]
        )
        model.Add(
            sum(
                dic[turn + "_" + "ruin" + "_" + e['name']] for e in getRuins()[0]['output']
            ) == 0
        )
    model.Add(dic[turn + "_" + "whale"] == 0).OnlyEnforceIf(
        dic[turn + "_" + 'whaling'].Not()
    )


def buildStarConstraints(turn):
    past_turn = getPastTurn(turn)
    # Add tribe meet stars

    for t, c in getCost()['techs'].items():
        model.Add(
            dic[turn + "_" + t + "_" + "stars"] ==
            (4 + c * dic[turn + "_" + "level-1"])
        )
        model.AddProdEquality(dic[turn + "_" + t + "_" + "cost"], [
            dic[turn + "_" + t],
            dic[turn + "_" + t + "_" + "stars"]
        ])
    for u, c in getCost()['units'].items():
        model.Add(
            dic[turn + "_" + u + "_" + "stars"] ==
            dic[turn + "_" + u] * c
        )
    for p in getPopulation():
        model.Add(
            dic[turn + "_" + p['name'] + "_" + "stars"] ==
            dic[turn + "_" + p['name']] * p['cost']
        )
    if past_turn is not None:
        model.Add(
            dic[turn + "_" + "star_income"] ==
            dic[past_turn + "_" + "spt"] +
            5 * dic[turn + "_" + "level-3-stars"] +
            10 * dic[turn + "_" + "whale_stars"] +
            10 * dic[turn + "_" + "ruin_stars"] -
            5 * dic[past_turn + "_" + "level-3-stars"] -
            10 * dic[past_turn + "_" + "whale_stars"] -
            10 * dic[past_turn + "_" + "ruin_stars"]
        )

        # model.Add(
        #     dic[turn + "_" + "stars"] >=
        #     dic[turn + "_" + "star_income"]
        # )

        # Add boat_upgrade
        model.Add(
            dic[turn + "_" + "star_spending"] ==
            sum(dic[turn + "_" + u + "_" + "stars"] for u, c in getCost()['units'].items()) +
            sum(dic[turn + "_" + t + "_" + "cost"] for t, c in getCost()['techs'].items()) +
            sum(dic[turn + "_" + p['name'] + "_" + "stars"] for p in getPopulation()) -
            sum(dic[past_turn + "_" + u + "_" + "stars"] for u, c in getCost()['units'].items()) -
            sum(dic[past_turn + "_" + t + "_" + "cost"]
                for t, c in getCost()['techs'].items()) -
            sum(dic[past_turn + "_" + p['name'] + "_" + "stars"]
                for p in getPopulation())
        )
    else:
        model.Add(
            dic[turn + "_" + "star_income"] ==
            5 * dic[turn + "_" + "level-3-stars"] +
            10 * dic[turn + "_" + "whale_stars"] +
            10 * dic[turn + "_" + "ruin_stars"]
        )
        # model.Add(
        #     dic[turn + "_" + "stars"] >=
        #     dic[turn + "_" + "star_income"]
        # )

        # Add boat_upgrade
        model.Add(
            dic[turn + "_" + "star_spending"] ==
            sum(dic[turn + "_" + t + "_" + "cost"] for t, c in getCost()['techs'].items()) +
            sum(dic[turn + "_" + p['name'] + "_" + "stars"] for p in getPopulation())
        )


# Add Temple score adding


def buildTechPopConstraints(turn):
    pop = getPopulation()
    for p in pop:
        model.Add(
            dic[turn + "_" + p['name']] == 0
        ).OnlyEnforceIf(
            dic[turn + "_" + p["requires"]].Not()
        )
        if type(p['pop']) == str:
            model.AddProdEquality(
                dic[turn + "_" + p['name'] + "_" + p['pop']], [
                    dic[turn + "_" + p['pop']],
                    dic[turn + "_" + p['name']]
                ]
            )

    model.Add(
        dic[turn + "_" + "population"] >=
        sum(dic[turn + "_" + p['name']] *
            p['pop'] for p in pop if type(p['pop']) == int) +
        dic[turn + "_" + "level-4_population"] +
        dic[turn + "_" + "ruin_population"]
    )

    model.Add(
        dic[turn + "_" + "population"] <=
        sum(dic[turn + "_" + p['name']] *
            p['pop'] for p in pop if type(p['pop']) == int) +
        sum(
            dic[turn + "_" + p['name'] + "_" + p['pop']]
            for p in pop if type(p['pop']) == str) +
        dic[turn + "_" + "level-4_population"] +
        dic[turn + "_" + "ruin_population"]
    )


def linkTurns(turn):
    past_turn = getPastTurn(turn)
    if past_turn is not None:
        for t in getTechList():
            model.Add(dic[name(past_turn, t)] <= dic[name(past_turn, t)])
        for t in getTribes():
            model.Add(dic[name(past_turn, t['name'])] <= dic[name(past_turn, t['name'])])
        for v in ['revealed']:
            model.Add(dic[name(past_turn, v)] <= dic[name(past_turn, v)])
        # Add unit: past + trained - killed == current
        # Add cities: past  + captured - lost == current
        # Add stars: past + spt + special - spent == current


def buildCityUpgradeConstraints(turn):
    # diff city level with previous turn + capture == consequences for each level
    # - level-2: spt or explorer
    # - level-3: stars or nothing, so >=
    # - level-4: border growth or pop
    # - level-5+: giant or garden
    model.Add(dic[turn + "_" + "level-2"] ==
              (dic[turn + "_" + "level-2_spt"] +
              dic[turn + "_" + "level-2_explorer"]))

    # Add Ruin explorer
    model.Add(
        dic[turn + "_" + "level-2_explorer"] +
        dic[turn + "_" + "ruin" + "_" + "explorer"] ==
        dic[turn + "_" + "explorer"])

    # Add level-3 stars
    model.Add(
        dic[turn + "_" + "level-3"] >=
        dic[turn + "_" + "level-3-stars"]
    )

    model.Add(dic[turn + "_" + "level-4"] ==
              dic[turn + "_" + "level-4_border_growth"] +
              dic[turn + "_" + "level-4_population"])

    for i in range(5, MAX_CITY_LEVEL + 1):
        model.Add(dic[turn + "_" + "level-" + str(i)] ==
                  dic[turn + "_" + "level-" + str(i) + "_giant"] +
                  dic[turn + "_" + "level-" + str(i) + "_garden"])


def buildAllConstraints(turn):
    buildVisionConstraint(turn)
    buildFullScoreConstraint(turn)
    buildTechUnitsConstraints(turn)
    buildTechTreeConstraints(turn)
    buildCityConstrains(turn)
    buildTribeConstraints(turn)
    linkTechAndTech(turn)
    buildCityPopConstraints(turn)
    buildUnitTrainByCityConstraint(turn)
    buildClaimedConstraints(turn)
    buildCityUpgradeConstraints(turn)
    buildRuinLinkConstraints(turn)
    buildGiantConstraints(turn)
    buildMaxUnitConstraint(turn)
    buildSpecialConstraints(turn)
    buildPopulationConstraints(turn)
    buildTechPopConstraints(turn)
    # buildStarConstraints(turn)

    linkTurns(turn)
