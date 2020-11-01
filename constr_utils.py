from file_utils import getValues, getTechs, getTribes, getRuins, getPopulation, getCost
from file_utils import MAX_CITY_LEVEL, dic, model, getTechList, name, getPastTurn, get, diff


# Add Temple score
# Add Garden score


def buildCityConstrains(turn):
    """
    past + village capture + city capture (+) - city capture (-) == current
    """
    for i in range(1, MAX_CITY_LEVEL):
        model.Add(dic[turn + "_" + "level-" + str(i)] >= dic[turn + "_" + "level-" + str(i + 1)])
    model.Add(dic[turn + "_custom house"] == 0).OnlyEnforceIf(dic[turn + "_trade"].Not())


def buildGiantConstraints(turn):
    """
    past + trained giants (city lvl 5+, giant) + ruin giant + convert giant (+) 
        - killed giant - convert giant (-) == current
    """
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


def buildUnitConstraints(turn):
    """
    trained (except giant) <= city lvl 1
    past + trained - kill + converted (+) - converted (-) == current
    units <= sum of (counted + 1) + converted
    """
    values = getValues()
    past_turn = getPastTurn(turn)
    # add conversion
    for u in values['units'].keys():
        if u != 'giant':
            if past_turn is not None:
                model.Add(
                    dic[turn + "_" + u] == (
                        dic[past_turn + "_" + u] +
                        dic[turn + "_trained_" + u] -
                        dic[turn + "_killed_" + u]
                    )
                )
                # model.Add(diff(turn, u) == dic[turn + "_trained_" + u] - dic[turn + "_killed_" + u])
                model.Add(dic[past_turn + "_" + u] >= dic[turn + "_killed_" + u])
            else:
                model.Add(dic[turn + "_trained_" + u] == 0)
                model.Add(dic[turn + "_killed_" + u] == 0)
    model.Add(
        sum(dic[turn + "_trained_" + u] for u in values['units'].keys() if u != 'giant') <= dic[turn + "_level-1"]
    )
    if past_turn is not None:
        model.Add(
            sum(dic[turn + "_" + "level-" + str(i) + "_count"] * (i + 1)
                for i in range(1, MAX_CITY_LEVEL + 1)) >=
            sum(dic[turn + "_" + e] for e in values['units'].keys()))


def buildTechConstraints(turn):
    """
    past + bought + ruin tech + tribe meet tech == current
    tech are for life
    """
    model.Add(
        diff(turn, "ruin" + "_" + "tech") <=
        sum(diff(turn, "tier-" + str(i)) for i in range(1, 4))
    )
    for t in getTechList():
        model.Add(diff(turn, t) >= 0)


def buildTribeConstraints(turn):
    """
    sum(tribes) == 1
    tribe => starting tech
    tribe is cst between turn
    """
    tribes = getTribes()
    for t1 in tribes:
        k1 = turn + "_" + t1['name']
        if t1['tech'] != '':
            model.Add(dic[turn + "_" + t1['tech']] == 1).OnlyEnforceIf(dic[k1])
    model.Add(sum(dic[turn + "_" + t['name']] for t in getTribes()) == 1)
    if getPastTurn(turn) is not None:
        for t in getTribes():
            model.Add(diff(turn, t['name']) == 0)


def buildTechTreeConstraints(turn):
    """
    tech can be purchased only after having bought the previous one
    """
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
    """
    units need the corresponding tech to be trained
    """
    techs = getTechs()
    for t1 in techs:
        k1 = turn + "_" + t1['name']
        for u in t1['units']:
            model.Add(dic[turn + "_" + u] == 0).OnlyEnforceIf(dic[k1].Not())
            model.Add(dic[turn + "_trained_" + u] == 0).OnlyEnforceIf(dic[k1].Not())
        for t2 in t1['allows']:
            k2 = turn + "_" + t2['name']
            for u in t2['units']:
                model.Add(dic[turn + "_" + u] == 0).OnlyEnforceIf(dic[k2].Not())
                model.Add(dic[turn + "_trained_" + u] == 0).OnlyEnforceIf(dic[k2].Not())
            for t3 in t2['allows']:
                k3 = turn + "_" + t3['name']
                for u in t3['units']:
                    model.Add(dic[turn + "_" + u] == 0).OnlyEnforceIf(dic[k3].Not())
                    model.Add(dic[turn + "_trained_" + u] == 0).OnlyEnforceIf(dic[k3].Not())


def linkTechTiersAndTech(turn):
    """
    techs tiers group tech together
    """
    techs = getTechs()
    model.Add(dic[turn + "_" + 'tier-1'] == sum([dic[turn + "_" + t['name']] for t in techs]))
    model.Add(dic[turn + "_" + 'tier-2'] == sum(
        [dic[turn + "_" + t['name']] for u in techs for t in u['allows']]))
    model.Add(dic[turn + "_" + 'tier-3'] == sum(
        [dic[turn + "_" + t['name']] for v in techs for u in v['allows'] for t in u['allows']]))


def buildFullScoreConstraint(turn):
    """
    full score == unit score + city score + tech score + temple score + monument score + population score 
        + captured tile score + vision score
    full score - raw score == vision score
    """
    values = getValues()
    cat_eq_dict = {}
    for category in getValues().keys():
        cat_eq_dict[turn + "_" + category] = sum(
            [dic[turn + "_" + element] * score for element, score in values[category].items()])
    model.Add(sum([cat_eq_dict[cat] for cat in cat_eq_dict.keys()]) == dic[turn + "_" + 'full_score'])
    model.Add(dic[turn + "_" + 'revealed'] * 5 == dic[turn + "_" + 'full_score'] - dic[turn + "_" + 'raw_score'])


def buildSpecialConstraints(turn):
    """
    past + ruin stars + ruin tech + ruin explorer + ruin pop + ruin giant == current
    whale stars == whale
    whales require whaling
    explorer == lvl 2 explorer + ruin explorer
    """
    past_turn = getPastTurn(turn)

    for r in getRuins():
        ruins = [dic[turn + "_" + r['name'] + "_" + s['name']] for s in r['output']]
        model.Add(dic[turn + "_" + r['name']] == sum(ruins))

    model.Add(dic[turn + "_" + "whale"] == 0).OnlyEnforceIf(
        dic[turn + "_" + 'whaling'].Not()
    )

    model.Add(
        dic[turn + "_" + "level-2_explorer"] +
        dic[turn + "_" + "ruin" + "_" + "explorer"] ==
        dic[turn + "_" + "explorer"])

    if past_turn is not None:
        model.Add(
            sum(dic[turn + "_" + "ruin" + "_" + e['name']] for e in getRuins()[0]['output']) <=
            sum(dic[past_turn + "_" + e] for e in getValues()['units'])
        )
    model.Add(
        sum(get(turn, u) for u in getValues()['units']) >=
        diff(turn, "ruin") +
        diff(turn, "whale")
    )


def buildPopulationConstraints(turn):
    """
    min pop as fct of lvl i count
    max pop as fct of lvl i count
    min pop <= pop <= max pop
    past + ruin pop + building pop + lvl 4 pop + tribe pop == current
    lvl i count == cities at level
    """
    # Add meet tribe pop
    model.Add(
        diff(turn, "population") >=
        diff(turn, "ruin_population") +
        diff(turn, "level-4_population")
    )

    for i in range(1, MAX_CITY_LEVEL):
        model.Add(
            dic[turn + "_" + "level-" + str(i) + "_count"] ==
            dic[turn + "_" + "level-" + str(i)] -
            dic[turn + "_" + "level-" + str(i + 1)]
        )
    model.Add(
        dic[turn + "_" + "level-" + str(MAX_CITY_LEVEL) + "_count"] ==
        dic[turn + "_" + "level-" + str(MAX_CITY_LEVEL)]
    )

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


def buildClaimedConstraints(turn):
    """
    lvl 1 cst and lvl 4 cst and city capture (+) == 0 and city capture (-) == 0 -> claimed cst
    """
    model.Add(
        dic[turn + "_claimed"] ==
        9 * dic[turn + "_" + "level-1"] +  # TODO allow for some modularity
        16 * dic[turn + "_" + "level-4_border_growth"]
    )
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
        # x -> y <=> not x or y

        lvl_1_diff = dic[turn + "_" + "level-1"] - dic[past_turn + "_" + "level-1"]
        lvl_4_diff = dic[turn + "_" + "level-4_border_growth"] - dic[past_turn + "_" + "level-4_border_growth"]
        claimed_diff = dic[turn + "_" + "claimed"] - dic[past_turn + "_" + "claimed"]
        lvl_1_cst = dic[turn + "_" + 'lvl_1_cst']
        claimed_cst = dic[turn + "_" + 'claimed_cst']

        model.Add(lvl_4_diff == 0).OnlyEnforceIf(lvl_1_cst)
        model.Add(lvl_1_diff == 0).OnlyEnforceIf(lvl_1_cst)
        model.Add(123 * lvl_1_diff + 456 * lvl_4_diff != 0).OnlyEnforceIf(lvl_1_cst.Not())
        model.Add(claimed_diff == 0).OnlyEnforceIf(claimed_cst)
        model.Add(claimed_diff != 0).OnlyEnforceIf(claimed_cst.Not())
        # model.Add(1 - lvl_1_cst + claimed_cst >= 1)

        # model.Add(lvl_1_diff <= sum(dic[past_turn + "_" + u] for u in getValues()['units'].keys()))


def buildStarConstraints(turn):
    """
    ...
    """
    past_turn = getPastTurn(turn)
    # Add tribe meet stars
    # Add spt == city spt + ch spt
    # Add stars: past + spt + special - spent == current

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
            dic[turn + "_trained_" + u] * c
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
            5 * diff(turn, "level-3-stars") +
            10 * diff(turn, "whale_stars") +
            10 * diff(turn, "ruin_stars")
        )
    else:
        model.Add(
            dic[turn + "_" + "star_income"] ==
            5 * diff(turn, "level-3-stars") +
            10 * diff(turn, "whale_stars") +
            10 * diff(turn, "ruin_stars")
        )

    if past_turn is not None:
        model.Add(
            dic[turn + "_" + "stars"] ==
            dic[past_turn + "_" + "stars"] +
            dic[turn + "_" + "star_income"] -
            dic[turn + "_" + "star_spending"]
        )

    model.Add(
        dic[turn + "_" + "unit_stars"] ==
        sum(diff(turn, u + "_" + "stars") for u, c in getCost()['units'].items())
    )
    model.Add(
        dic[turn + "_" + "tech_stars"] ==
        sum(diff(turn, t + "_" + "cost") for t, c in getCost()['techs'].items())
    )
    model.Add(
        dic[turn + "_" + "pop_stars"] ==
        sum(diff(turn, p['name'] + "_" + "stars") for p in getPopulation())
    )
    # Add boat_upgrade, add ch
    model.Add(
        dic[turn + "_" + "star_spending"] ==
        dic[turn + "_" + "unit_stars"] +
        dic[turn + "_" + "tech_stars"] +
        dic[turn + "_" + "pop_stars"]
    )

    model.Add(
        dic[turn + "_" + "city_spt"] ==
        1 +  # for cap; TODO change if cap lost
        sum(dic[turn + "_level-" + str(i)] for i in range(1, MAX_CITY_LEVEL + 1))
    )
    model.Add(
        dic[turn + "_" + "ch_spt"] <=
        2 * 8 * dic[turn + "_custom house"]
    )
    model.Add(
        dic[turn + "_" + "spt"] ==
        dic[turn + "_" + "ch_spt"] +
        dic[turn + "_" + "city_spt"] +
        dic[turn + "_level-2_spt"]
    )


def buildTechPopConstraints(turn):
    """
    pop can be built only after having bought the corresponding tech
    pop for sawmill, windmill and forge depend on farms, lumber houses and mines
    """
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
        3 * dic[turn + "_" + "ruin_population"]
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


def buildCityUpgradeConstraints(turn):

    # diff city level with previous turn + capture == consequences for each level
    # - level-2: spt or explorer
    # - level-3: stars or wall
    # - level-4: border growth or pop
    # - level-5+: giant or garden
    model.Add(dic[turn + "_" + "level-2"] ==
              (dic[turn + "_" + "level-2_spt"] +
              dic[turn + "_" + "level-2_explorer"]))

    model.Add(
        dic[turn + "_" + "level-3"] ==
        dic[turn + "_" + "level-3-stars"] +
        dic[turn + "_" + "level-3-walls"]
    )

    model.Add(dic[turn + "_" + "level-4"] ==
              dic[turn + "_" + "level-4_border_growth"] +
              dic[turn + "_" + "level-4_population"])

    for i in range(5, MAX_CITY_LEVEL + 1):
        model.Add(dic[turn + "_" + "level-" + str(i)] ==
                  dic[turn + "_" + "level-" + str(i) + "_giant"] +
                  dic[turn + "_" + "level-" + str(i) + "_garden"])


def buildAllConstraints(turn):
    buildFullScoreConstraint(turn)
    buildTechUnitsConstraints(turn)
    buildTechTreeConstraints(turn)
    buildCityConstrains(turn)
    buildTribeConstraints(turn)
    linkTechTiersAndTech(turn)
    buildClaimedConstraints(turn)
    buildCityUpgradeConstraints(turn)
    buildGiantConstraints(turn)
    buildSpecialConstraints(turn)
    buildPopulationConstraints(turn)
    buildTechPopConstraints(turn)
    buildTechConstraints(turn)
    buildUnitConstraints(turn)

    buildStarConstraints(turn)

    for v in ['revealed']:
        model.Add(diff(turn, v) >= 0)
    pass
