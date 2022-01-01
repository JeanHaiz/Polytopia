from python.action_solver.file_utils import getMoves, getValues, getTechs, getTribes, getRuins, getPopulation, getCost
from python.action_solver.file_utils import MAX_CITY_LEVEL, dic, model, getTechList, getPastTurn, get, diff, name


# Add Temple score
# Add Garden score
killed = {}


def buildCityConstrains(turn):
    """
    # past + village capture + city capture (+) - city capture (-) == current
    no trade -> no custom house
    for all i: there is allways more or equal cities of lvl i than cities of lvl i + 1
    """
    # TODO add first equation
    for i in range(1, MAX_CITY_LEVEL):
        model.Add(get(turn, "level-" + str(i)) >= get(turn, "level-" + str(i + 1)))
    model.Add(get(turn, "custom house") == 0).OnlyEnforceIf(get(turn, "trade").Not())


def buildGiantConstraints(turn):
    """
    past + trained giants (city lvl 5+, giant) + ruin giant + convert giant (+)
        - killed giant - convert giant (-) == current
    """
    # TODO add conversion
    past_turn = getPastTurn(turn)
    if past_turn is not None:
        model.Add(
            get(turn, 'ruin' + "_" + "giant") +   # ruin giants
            sum(
                diff(turn, "level-" + str(i) + "_giant") for i in range(5, MAX_CITY_LEVEL + 1)  # trained giants
            ) == diff(turn, "giant")  # current - past giants
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
                # TODO add conversion
                model.Add(diff(turn, u) == diff(turn, "trained_" + u) - diff(turn, "killed_" + u))
                model.Add(get(past_turn, u) >= get(turn, "killed_" + u))  # !!!
            else:
                model.Add(diff(turn, "trained_" + u) == 0)
                model.Add(diff(turn, "killed_" + u) == 0)

    model.Add(  # at most 1 unit trained by city
        sum(diff(turn, "trained_" + u) for u in values['units'].keys() if u != 'giant') <=
        get(turn, "level-1") - diff(turn, "level-1"))

    if past_turn is not None:
        # no more than 5 giants can be killed per participant each turn
        model.Add(diff(turn, "killed_giant") == 0)  # !!!

        model.Add(  # unit capacity per city # !!! except for conversion
            sum(get(turn, "level-" + str(i) + "_count") * (i + 1)
                for i in range(1, MAX_CITY_LEVEL + 1)) >=
            sum(get(turn, u) for u in values['units'].keys()))


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
        k1 = get(turn, t1['name'])
        if t1['tech'] != '':
            model.Add(get(turn, t1['tech']) == 1).OnlyEnforceIf(k1)
    model.Add(sum(get(turn, t['name']) for t in getTribes()) == 1)
    if getPastTurn(turn) is not None:
        for t in getTribes():
            model.Add(diff(turn, t['name']) == 0)


def buildTechTreeConstraints(turn):
    """
    tech can be purchased only after having bought the previous one
    """
    techs = getTechs()
    for t1 in techs:
        k1 = get(turn, t1['name'])
        for t2 in t1['allows']:
            k2 = get(turn, t2['name'])
            model.Add(k2 == 0).OnlyEnforceIf(k1.Not())
            for t3 in t2['allows']:
                model.Add(get(turn, t3['name']) == 0).OnlyEnforceIf(k2.Not())


def buildTechUnitsConstraints(turn):
    """
    units need the corresponding tech to be trained
    """
    techs = getTechs()
    for t1 in techs:
        k1 = get(turn, t1['name'])
        for u in t1['units']:
            model.Add(get(turn, u) == 0).OnlyEnforceIf(k1.Not())
            model.Add(get(turn, "trained_" + u) == 0).OnlyEnforceIf(k1.Not())
        for t2 in t1['allows']:
            k2 = get(turn, t2['name'])
            for u in t2['units']:
                model.Add(get(turn, u) == 0).OnlyEnforceIf(k2.Not())
                model.Add(get(turn, "trained_" + u) == 0).OnlyEnforceIf(k2.Not())
            for t3 in t2['allows']:
                k3 = get(turn, t3['name'])
                for u in t3['units']:
                    model.Add(get(turn, u) == 0).OnlyEnforceIf(k3.Not())
                    model.Add(get(turn, "trained_" + u) == 0).OnlyEnforceIf(k3.Not())


def linkTechTiersAndTech(turn):
    """
    techs tiers group tech together
    """
    techs = getTechs()
    model.Add(get(turn, 'tier-1') == sum([get(turn, t['name']) for t in techs]))
    model.Add(get(turn, 'tier-2') == sum(
        [get(turn, t['name']) for u in techs for t in u['allows']]))
    model.Add(get(turn, 'tier-3') == sum(
        [get(turn, t['name']) for v in techs for u in v['allows'] for t in u['allows']]))


def buildMovesConstraints(turn):
    """
    moves == unit move + rider bump unit attack + tech purchase + monument placement + road placement
        + population growth + city upgrades + city caputres + unit training
    """
    past_turn = getPastTurn(turn)
    if past_turn is not None:
        model.Add(get(turn, "delta_moves") == sum(diff(turn, "move_" + m.replace(" ", "_")) for m in getMoves()))
        model.Add(
            diff(turn, "move_move") <=
            sum(get(turn, u) for u, c in getValues()['units'].items()) -
            diff(turn, "move_city_capture"))
        model.Add(  # !!! maybe missing giant
            diff(turn, "move_training") ==
            sum(diff(turn, "trained_" + u) for u, c in getValues()['units'].items() if u != "giant"))
        model.Add(diff(turn, "move_rider_bump") <= get(turn, "rider"))
        model.Add(
            diff(turn, "move_attack") <= sum(diff(turn, u) for u, c in getValues()['units'].items()))  # !!! giants
        model.Add(
            diff(turn, "move_knight_chain") == 0).OnlyEnforceIf(get(turn, "chivalry").Not())  # !!! could be stricter
        model.Add(diff(turn, "move_tech_purchase") <= sum(diff(turn, "tier-" + str(i)) for i in range(1, 4)))  # !!!
        # TODO add together reasons for tech increase
        # TODO add together reasons for population increase
        model.Add(get(turn, "move_road_building") == get(turn, "roads"))
        model.Add(diff(turn, "move_population") <= diff(turn, "population"))
        model.Add(
            diff(turn, "move_city_upgrade") ==
            sum(diff(turn, "level-" + str(i + 1)) for i in range(MAX_CITY_LEVEL)))
        model.Add(get(turn, "move_city_capture") == 0)
        for m in getMoves():
            model.Add(diff(turn, "move_" + m.replace(" ", "_")) >= 0)


def buildFullScoreConstraint(turn):
    """
    full score == unit score + city score + tech score + temple score + monument score + population score
        + captured tile score + vision score
    """
    values = getValues()
    cat_eq_dict = {}
    for category in getValues().keys():
        cat_eq_dict[turn + "_" + category] = sum(
            [get(turn, element) * score for element, score in values[category].items()])
    model.Add(sum([cat_eq_dict[cat] for cat in cat_eq_dict.keys()]) == get(turn, 'full_score'))


def buildSpecialConstraints(turn):
    """
    past + ruin stars + ruin tech + ruin explorer + ruin pop + ruin giant == current
    whale stars == whale
    whales require whaling
    explorer == lvl 2 explorer + ruin explorer
    """
    past_turn = getPastTurn(turn)

    for r in getRuins():
        ruins = [get(turn, r['name'] + "_" + s['name']) for s in r['output']]
        model.Add(get(turn, r['name']) == sum(ruins))

    model.Add(get(turn, "whale") == 0).OnlyEnforceIf(
        get(turn, 'whaling').Not()
    )

    model.Add(
        get(turn, "level-2_explorer") +
        get(turn, "ruin" + "_" + "explorer") ==
        get(turn, "explorer"))

    if past_turn is not None:
        model.Add(  # !!!
            sum(get(turn, "ruin" + "_" + e['name']) for e in getRuins()[0]['output']) <=
            sum(get(past_turn, e) for e in getValues()['units'])
        )
    model.Add(  # !!!
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
            get(turn, "level-" + str(i) + "_count") ==
            get(turn, "level-" + str(i)) -
            get(turn, "level-" + str(i + 1))
        )
    model.Add(
        get(turn, "level-" + str(MAX_CITY_LEVEL) + "_count") ==
        get(turn, "level-" + str(MAX_CITY_LEVEL))
    )

    model.Add(
        get(turn, "pop_min") ==
        sum(
            get(turn, "level-" + str(i) + "_count") *
            int(i * (i + 1) / 2 - 1) for i in range(1, MAX_CITY_LEVEL + 1)
        )
    )
    model.Add(
        get(turn, "population") >=
        get(turn, "pop_min")
    )

    model.Add(
        get(turn, "pop_max") ==
        sum(
            get(turn, "level-" + str(i) + "_count") *
            int((i + 1) * (i + 2) / 2 - 2) for i in range(1, MAX_CITY_LEVEL + 1)
        )
    )
    model.Add(
        get(turn, "population") <=
        get(turn, "pop_max")
    )


def buildClaimedConstraints(turn):
    """
    lvl 1 cst and lvl 4 cst and city capture (+) == 0 and city capture (-) == 0 -> claimed cst
    """
    model.Add(
        get(turn, "claimed") ==
        9 * get(turn, "level-1") +
        16 * get(turn, "level-4_border_growth")  # TODO allow for some modularity
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

        lvl_1_diff = diff(turn, "level-1")
        lvl_4_diff = diff(turn, "level-4_border_growth")
        claimed_diff = diff(turn, "claimed")
        lvl_1_cst = get(turn, 'lvl_1_cst')
        claimed_cst = get(turn, 'claimed_cst')

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
            get(turn, t + "_" + "stars") ==
            (4 + c * get(turn, "level-1"))
        )
        model.AddProdEquality(get(turn, t + "_" + "cost"), [
            get(turn, t),
            get(turn, t + "_" + "stars")
        ])
    for u, c in getCost()['units'].items():
        model.Add(
            get(turn, u + "_" + "stars") ==
            dic[turn + "_trained_" + u] * c
        )
    for p in getPopulation():
        model.Add(
            get(turn, p['name'] + "_" + "stars") ==
            get(turn, p['name']) * p['cost']
        )

    if past_turn is not None:
        model.Add(
            get(turn, "star_income") ==
            dic[past_turn + "_" + "spt"] +
            5 * diff(turn, "level-3-stars") +
            10 * diff(turn, "whale_stars") +
            10 * diff(turn, "ruin_stars")
        )
    else:
        model.Add(
            get(turn, "star_income") ==
            5 * diff(turn, "level-3-stars") +
            10 * diff(turn, "whale_stars") +
            10 * diff(turn, "ruin_stars")
        )

    if past_turn is not None:
        model.Add(
            get(turn, "stars") ==
            dic[past_turn + "_" + "stars"] +
            get(turn, "star_income") -
            get(turn, "star_spending")
        )

    model.Add(
        get(turn, "unit_stars") ==
        sum(diff(turn, u + "_" + "stars") for u, c in getCost()['units'].items())
    )
    model.Add(
        get(turn, "tech_stars") ==
        sum(diff(turn, t + "_" + "cost") for t, c in getCost()['techs'].items())
    )
    model.Add(
        get(turn, "pop_stars") ==
        sum(diff(turn, p['name'] + "_" + "stars") for p in getPopulation())
    )
    # Add boat_upgrade, add ch
    model.Add(
        get(turn, "star_spending") ==
        get(turn, "unit_stars") +
        get(turn, "tech_stars") +
        get(turn, "pop_stars")
    )

    model.Add(
        get(turn, "city_spt") ==
        1 +  # for cap; TODO change if cap lost
        sum(dic[turn + "_level-" + str(i)] for i in range(1, MAX_CITY_LEVEL + 1))
    )
    model.Add(
        get(turn, "ch_spt") <=
        2 * 8 * dic[turn + "_custom house"]
    )
    model.Add(
        get(turn, "spt") ==
        get(turn, "ch_spt") +
        get(turn, "city_spt") +
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
            get(turn, p['name']) == 0
        ).OnlyEnforceIf(
            get(turn, p["requires"]).Not()
        )
        if type(p['pop']) == str:
            model.AddProdEquality(
                get(turn, p['name'] + "_" + p['pop']), [
                    get(turn, p['pop']),
                    get(turn, p['name'])
                ]
            )

    model.Add(
        get(turn, "population") >=
        sum(get(turn, p['name']) *
            p['pop'] for p in pop if type(p['pop']) == int) +
        get(turn, "level-4_population") +
        3 * get(turn, "ruin_population")
    )

    model.Add(
        get(turn, "population") <=
        sum(get(turn, p['name']) *
            p['pop'] for p in pop if type(p['pop']) == int) +
        sum(
            get(turn, p['name'] + "_" + p['pop'])
            for p in pop if type(p['pop']) == str) +
        get(turn, "level-4_population") +
        get(turn, "ruin_population")
    )


def buildKilledConstraints(turn):
    for u in getValues()['units']:
        if name(turn, u) in killed:
            model.Add(dic[name(turn, "killed_" + u)] == killed[name(turn, u)])
        else:
            model.Add(dic[name(turn, "killed_" + u)] == 0)


def buildCityUpgradeConstraints(turn):

    # diff city level with previous turn + capture == consequences for each level
    # - level-2: spt or explorer
    # - level-3: stars or wall
    # - level-4: border growth or pop
    # - level-5+: giant or garden
    model.Add(get(turn, "level-2") ==
              (get(turn, "level-2_spt") +
              get(turn, "level-2_explorer")))

    model.Add(
        get(turn, "level-3") ==
        get(turn, "level-3-stars") +
        get(turn, "level-3-walls")
    )

    model.Add(get(turn, "level-4") ==
              get(turn, "level-4_border_growth") +
              get(turn, "level-4_population"))

    for i in range(5, MAX_CITY_LEVEL + 1):
        model.Add(get(turn, "level-" + str(i)) ==
                  get(turn, "level-" + str(i) + "_giant") +
                  get(turn, "level-" + str(i) + "_garden"))


def buildAllConstraints(turn, stars, moves):
    buildFullScoreConstraint(turn)
    buildTechUnitsConstraints(turn)
    buildTechTreeConstraints(turn)
    buildCityConstrains(turn)
    buildTribeConstraints(turn)
    linkTechTiersAndTech(turn)
    buildClaimedConstraints(turn)  #
    buildCityUpgradeConstraints(turn)
    buildGiantConstraints(turn)
    buildSpecialConstraints(turn)
    buildPopulationConstraints(turn)
    buildTechPopConstraints(turn)
    buildTechConstraints(turn)
    buildUnitConstraints(turn)
    if stars:
        buildStarConstraints(turn)

    buildKilledConstraints(turn)

    if moves:
        buildMovesConstraints(turn)

    for v in ['revealed']:
        model.Add(diff(turn, v) >= 0)


def setKilled(turn, unit, number):
    killed[name(turn, unit)] = number
