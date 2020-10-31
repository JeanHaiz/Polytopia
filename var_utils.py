from file_utils import getValues, getTechs, getTribes, getRuins, getPopulation, getCost, MAX_CITY_LEVEL, model, name


def addVars(turn):
    values = getValues()
    sym_dict = {}
    for category in values.keys():
        for element, score in values[category].items():
            var = name(turn, element)
            # replace 10 with max value of elements, static or dynamic
            sym_dict[var] = model.NewIntVar(0, 50, var)
    return sym_dict


def addTechs(turn):
    techs = getTechs()
    tech_dict = {}
    for t1 in techs:
        k1 = name(turn, t1['name'])
        tech_dict[k1] = model.NewBoolVar(k1)
        for t2 in t1['allows']:
            k2 = turn + "_" + t2['name']
            tech_dict[k2] = model.NewBoolVar(k2)
            for t3 in t2['allows']:
                k3 = turn + "_" + t3['name']
                tech_dict[k3] = model.NewBoolVar(k3)
    return tech_dict


def addTribes(turn):
    tribes = getTribes()
    tribe_dict = {}
    for t1 in tribes:
        k1 = turn + "_" + t1['name']
        tribe_dict[k1] = model.NewBoolVar(k1)
    return tribe_dict


def addScore(turn):
    score_dict = {
        name(turn, "raw_score"): model.NewIntVar(0, 200000, turn + "_" + "raw_score"),
        # turn + "_" + "delta_raw_score" : model.NewIntVar(0, 200000, turn + "_" + "delta_raw_score"),
        turn + "_" + "full_score": model.NewIntVar(0, 200000, turn + "_" + "full_score"),
        # turn + "_" + "delta_full_score" : model.NewIntVar(0, 200000, turn + "_" + "delta_full_score"),
        # turn + "_" + "computed_full_score": model.NewIntVar(0, 200000, turn + "_" + "computed_full_score")
        # turn + "_" + "computed_raw_score": model.NewIntVar(0, 200000, turn + "_" + "computed_raw_score"),
    }
    return score_dict


def addSpecial(turn):
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


def addCity(turn):
    city = {}
    city[turn + "_" + "level-2_spt"] = model.NewIntVar(0, 20, turn + "_" + "level-2_spt")
    city[turn + "_" + "level-2_explorer"] = model.NewIntVar(0, 20, turn + "_" + "level-2_explorer")
    city[turn + "_" + "level-3-stars"] = model.NewIntVar(0, 20, turn + "_" + "level-3-stars")
    city[turn + "_" + "level-4_border_growth"] = model.NewIntVar(0, 20, turn + "_" + "level-4_border_growth")
    city[turn + "_" + "level-4_population"] = model.NewIntVar(0, 20, turn + "_" + "level-4_population")
    for i in range(5, MAX_CITY_LEVEL + 1):
        city[turn + "_" + "level-" + str(i) + "_giant"] = \
            model.NewIntVar(0, 20, turn + "_" + "level-" + str(i) + "_giant")
        city[turn + "_" + "level-" + str(i) + "_garden"] = \
            model.NewIntVar(0, 20, turn + "_" + "level-" + str(i) + "_garden")

    for i in range(1, MAX_CITY_LEVEL + 1):
        city[turn + "_" + "level-" + str(i) + "_count"] = \
            model.NewIntVar(0, 20, turn + "_" + "level-" + str(i) + "_count")

    city[turn + "_" + "pop_min"] = model.NewIntVar(0, 40, turn + "_" + "pop_min")
    city[turn + "_" + "pop_max"] = model.NewIntVar(0, 40, turn + "_" + "pop_max")
    return city


def addPop(turn):
    pop = getPopulation()
    pop_dict = {}
    for p in pop:
        pop_dict[name(turn, p['name'])] = model.NewIntVar(0, 20, name(turn, p['name']))
        if type(p['pop']) == str:
            pop_dict[turn + "_" + p['name'] + "_" + p['pop']] =\
                model.NewIntVar(0, 20, turn + "_" + p['name'] + "_" + p['pop'])
    return pop_dict


def addStars(turn):
    stars = {}
    stars[turn + "_" + "stars"] = model.NewIntVar(0, 200, turn + "_" + "stars")
    stars[turn + "_" + "spt"] = model.NewIntVar(0, 100, turn + "_" + "spt")
    stars[turn + "_" + "star_income"] = model.NewIntVar(-20, 100, turn + "_" + "star_income")
    stars[turn + "_" + "star_spending"] = model.NewIntVar(-20, 100, turn + "_" + "star_spending")
    for p in getPopulation():
        stars[turn + "_" + p['name'] + "_" + "stars"] = model.NewIntVar(0, 200, turn + "_" + p['name'] + "_" + "stars")
    for topic, cost_dict in getCost().items():
        for item, cost in cost_dict.items():
            stars[turn + "_" + item + "_" + "stars"] = model.NewIntVar(0, 200, turn + "_" + item + "_" + "stars")
            if topic == 'techs':
                stars[turn + "_" + item + "_" + "cost"] = model.NewIntVar(0, 25, turn + "_" + item + "_" + "cost")
                stars[turn + "_" + item + "_" + "stars"] = model.NewIntVar(0, 25, turn + "_" + item + "_" + "stars")
    return stars
