def validateScore(player_scores):
    for a in range(len(player_scores)):
        current = player_scores[a]
        for i in range(len(current) - 1):
            for j in range(len(current[i])):
                if current[i][j] > current[i + 1][j]:
                    return False
    return True


def filter_scores(g):
    def rec(g, g2):
        if g is None or g2 is None:
            pass
        elif g[0] is None or g2[0] is None:
            if g[1] is None or g2[1] is None:
                pass
            else:
                rec(g[1], g2[1])
        elif g[0][0] is None or g2[0][0] is None:
            if g[0][1] is None or g2[0][1] is None:
                pass
            else:
                rec
        elif type(g[0][0]) == int:
            if [g, g2] not in output and [g2, g] not in output and validateScore([g, g2]):
                output.append([g, g2])
        else:
            rec(g[0], g2[1])
            rec(g[1], g2[0])
    output = []
    rec(g, g)
    return output


def generateScores(combined_scores):
    def inner(choice, acc):
        if len(choice) == 0:
            return acc
        else:
            for i in range(2):
                next_accs = [
                    acc + [[choice[0][0][i], choice[0][1][0]]],
                    acc + [[choice[0][0][1 - i], choice[0][1][1]]]
                ]
                in1 = inner(choice[1:], next_accs[0])
                in2 = inner(choice[1:], next_accs[1])
                return [in1, in2]
    return inner(combined_scores, [])


def getPossibleScores(combined_scores):
    return filter_scores(generateScores(combined_scores))