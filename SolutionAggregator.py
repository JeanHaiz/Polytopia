from ortools.sat.python import cp_model
import pandas as pd


class SolutionAggregator(cp_model.CpSolverSolutionCallback):
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
