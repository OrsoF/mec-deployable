"""Historical workspace method signatures; active callers use canonical operations."""

from mecbench.domain.solution import ArcSolution, CompactSolution


class LegacyWorkspaceOperations:
    def validate_graph_attributes(self, graph, placement, routing, quantities):
        return self.is_feasible(placement, routing, quantities)

    def update_graph_attributes(self, graph, placement, routing, quantities):
        return self.update_graph(CompactSolution(placement, routing, quantities), graph=graph)

    def update_graph_attributes_from_arc_variables(self, graph, x_variable, y_variable, q_variable):
        return self.update_graph(ArcSolution(x_variable, y_variable, q_variable), graph=graph)
