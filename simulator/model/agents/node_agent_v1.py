from simulator.model.agents.abstract_node_agent import NodeAgent


class NodeAgentV1(NodeAgent):
    """An agent with fixed initial stake."""

    def __init__(self, unique_id, stake):
        super().__init__(unique_id)
        self.stake = stake

    def get_stake(self):
        return self.stake

    def __repr__(self):
        return f"NodeAgent(unique_id={self.unique_id}, stake={self.get_stake()}, is_malicious={self.is_malicious}, " \
               f"stop_epochs={self.stop_epochs})"

    def step(self):
        pass
