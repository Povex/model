

class NodeAgent:
    """
        Abstract class of a node agent.
        Contains common attributes between all agents.
    """

    def __init__(self, unique_id):
        self.unique_id = unique_id
        self.is_malicious = False
        self.stop_epochs = 0

    def step(self):
        pass

    def get_stake(self):
        pass
