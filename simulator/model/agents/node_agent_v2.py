from simulator.model.agents.abstract_node_agent import NodeAgent
from simulator.model.agents.coins import Coins


class NodeAgentV2(NodeAgent):
    """An agent with fixed initial stake."""

    def __init__(self, unique_id):
        super().__init__(unique_id)
        self.stake = []
    
    def add_stake(self, amount, epoch):
        coins = Coins(amount, epoch)
        self.stake.append(coins)

    def remove_stake(self, amount):
        to_dereference = []
        for coins in self.stake[::-1]:
            if coins.amount >= amount:
                coins.amount -= amount
                break
            else:
                to_dereference.append(coins)
                amount -= coins.amount
        for c in to_dereference:
            self.stake.remove(c)
    
    def get_stake_weight(self, current_epoch, min_coin_age, max_coin_age):
        stake_weight = 0
        for coins in self.stake:
            age = (current_epoch - coins.epoch)
            if min_coin_age <= age <= max_coin_age:
                stake_weight += coins.amount * age
            elif age > max_coin_age:
                stake_weight += coins.amount * max_coin_age

        return stake_weight

    def get_stake(self):
        return sum([s.amount for s in self.stake])

    def step(self):
        pass

    def __repr__(self):
        return f"NodeAgentv2(unique_id={self.unique_id}, stake={self.get_stake()}, is_malicious={self.is_malicious}, " \
               f"stop_epochs={self.stop_epochs})"
        