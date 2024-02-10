import numpy as np
from typing import Callable, List, Tuple, Dict
from collections import defaultdict
import matplotlib.pyplot as plt

from env import Gridworld5x5
from utils import plot_cardinal_value_and_policy, plot_car_rental_policy_map

def get_equiproable_policy(env, actions) -> Dict[Tuple[int, int], List[Tuple[int, float]]]:
    """
    returns an equiprobable policy for the given environment

    Args:
    env: the environment
    actions: the list of actions
    """
    return {s: [(a, 1/len(actions)) for a in actions] for s in env.state_space}

def get_favorable_policy(env, actions) -> Dict[Tuple[int, int], List[Tuple[int, float]]]:
    return {s: [(a, 1) if a == 0 else (a, 0) for a in actions] for s in env.state_space}

def iterative_policy_evaluation(env, policy: Dict[Tuple[int, int], List[Tuple[int, float]]], discount_factor: float = 0.9, theta: float = 1e-3) -> np.ndarray:
    """
    iteratively evaluates a policy until the value function changes less than theta for all states

    Args:
    env: the environment
    policy: a function that maps states to a list of action, probability pairs
    discount_factor: the discount factor
    theta: the minimum change in the value function to continue iterating
    """
    V = np.zeros(env.shape)
    while True:
        delta = 0
        for s in env.state_space:
            v = V[s]
            new_v = 0
            for a, action_prob in policy[s]:
                new_v += action_prob * env.expected_return(V, s, a, discount_factor)
            V[s] = new_v
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    V = V.round(2)
    return V

def value_iteration(env, actions: List, discount_factor: float = 0.9, theta: float = 1e-3) -> Tuple[np.ndarray, Dict[Tuple[int, int], List[int]]]:
    """
    performs value iteration to find the optimal value function and policy

    Args:
    env: the environment
    actions: the list of actions
    discount_factor: the discount factor
    theta: the minimum change in the value function to continue iterating
    """
    rows, cols = env.rows, env.cols
    V = np.zeros((rows, cols))
    while True:
        delta = 0
        for s in env.state_space:
            v = V[s]
            V[s] = max(env.expected_return(V, s, a, discount_factor) for a in actions)
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    policy = defaultdict(list)
    V = V.round(2) # round to 2 decimal places to be consistent with figure 3.5
    for s in env.state_space:
        action_expected_returns = [env.expected_return(V, s, a, discount_factor) for a in actions]
        equip_optimal_actions = np.where(action_expected_returns == np.max(action_expected_returns))[0]
        action_prob_pairs = [(a, 1/len(equip_optimal_actions)) for a in equip_optimal_actions]
        policy[s] = action_prob_pairs
    return V, policy


def policy_iteration(env, actions: List, discount_factor: float = 0.9, theta: float = 1e-3, initial_policy: Callable = get_equiproable_policy, display: bool = False) -> Tuple[np.ndarray, Dict[Tuple[int, int], List[int]]]:
    """
    performs policy iteration to find the optimal value function and policy

    Args:
    env: the environment
    actions: the list of actions
    discount_factor: the discount factor
    theta: the minimum change in the value function to continue iterating
    actions: the list of actions
    display: whether to display the value function and policy
    """
    V = np.zeros(env.shape)
    policy = initial_policy(env, actions) # initialize the policy to be equiprobable for all states
    i = 0

    while True: 
        V = iterative_policy_evaluation(env, policy, discount_factor, theta)
        policy_stable = True
        num_missing_actions = 0

        new_policy_mapping = defaultdict(list)

        for s in env.state_space:
            old_action_prob_pairs = policy[s]
            old_actions = [a for a, _ in old_action_prob_pairs]
            action_expected_returns = [env.expected_return(V, s, a, discount_factor) for a in actions]
            # get all actions with the same max expected return
            equal_max_actions_indices = np.where(action_expected_returns == np.max(action_expected_returns))[0] 
            equal_max_actions = [actions[i] for i in equal_max_actions_indices]

            # group the optimal action with uniform probability
            new_policy_mapping[s] = list(zip(equal_max_actions, [1/len(equal_max_actions)]*len(equal_max_actions)))

            assert len(old_actions) > 0
            
            # check if the policy is stable
            if set(old_actions) != set(equal_max_actions):
                num_missing_actions += 1
                policy_stable = False
            
        policy = new_policy_mapping

        if display:
            if len(actions) == 4:
                plot_cardinal_value_and_policy(V, policy, title="Policy Iteration")
            else:
                fig, ax = plt.subplots()
                plot_car_rental_policy_map(ax, policy, title=f"$\pi_{i}$")

        if policy_stable:
            break
        i += 1

    return V, policy
