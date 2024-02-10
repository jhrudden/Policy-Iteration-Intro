from scipy.stats import poisson
import numpy as np
from enum import IntEnum
from typing import Tuple


class ActionGW(IntEnum):
    """ActionGW"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: ActionGW) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (ActionGW): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        ActionGW.LEFT: (-1, 0),
        ActionGW.DOWN: (0, -1),
        ActionGW.RIGHT: (1, 0),
        ActionGW.UP: (0, 1),
    }
    return mapping[action]


class Gridworld5x5:
    """5x5 Gridworld"""

    def __init__(self) -> None:
        """
        State: (x, y) coordinates

        ActionGWs: See class(ActionGW).
        """
        self.rows = 5
        self.cols = 5
        self.shape = (self.rows, self.cols)
        self.state_space = [
            (x, y) for x in range(0, self.rows) for y in range(0, self.cols)
        ]
        self.action_space = len(ActionGW)

        # locations of A and B, the next locations, and their rewards
        self.A = (0,1)
        self.A_prime = (4,1)
        self.A_reward = 10
        self.B = (0,3)
        self.B_prime = (2,3)
        self.B_reward = 5

    def transitions(
        self, state: Tuple, action: ActionGW
    ) -> Tuple[Tuple[int, int], float]:
        """Get transitions from given (state, action) pair.

        Note that this is the 4-argument transition version p(s',r|s,a).
        This particular environment has deterministic transitions

        Args:
            state (Tuple): state
            action (ActionGW): action

        Returns:
            next_state: Tuple[int, int]
            reward: float
        """

        if state == self.A:
            reward = self.A_reward
            next_state = self.A_prime
        elif state == self.B:
            reward = self.B_reward
            next_state = self.B_prime
        else:
            d_x, d_y = actions_to_dxdy(action)
            next_state = (state[0] + d_x, state[1] + d_y)
            reward = 0
            if next_state[0] < 0 or next_state[0] >= self.cols or next_state[1] < 0 or next_state[1] >= self.rows:
                next_state = state
                reward = -1

        return next_state, reward

    def expected_return(
        self, V, state: Tuple[int, int], action: ActionGW, gamma: float
    ) -> float:
        """Compute the expected_return for all transitions from the (s,a) pair, i.e. do a 1-step Bellman backup.

        Args:
            V (np.ndarray): list of state values (length = number of states)
            state (Tuple[int, int]): state
            action (ActionGW): action
            gamma (float): discount factor

        Returns:
            ret (float): the expected return
        """

        next_state, reward = self.transitions(state, action)

        ret = reward + gamma * V[next_state]

        return ret


class JacksCarRental:
    def __init__(self, modified: bool = False) -> None:
        """JacksCarRental

        Args:
           modified (bool): False = original problem Q6a, True = modified problem for Q6b

        State: tuple of (# cars at location A, # cars at location B)

        ActionGW (int): -5 to +5
            Positive if moving cars from location A to B
            Negative if moving cars from location B to A
        """
        self.shape = (21, 21)
        self.modified = modified

        self.action_space = list(range(-5, 6))

        self.rent_reward = 10
        self.move_cost = 2

        # For modified problem
        self.overflow_cars = 10
        self.overflow_cost = 4

        # Rent and return Poisson process parameters
        # Save as an array for each location (Loc A, Loc B)
        self.rent = [poisson(3), poisson(4)]
        self.return_ = [poisson(3), poisson(2)]

        # Max number of cars at end of day
        self.max_cars_end = 20
        # Max number of cars at start of day
        self.max_cars_start = self.max_cars_end + max(self.action_space)

        self.state_space = [
            (x, y)
            for x in range(0, self.max_cars_end + 1)
            for y in range(0, self.max_cars_end + 1)
        ]

        # Store all possible transitions here as a multi-dimensional array (locA, locB, action, locA', locB')
        # This is the 3-argument transition function p(s'|s,a)
        self.t = np.zeros(
            (
                self.max_cars_end + 1,
                self.max_cars_end + 1,
                len(self.action_space),
                self.max_cars_end + 1,
                self.max_cars_end + 1,
            ),
        )

        # Store all possible rewards (locA, locB, action)
        # This is the reward function r(s,a)
        self.r = np.zeros(
            (self.max_cars_end + 1, self.max_cars_end + 1, len(self.action_space))
        )

        self.precompute_transitions()

    def _open_to_close(self, loc_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the probability of ending the day with s_end \in [0,20] cars given that the location started with s_start \in [0, 20+5] cars.

        Args:
            loc_idx (int): the location index. 0 is for A and 1 is for B. All other values are invalid
        Returns:
            probs (np.ndarray): list of probabilities for all possible combination of s_start and s_end
            rewards (np.ndarray): average rewards for all possible s_start
        """
        probs = np.zeros((self.max_cars_start + 1, self.max_cars_end + 1))
        rewards = np.zeros(self.max_cars_start + 1)
        for start in range(probs.shape[0]):
            # For all possible s_start, calculate the probability of renting k cars.
            # Be sure to consider the case where business is lost (i.e. renting k > s_start cars)
            prob_not_enough_cars = self.rent[loc_idx].sf(start)

            valid_rents = np.arange(0, start + 1)
            valid_rent_probs = self.rent[loc_idx].pmf(valid_rents)

            avg_rent = np.sum(np.multiply(valid_rents, valid_rent_probs))

            # handle the expect return when k > s_start
            avg_rent += prob_not_enough_cars * start

            rewards[start] = self.rent_reward * avg_rent

            # Loop over every possible s_end
            for end in range(probs.shape[1]):
                prob = 0.0
                if end == self.max_cars_end:
                    probs[start,end] = 1 - sum(probs[start,:-1])
                else:
                    # Since s_start and s_end are specified,
                    # you must rent a minimum of max(0, start-end)
                    min_rent = max(0, start - end)
                    for i in range(min_rent, start + 1):
                        num_return = end - (start - i)
                        prob += self.rent[loc_idx].pmf(i) * self.return_[loc_idx].pmf(num_return)
                    # If k > s_start, then the probability of ending with end is (prob_k > start) * returning start cars
                    prob += prob_not_enough_cars * self.return_[loc_idx].pmf(end)
                    probs[start, end] = prob

        return probs, rewards

    def _calculate_cost(self, state: Tuple[int, int], action: int) -> float:
        """A helper function to compute the cost of moving cars for a given (state, action)

        Note that you should compute costs differently if this is the modified problem.

        Args:
            state (Tuple[int,int]): state
            action (int): action
        """
        cost = abs(action) * self.move_cost

        if self.modified:
            # Handling employee shuttle
            if action > 0:
                # Employee shuttles a single car from A to B for free
                cost -= self.move_cost

                # If dealership A moved cars to B, but still has overflow cars
                if state[0] - action > self.overflow_cars:
                    cost += self.overflow_cost
            
            else:
                # if dealership B moved cars to A, but still has overflow cars
                if state[1] + action > self.overflow_cars:
                    cost += self.overflow_cost
                    
        return cost

    def _valid_action(self, state: Tuple[int, int], action: int) -> bool:
        """Helper function to check if this action is valid for the given state

        Args:
            state:
            action:
        """
        if state[0] < action or state[1] < -(action):
            return False
        else:
            return True

    def precompute_transitions(self) -> None:
        """Function to precompute the transitions and rewards.

        This function should have been run at least once before calling expected_return().
        You can call this function in __init__() or separately.

        """
        # Calculate open_to_close for each location
        day_probs_A, day_rewards_A = self._open_to_close(0)
        day_probs_B, day_rewards_B = self._open_to_close(1)

        # Perform action first then calculate daytime probabilities
        for locA in range(self.max_cars_end + 1):
            for locB in range(self.max_cars_end + 1):
                for ia, action in enumerate(self.action_space):
                    # Check boundary conditions
                    if not self._valid_action((locA, locB), action):
                        self.t[locA, locB, ia, :, :] = 0
                        self.r[locA, locB, ia] = 0
                    else:
                        # Use day_rewards_A and day_rewards_B and _calculate_cost()
                        cost = self._calculate_cost((locA, locB), action) 
                        self.r[locA, locB, ia] = day_rewards_A[locA - action] + day_rewards_B[locB + action] - cost

                        # Loop over all combinations of locA_ and locB_
                        for locA_ in range(self.max_cars_end + 1):
                            for locB_ in range(self.max_cars_end + 1):
                                # Use the probabilities computed from open_to_close
                                self.t[locA, locB, ia, locA_, locB_] = day_probs_A[locA - action, locA_] * day_probs_B[locB + action, locB_]

    def expected_return(
        self, V, state: Tuple[int, int], action: int, gamma: float
    ) -> float:
        """Compute the expected_return for all transitions from the (s,a) pair, i.e. do a 1-step Bellman backup.

        Args:
            V (np.ndarray): list of state values (length = number of states)
            state (Tuple[int, int]): state
            action: action
            gamma (float): discount factor

        Returns:
            ret (float): the expected return
        """

        
        p_s_prime = self.transitions(state, action)
        r_sa = self.rewards(state, action)
        ret = 0
        for s_prime in self.state_space:
            ret += p_s_prime[s_prime] * (r_sa + gamma * V[s_prime])

        return ret

    def transitions(self, state: Tuple, action: int) -> np.ndarray:
        """Get transition probabilities for given (state, action) pair.

        Note that this is the 3-argument transition version p(s'|s,a).
        This particular environment has stochastic transitions

        Args:
            state (Tuple): state
            action (Action): action

        Returns:
            probs (np.ndarray): return probabilities for next states. Since transition function is of shape (locA, locB, action, locA', locB'), probs should be of shape (locA', locB')
        """
        locA, locB = state
        return self.t[locA, locB, action + 5, :, :]

    def rewards(self, state, action) -> float:
        """Reward function r(s,a)

        Args:
            state (Tuple): state
            action (ActionGW): action
        Returns:
            reward: float
        """
        return self.r[state[0], state[1], action + 5]
