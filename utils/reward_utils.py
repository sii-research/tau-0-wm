import numpy as np


def compute_monte_carlo_returns(
    num_steps: int,
    terminal_reward: float,
    gamma: float,
    rescale: bool = True,
) -> np.ndarray:
    """
    Compute Monte Carlo returns with sparse reward at the end of an episode.
    Returns are rescaled from [0, terminal_reward] to [-1, 1].

    Args:
        num_steps (int): Number of steps in the episode.
        terminal_reward (float): Reward at the terminal state (usually 0 or 1).
        gamma (float): Discount factor.

    Returns:
        np.ndarray: Returns array of shape (num_steps,) rescaled to [-1, 1].
    """
    T = num_steps
    rewards = np.zeros(T, dtype=np.float32)
    rewards[-1] = terminal_reward

    # Compute Monte-Carlo returns
    returns = np.zeros_like(rewards)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
        
    if rescale:
        # Rescale returns from [0, terminal_reward] to [-1, 1]
        if terminal_reward > 0:
            returns = 2 * returns / terminal_reward - 1
        else:
            returns = 2 * returns - 1

    return returns



def align_rewards_to_indices(sample_indices, valid_indices, valid_returns):
    sample_indices = np.asarray(sample_indices)
    valid_indices = np.asarray(valid_indices)
    valid_returns = np.asarray(valid_returns, dtype=np.float32)

    failure_reward = -1.0

    if valid_indices.size == 0:
        return np.full(sample_indices.shape, failure_reward, dtype=np.float32)

    if valid_indices.shape[0] != valid_returns.shape[0]:
        raise ValueError("valid_indices and valid_returns must have the same length.")
    
    pos = np.searchsorted(valid_indices, sample_indices, side="right") - 1
    valid_mask = np.isin(sample_indices, valid_indices)
    
    out = np.full(sample_indices.shape, failure_reward, dtype=np.float32)

    out[valid_mask] = valid_returns[pos[valid_mask]]

    return out