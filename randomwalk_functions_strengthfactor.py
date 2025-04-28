import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


### 1D SYMMETRIC random walk
def generate_1d_walk(num_steps = 10_000):

    """
    Generates a single 1D symmetric random walk.
    Input: num_steps, k (strength factor)
    Output: trajectory (as a 1D array)
    """

    # Initialize trajectory
    trajectory = [0] * (num_steps+1)
    directions = np.array([-1, 1])
    probabilities = np.array([1/2, 1/2])
    curr = 0 

    for i in range(1, num_steps+1):
        step = np.random.choice(directions, p=probabilities).item()

        # Add step to trajectory
        curr += step
        trajectory[i] = curr

    return trajectory


### 1D POSITIVE reinforcement random walk
def generate_1d_walk_positive(num_steps = 10_000, k=1):

    """
    Generates a single 1D random walk with POSITIVE reinforcement.
    Input: num_steps
    Output: trajectory (as a 1D array)
    """

    # Initialize trajectory and reinforcement factor (transition probabilities)
    trajectory = [0]
    directions = np.array([-1, 1])
    l = 1
    r = 1
    total = 2
    probabilities = np.array([1/2, 1/2])
    curr = 0 

    for i in range(num_steps):
        step = np.random.choice(directions, p=probabilities).item()
        if step == -1:
            l += k
        elif step == 1:
            r += k
        total += k

        # add step to trajectory
        curr += step
        trajectory.append(curr)

        # update probabilities
        probabilities[0] = l/total
        probabilities[1] = r/total
    

    return trajectory


### 1D NEGATIVE reinforcement random walk
def generate_1d_walk_negative(num_steps = 10_000, k=1):

    """
    Generates a single 1D random walk with NEGATIVE reinforcement.
    Input: num_steps
    Output: trajectory (as a 1D array)
    """

    # Initialize trajectory and reinforcement factor (transition probabilities)
    trajectory = [0]
    directions = np.array([-1, 1])
    l = 1
    r = 1
    total = 2
    probabilities = np.array([1/2, 1/2])
    curr = 0 

    for i in range(num_steps):
        step = np.random.choice(directions, p=probabilities).item()
        if step == -1:
            r += k
        elif step == 1:
            l += k
        total += k

        # add step to trajectory
        curr += step
        trajectory.append(curr)

        # update probabilities
        probabilities[0] = l/total
        probabilities[1] = r/total
    

    return trajectory

# Generate a lot of 1D walks
def generate_many_walks(num_walks, num_steps, type, k=1):

    """
    Generate num_walks number of trajectories with num_steps, of a specified type.
    `type` can take: ["normal", "positive", "negative"]
    Output: DataFrame
    """

    # Dimensions
    # num_walks is the number of rows (each row is a RW trajectory)
    # num_steps+1 is the number of columns (each column is a step)

    if type == "normal":
        trajectories = [generate_1d_walk(num_steps) for _ in range(num_walks)]
    elif type == "positive":
        trajectories = [generate_1d_walk_positive(num_steps, k) for _ in range(num_walks)]
    elif type == "negative":
        trajectories = [generate_1d_walk_negative(num_steps, k) for _ in range(num_walks)]

    # Convert to DataFrame
    columns = [f'X_{i}' for i in range(num_steps+1)]
    df = pd.DataFrame(trajectories, columns=columns)

    return df