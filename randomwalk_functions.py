import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


### 1D SYMMETRIC random walk
def generate_1d_walk(num_steps = 10_000):

    """
    Generates a single 1D symmetric random walk.
    Input: num_steps
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
def generate_1d_walk_positive(num_steps = 10_000):

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
            l += 1
        elif step == 1:
            r += 1
        total += 1

        # add step to trajectory
        curr += step
        trajectory.append(curr)

        # update probabilities
        probabilities[0] = l/total
        probabilities[1] = r/total
    

    return trajectory


### 1D NEGATIVE reinforcement random walk
def generate_1d_walk_negative(num_steps = 10_000):

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
            r += 1
        elif step == 1:
            l += 1
        total += 1

        # add step to trajectory
        curr += step
        trajectory.append(curr)

        # update probabilities
        probabilities[0] = l/total
        probabilities[1] = r/total
    

    return trajectory

# Generate a lot of 1D walks
def generate_many_walks(num_walks, num_steps, type):

    """
    Generate num_walks number of trajectories with num_steps, of a specified type.
    `type` can take: ["normal", "positive", "negative"]
    Output: DataFrame
    """

    # Dimensions
    # num_walks is the number of rows (each row is a RW trajectory)
    # num_steps+1 is the number of columns (each column is a step)

    # Initialize empty numpy array
    data = np.empty((num_walks, num_steps+1))

    if type == "normal":
        for i in range(num_walks):
            data[i] = generate_1d_walk(num_steps)
    elif type == "positive":
        for i in range(num_walks):
            data[i] = generate_1d_walk_positive(num_steps)
    elif type == "negative":
        for i in range(num_walks):
            data[i] = generate_1d_walk_negative(num_steps)

    # Convert to DataFrame
    columns = [f'X_{i}' for i in range(num_steps+1)]
    df = pd.DataFrame(data, columns=columns)

    return df


###########

### 2D SYMMETRIC random walk
def generate_2d_walk(num_steps = 1_000):

    """
    Generates a 2D symmetric random walk.
    Input: num_steps
    Output: trajectory (as a 1D array of coordinates)
    """

    # Initialize trajectory
    trajectory = [np.array([0,0])] * (num_steps+1)
    directions = np.array([[0,-1], [0,1], [1,0], [-1,0]])
    probabilities = np.array([1/4, 1/4, 1/4, 1/4])
    curr = np.array([0,0])

    for i in range(1, num_steps+1):
        index = np.random.choice(range(4), p=probabilities)
        step = directions[index]

        # Add step to trajectory
        curr += step
        trajectory[i] = curr

    return trajectory


# 2D POSITIVE reinforcement random walk
def generate_2d_walk_positive(num_steps = 1_000):

    """
    Generates a 2D positive random walk.
    Input: num_steps
    Output: trajectory (as a 1D array of coordinates)
    """

    # Initialize trajectory
    trajectory = [np.array([0,0])] * (num_steps+1)
    directions = np.array([[0,-1], [0,1], [1,0], [-1,0]])
    probabilities = np.array([1/4, 1/4, 1/4, 1/4])
    curr = np.array([0,0])
    
    left = 1
    right = 1
    down = 1
    up = 1
    total = 4

    for i in range(1, num_steps+1):
        index = np.random.choice(range(4), p=probabilities)
        step = directions[index]

        # Add step to trajectory
        curr = curr + step
        trajectory[i] = curr

        # update the probabilities
        if index == 0:
            down += 1
        elif index == 1:
            up += 1
        elif index == 2:
            right += 1
        else:
            left += 1
        total += 1

        # update probabilities
        probabilities[0] = down/total
        probabilities[1] = up/total
        probabilities[2]=right/total
        probabilities[3]=left/total


    return trajectory


# 2D NEGATIVE reinforcement random walk
def generate_2d_walk_negative(num_steps = 1_000):

    """
    Generates a 2D negative random walk.
    Input: num_steps
    Output: trajectory (as a 1D array of coordinates)
    """

    # Initialize trajectory
    trajectory = [np.array([0,0])] * (num_steps+1)
    directions = np.array([[0,-1], [0,1], [1,0], [-1,0]])
    probabilities = np.array([1/4, 1/4, 1/4, 1/4])
    curr = np.array([0,0])
    
    left = 1
    right = 1
    down = 1
    up = 1
    total = 4

    for i in range(1, num_steps+1):
        index = np.random.choice(range(4), p=probabilities)
        step = directions[index]

        # Add step to trajectory
        curr = curr + step
        trajectory[i] = curr

        # update the probabilities
        if index == 0:
            up += 1
            right += 1
            left += 1
        elif index == 1:
            down += 1
            right += 1
            left += 1
        elif index == 2:
            down += 1
            up += 1
            left += 1
        else:
            down += 1
            up += 1
            right += 1
        total += 3

        # update probabilities
        probabilities[0] = down/total
        probabilities[1] = up/total
        probabilities[2]=right/total
        probabilities[3]=left/total


    return trajectory


###########

# Other useful functions for answering our questions

def first_match_index(row, df):
    """
    For a given df, and each row in the df, return the first index at which the random walk hits 0
    (after the initial step).
    Output: Series
    """
    matches = row[1:][row[1:] == 0]    # Look only after the first step
    # Find the first match index
    result = df.columns.get_loc(matches.index[0]) if not matches.empty else None

    return result

def prop_returned_by_time(first_match_indices, total_num_walks):
    """
    Given a Series returned from first_match_index(), return a list that holds the proportion of walks 
    that have returned to the origin by time i.
    """

    # Get the maximum index at which the origin was travelled back to, to capture all possibilities, but
    # avoid having to search through unnecessary extra steps
    max_val = max(first_match_indices)

    # This is adding 1 to the counter if the first match index is less than each natural number up to max_val
    # In other words, getting the number of walks which have returned for the first time by each natural number index
    # and then converting that into a proportion
    result = [sum(1 for x in first_match_indices if x < i)/total_num_walks for i in range(int(max_val) + 1)]

    return result


def first_match_index_2d(row):
    for i, point in enumerate(row[1:], start=1):  # start=1 to offset skipping the first
        if np.array_equal(point, [0, 0]):
            return int(i)
    return None