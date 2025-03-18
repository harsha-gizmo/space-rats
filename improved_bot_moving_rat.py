import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import numpy as np
import copy
import math
from collections import deque
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError
import os
from scipy.spatial.distance import cdist
from datetime import datetime
import multiprocessing
import time
from collections import Counter
from heapq import heappush, heappop
from sklearn.cluster import DBSCAN
from collections import defaultdict
from functools import lru_cache

# Start the timer
start_time = time.time()

os.environ['PYTHONUNBUFFERED'] = "1"

GRID_SIZE = 30  # Grid size
# TARGET_OPEN_PERCENTAGE = 0.6  # Target percentage of open cells
# MOVEMENTS = 0
# BLOCKED_CELL_ACTIONS = 0
# SPACE_RAT_DETECTOR_ACTIONS = 0
max_cores = 15

# timeout_limit = 1500
# failed_simulations = 0

class SimulationState:
    def __init__(self):
        self.movements = 0
        self.blocked_cell_actions = 0
        self.space_rat_detector_actions = 0
        self.failed_simulations = 0
        self.actions = []

    def reset(self):
        self.movements = 0
        self.blocked_cell_actions = 0
        self.space_rat_detector_actions = 0
        self.failed_simulations = 0
        self.actions = []

# Create empty grid layout
def create_grid_backup(size):
    return [[1 for _ in range(size)] for _ in range(size)]  # 1 represents 'blocked'

def create_grid(size):
    grid = [[1 for _ in range(size)] for _ in range(size)]  # 1 represents 'blocked'
    for i in range(size):
        grid[0][i] = 1  # Top edge
        grid[size - 1][i] = 1  # Bottom edge
        grid[i][0] = 1  # Left edge
        grid[i][size - 1] = 1  # Right edge
    return grid

# Find a random open cell in the grid for the space rat's initial position
def get_random_open_position(grid):
    open_positions = [(x, y) for x in range(1, GRID_SIZE - 1) for y in range(1, GRID_SIZE - 1) if grid[x][y] == 0]
    return random.choice(open_positions) if open_positions else None

# Pick random starting cell in grid
def start_with_initial_cell(grid):
    x, y = random.randint(1, GRID_SIZE - 2), random.randint(1, GRID_SIZE - 2)
    grid[x][y] = 0  # 0 represents 'open'
    return (x, y)

def get_candidates_for_opening(grid):
    candidates = []
    for row in range(1, GRID_SIZE - 1):  # Exclude top and bottom edges
        for col in range(1, GRID_SIZE - 1):  # Exclude left and right edges
            if grid[row][col] == 1:  # 'blocked' cells
                open_neighbors = sum(
                    grid[nr][nc] == 0  # 'open' cells
                    for nr, nc in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE
                )
                if open_neighbors == 1:
                    candidates.append((row, col))
    return candidates

# Open the candidate cells retrieved in above function iteratively
def open_cells(grid):
    while True:
        candidates = get_candidates_for_opening(grid)
        if not candidates:
            break
        cell_to_open = random.choice(candidates)
        grid[cell_to_open[0]][cell_to_open[1]] = 0  # Set to 'open'

def find_dead_ends(grid):
    dead_ends = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if grid[row][col] == 0:  # 'open' cells
                open_neighbors = sum(
                    grid[nr][nc] == 0
                    for nr, nc in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE
                )
                if open_neighbors == 1:
                    dead_ends.append((row, col))
    return dead_ends

def expand_dead_ends(grid, dead_ends):
    selected_dead_ends = random.sample(dead_ends, len(dead_ends) // 2)
    for dead_end in selected_dead_ends:
        row, col = dead_end
        blocked_neighbors = [
            (nr, nc)
            for nr, nc in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and grid[nr][nc] == 1  # 'blocked'
        ]
        if blocked_neighbors:
            neighbor_to_open = random.choice(blocked_neighbors)
            grid[neighbor_to_open[0]][neighbor_to_open[1]] = 0  # Set to 'open'

# Check if 60 percent of grid open before expanding dead ends
def check_sixty_percent(grid):
    TARGET_OPEN_PERCENTAGE = 0.6  # Target percentage of open cells
    open_count = sum(row.count(0) for row in grid)  # Count 'open' cells
    total_cells = GRID_SIZE * GRID_SIZE
    return open_count >= TARGET_OPEN_PERCENTAGE * total_cells

def block_edge_cells(grid):
    # Block all cells in the first and last row
    for col in range(GRID_SIZE):
        grid[0][col] = 1  # Block the first row
        grid[GRID_SIZE - 1][col] = 1  # Block the last row

    # Block all cells in the first and last column
    for row in range(GRID_SIZE):
        grid[row][0] = 1  # Block the first column
        grid[row][GRID_SIZE - 1] = 1  # Block the last column

# Wrapper function for creating grid layout
def create_space_vessel_layout():
    while True:
        grid = create_grid(GRID_SIZE)
        start_with_initial_cell(grid)
        open_cells(grid)

        # if check_sixty_percent(grid):
        #     pass
        # else:
        #     print("60 percent not open, trying again")
        #     continue

        dead_ends = find_dead_ends(grid)
        expand_dead_ends(grid, dead_ends)

        # Ensure all edge cells are blocked
        block_edge_cells(grid)

        if check_sixty_percent(grid):
            pass
        else:
            # print("60 percent not open, trying again")
            continue

        break
                
    return grid

def convert_grid_to_numpy(grid):
    numpy_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r][c] == 1:  # 'blocked'
                numpy_grid[r, c] = 1
    return numpy_grid

def sense_blocked_neighbors(grid, position, state):
    # global BLOCKED_CELL_ACTIONS
    x, y = position
    directions = [(x-1, y), (x+1, y), (x, y-1), (x, y+1), 
                  (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)]
    count_blocked = 0
    for nx, ny in directions:
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx, ny] == 1:
            count_blocked += 1
    # BLOCKED_CELL_ACTIONS = BLOCKED_CELL_ACTIONS + 1
    return count_blocked

# def possible_positions_after_sensing(grid, sensed_blocked_neighbors, eliminated_positions, state):
#     # print(f"Target blocked neighbours: {sensed_blocked_neighbors}")
#     state.blocked_cell_actions += 1
#     possible_positions = []
#     for row in range(1, GRID_SIZE - 1):
#         for col in range(1, GRID_SIZE - 1):
#             if grid[row][col] == 0 and (row, col) not in eliminated_positions:  # Consider only open cells not in eliminated positions
#                 temp_blocked_neighbours = sense_blocked_neighbors(grid, (row, col), state)
#                 if temp_blocked_neighbours == sensed_blocked_neighbors:
#                     # print(f"Adding position {(row, col)} because it has {temp_blocked_neighbours} blocked neighbours")
#                     possible_positions.append((row, col))
#                 else:
#                     dummy = None
#                     # print(f"Eliminating position {(row, col)} because it has {temp_blocked_neighbours} blocked neighbours")
#     return possible_positions

def possible_positions_after_sensing(grid, sensed_blocked_neighbors, eliminated_positions, state):
    """
    Update possible positions based on sensed blocked neighbors using probabilities.
    """
    state.blocked_cell_actions += 1
    possible_positions = []
    for row in range(1, GRID_SIZE - 1):
        for col in range(1, GRID_SIZE - 1):
            if grid[row][col] == 0 and (row, col) not in eliminated_positions:  # Consider only open cells not in eliminated positions
                temp_blocked_neighbors = sense_blocked_neighbors(grid, (row, col), state)
                if temp_blocked_neighbors == sensed_blocked_neighbors:
                    possible_positions.append((row, col))
    return possible_positions


def move_bot_bkp(grid, possible_positions, current_position, previous_position):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
    random.shuffle(directions)
    open_directions = {d: 0 for d in directions}

    # Count open directions among possible positions
    for pos in possible_positions:
        x, y = pos
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx, ny] == 0:
                open_directions[(dx, dy)] += 1

    # Sort directions by the count of open directions, in descending order
    sorted_directions = sorted(open_directions.items(), key=lambda item: item[1], reverse=True)

    # Find the maximum open directions count
    max_count = sorted_directions[0][1]
    best_directions = [d for d, count in sorted_directions if count == max_count]

    # Step 1: Choose a random direction from the best directions
    chosen_direction = random.choice(best_directions)

    # Step 2: Check if this direction leads back to the previous position
    if previous_position:
        reverse_direction = (previous_position[0] - current_position[0], previous_position[1] - current_position[1])
        if chosen_direction == reverse_direction:
            # Step 3: Choose another best direction if available
            if len(best_directions) > 1:
                best_directions.remove(reverse_direction)
                chosen_direction = random.choice(best_directions)
            else:
                # Step 4: If only one best direction and it is the reverse direction,
                # choose it, otherwise select the next best direction
                found_alternative = False
                for direction, _ in sorted_directions:
                    if direction != reverse_direction:
                        chosen_direction = direction
                        found_alternative = True
                        break
                if not found_alternative:
                    # If no other option exists, stick with the reverse direction
                    chosen_direction = reverse_direction

    # Try the selected direction
    new_x, new_y = current_position[0] + chosen_direction[0], current_position[1] + chosen_direction[1]
    if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and grid[new_x, new_y] == 0:
        return (new_x, new_y), True, chosen_direction  # Successful move

    # If no valid move is found, return the current position
    return current_position, False, chosen_direction  # No valid move possible

def move_bot_v2(grid, possible_positions, current_position, previous_position, visited_positions):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
    random.shuffle(directions)
    open_directions = {d: 0 for d in directions}

    # Count open directions among possible positions
    for pos in possible_positions:
        x, y = pos
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx, ny] == 0:
                open_directions[(dx, dy)] += 1

    # Sort directions by the count of open directions, in descending order
    sorted_directions = sorted(open_directions.items(), key=lambda item: item[1], reverse=True)

    # Filter out directions leading to already visited or blocked positions
    filtered_directions = []
    for direction, _ in sorted_directions:
        new_x, new_y = current_position[0] + direction[0], current_position[1] + direction[1]
        if (new_x, new_y) not in visited_positions and 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and grid[new_x, new_y] == 0:
            filtered_directions.append(direction)

    # If all directions are visited or blocked, reset filtered directions to all possible (to avoid being stuck)
    if not filtered_directions:
        filtered_directions = [d for d, _ in sorted_directions if 0 <= current_position[0] + d[0] < GRID_SIZE and 0 <= current_position[1] + d[1] < GRID_SIZE and grid[current_position[0] + d[0], current_position[1] + d[1]] == 0]

    # If still no directions, consider the current position as a dead-end (return current state without change)
    if not filtered_directions:
        return current_position, False, (0, 0)  # No valid move

    # Choose a random direction from the best available directions
    chosen_direction = random.choice(filtered_directions)

    # Step 2: Check if this direction leads back to the previous position
    if previous_position:
        reverse_direction = (previous_position[0] - current_position[0], previous_position[1] - current_position[1])
        if chosen_direction == reverse_direction and len(filtered_directions) > 1:
            filtered_directions.remove(reverse_direction)
            chosen_direction = random.choice(filtered_directions)

    # Try the selected direction
    new_x, new_y = current_position[0] + chosen_direction[0], current_position[1] + chosen_direction[1]
    if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and grid[new_x, new_y] == 0:
        visited_positions.add((new_x, new_y))  # Mark as visited
        return (new_x, new_y), True, chosen_direction  # Successful move

    # If no valid move is found, return the current position
    return current_position, False, chosen_direction  # No valid move possible

def move_bot_bkp_latest(grid, possible_positions, current_position, previous_position, visited_positions):
    """
    DFS movement strategy that prioritizes open directions based on possible_positions
    but does not restrict movement solely to these positions. If no valid move is found among
    unvisited cells, it attempts a DFS-based move among visited cells (excluding previous_position).
    """
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
    random.shuffle(directions)  # Shuffle directions to add variability

    # Calculate direction priorities based on open paths around possible_positions
    open_directions = {d: 0 for d in directions}
    for pos in possible_positions:
        x, y = pos
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx][ny] == 0:
                open_directions[(dx, dy)] += 1

    # Sort directions by the count of open directions, in descending order
    sorted_directions = sorted(open_directions.items(), key=lambda item: item[1], reverse=True)

    # DFS Stack Initialization
    stack = [(current_position, [])]
    visited = visited_positions.copy()  # Maintain a copy of the visited set
    visited.add(current_position)  # Mark current position as visited

    # Step 1: Attempt DFS exploration among unvisited cells
    while stack:
        pos, path = stack.pop()

        # Attempt to move in each sorted direction (prioritizing open paths)
        for (dx, dy), _ in sorted_directions:
            new_x, new_y = pos[0] + dx, pos[1] + dy
            next_pos = (new_x, new_y)

            # Check if the move is valid (ignoring whether it's in possible_positions)
            if (
                0 <= new_x < GRID_SIZE and
                0 <= new_y < GRID_SIZE and
                grid[new_x][new_y] == 0 and  # Move to open cells only
                next_pos not in visited  # Avoid revisiting nodes initially
            ):
                visited.add(next_pos)
                new_path = path + [next_pos] if path else [next_pos]
                stack.append((next_pos, new_path))

                # If a move is found, prioritize this path
                if path:
                    chosen_direction = (new_path[0][0] - current_position[0], new_path[0][1] - current_position[1])
                    return new_path[0], True, chosen_direction  # Move to the first step in the new path
                else:
                    return next_pos, True, (dx, dy)  # Move to the new position

    # Step 2: DFS exploration among visited cells (excluding previous_position)
    stack = [(current_position, [])]  # Reinitialize stack for exploring visited cells
    while stack:
        pos, path = stack.pop()

        for (dx, dy), _ in sorted_directions:
            new_x, new_y = pos[0] + dx, pos[1] + dy
            next_pos = (new_x, new_y)

            # Allow movement to visited cells, but exclude previous_position
            if (
                0 <= new_x < GRID_SIZE and
                0 <= new_y < GRID_SIZE and
                grid[new_x][new_y] == 0 and
                next_pos != previous_position  # Exclude the previous position
            ):
                new_path = path + [next_pos] if path else [next_pos]
                stack.append((next_pos, new_path))

                # If a move is found, prioritize this path
                if path:
                    chosen_direction = (new_path[0][0] - current_position[0], new_path[0][1] - current_position[1])
                    return new_path[0], True, chosen_direction  # Move to the first step in the new path
                else:
                    return next_pos, True, (dx, dy)  # Move to the new position

    # Step 3: If no valid move is found, fall back to moving to the previous position
    if previous_position:
        chosen_direction = (previous_position[0] - current_position[0], previous_position[1] - current_position[1])
        return previous_position, True, chosen_direction

    # No valid move found, stay in the current position
    return current_position, False, (0, 0)

def cluster_positions(positions):
    """
    Cluster positions based on their proximity. Uses a simple clustering approach
    by grouping positions that are within a certain distance threshold.
    """
    clusters = []
    threshold = 3  # Distance threshold for clustering

    for pos in positions:
        added = False
        for cluster in clusters:
            # Check if the position is close enough to any existing cluster
            if cdist([pos], cluster, metric='euclidean').min() < threshold:
                cluster.append(pos)
                added = True
                break
        if not added:
            clusters.append([pos])

    return clusters

# def move_bot(grid, possible_positions, current_position, previous_position, visited_positions):
#     """
#     Heuristic-based movement strategy that prioritizes directions with higher connectivity.
#     Clusters possible positions and prioritizes movement towards the largest cluster.
#     """

#     directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
#     # Calculate a heuristic for each position based on surrounding open cells
#     def heuristic(position):
#         x, y = position
#         open_neighbors = sum(
#             1 for dx, dy in directions
#             if 0 <= x + dx < GRID_SIZE and 0 <= y + dy < GRID_SIZE and grid[x + dx][y + dy] == 0
#         )
#         return open_neighbors

#     # Sort possible positions based on the heuristic value (highest first)
#     possible_positions.sort(key=heuristic, reverse=True)

#     # Implement clustering logic to group nearby possible positions
#     if possible_positions:
#         clusters = cluster_positions(possible_positions)
#         # Move towards the largest cluster (by number of positions)
#         largest_cluster = max(clusters, key=len)
#         target_position = random.choice(largest_cluster)
#     else:
#         target_position = current_position

#     # Existing logic for movement with modifications for clustering
#     directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
#     random.shuffle(directions)  # Optional: Randomize to add variability
#     open_directions = {d: 0 for d in directions}

#     # Count open directions around the target position
#     x, y = target_position
#     for dx, dy in directions:
#         nx, ny = x + dx, y + dy
#         if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx, ny] == 0:
#             open_directions[(dx, dy)] += 1

#     # Sort directions by open connectivity count
#     sorted_directions = sorted(open_directions.items(), key=lambda item: item[1], reverse=True)

#     # Attempt to move towards the prioritized direction
#     for (dx, dy), _ in sorted_directions:
#         new_x, new_y = current_position[0] + dx, current_position[1] + dy
#         if (
#             0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and
#             grid[new_x, new_y] == 0 and (new_x, new_y) not in visited_positions
#         ):
#             visited_positions.add((new_x, new_y))
#             return (new_x, new_y), True, (dx, dy)

#     # If no valid move found, stay in place
#     return current_position, False, (0, 0)

def move_bot(grid, possible_positions, current_position, previous_position, visited_positions):
    """
    A* algorithm with heuristic for prioritizing movements.
    """
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

    def heuristic(position, target_position):
        """Heuristic function: Manhattan distance + open neighbors."""
        x, y = position
        tx, ty = target_position
        distance = abs(x - tx) + abs(y - ty)
        open_neighbors = sum(
            1 for dx, dy in directions
            if 0 <= x + dx < GRID_SIZE and 0 <= y + dy < GRID_SIZE and grid[x + dx][y + dy] == 0
        )
        return distance - open_neighbors  # Prioritize paths with more open neighbors

    if not possible_positions:
        return current_position, False, (0, 0)

    clusters = cluster_positions(possible_positions)
    largest_cluster = max(clusters, key=len)  # Target the largest cluster
    target_position = random.choice(largest_cluster)

    open_list = []
    heappush(open_list, (0, current_position, []))  # (f-cost, position, path)
    g_costs = {current_position: 0}
    visited = set()

    while open_list:
        _, current, path = heappop(open_list)
        if current in visited:
            continue
        visited.add(current)

        if current == target_position:
            # Follow the path to the next position
            if path:
                next_step = path[0]
                visited_positions.add(next_step)
                dx = next_step[0] - current_position[0]
                dy = next_step[1] - current_position[1]
                return next_step, True, (dx, dy)
            else:
                return current_position, False, (0, 0)  # Stay in place if no path

        x, y = current
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            new_position = (nx, ny)
            if (
                0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and
                grid[nx, ny] == 0 and new_position not in visited_positions
            ):
                new_g = g_costs[current] + 1  # Each step costs 1
                if new_position not in g_costs or new_g < g_costs[new_position]:
                    g_costs[new_position] = new_g
                    f_cost = new_g + heuristic(new_position, target_position)
                    heappush(open_list, (f_cost, new_position, path + [new_position]))

    # Fallback: No valid move found, stay in place
    return current_position, False, (0, 0)

def plot_grid(ax, grid, current_position, possible_positions, title, eliminated_positions):
    ax.clear()  # Clear the previous plot
    cmap = mcolors.ListedColormap(['white', 'black'])  # Open (0) -> white, Blocked (1) -> black
    ax.imshow(grid, cmap=cmap)

    # Highlight eliminated positions
    for pos in eliminated_positions:
        ax.text(pos[1], pos[0], 'X', color='red', ha='center', va='center', fontsize=10, fontweight='bold')

    # Highlight possible positions
    for pos in possible_positions:
        ax.scatter(pos[1], pos[0], color='yellow', s=100, edgecolor='black', label='Possible Positions' if pos == possible_positions[0] else "")

    # Highlight current position
    ax.scatter(current_position[1], current_position[0], color='red', s=100, edgecolor='black', label='Bot Current Position')

    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.invert_yaxis()  # Invert y-axis for intuitive visualization
    plt.draw()
    plt.pause(0.1)  # Pause to visualize the update

def update_space_rat_knowledge_base(grid, bot_position, rat_positions, alpha, ping_received, state):
    """
    Update possible positions of the space rat using Bayesian inference based on sensor data.
    """
    # global SPACE_RAT_DETECTOR_ACTIONS
    # SPACE_RAT_DETECTOR_ACTIONS = SPACE_RAT_DETECTOR_ACTIONS + 1
    state.space_rat_detector_actions += 1
    state.actions.append("detection")
    bx, by = bot_position
    updated_rat_positions = []
    posterior_probabilities = {}

    # Compute prior probabilities
    total_prior = len(rat_positions)
    prior_prob = 1 / total_prior if total_prior > 0 else 0

    for rx, ry in rat_positions:
        distance = abs(bx - rx) + abs(by - ry)  # Manhattan distance
        likelihood = math.exp(-alpha * (distance - 1)) if ping_received else 1 - math.exp(-alpha * (distance - 1))
        
        # Bayes' theorem: posterior = (likelihood * prior) / normalization constant
        posterior_prob = likelihood * prior_prob
        posterior_probabilities[(rx, ry)] = posterior_prob

    # Normalize probabilities
    total_posterior = sum(posterior_probabilities.values()) or 1
    normalized_probabilities = {pos: prob / total_posterior for pos, prob in posterior_probabilities.items()}

    # Update the list of possible rat positions based on the new probabilities
    updated_rat_positions = [pos for pos, prob in normalized_probabilities.items() if prob > 0]

    return updated_rat_positions, normalized_probabilities

def manhattan_distance(pos1, pos2):
    """Helper function to compute Manhattan distance."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# def markov_filter_update(grid, possible_rat_positions, current_position, alpha, ping_received, state, rat_probabilities):
#     new_rat_probabilities = {}
#     state.space_rat_detector_actions += 1
#     state.actions.append("detection")

#     # Update probabilities based on the received ping
#     for x, y in possible_rat_positions:
#         distance = abs(x - current_position[0]) + abs(y - current_position[1])
#         probability_of_ping = math.exp(-alpha * (distance - 1)) if distance > 0 else 1
#         if ping_received:
#             new_rat_probabilities[(x, y)] = rat_probabilities.get((x, y), 1 / len(possible_rat_positions)) * probability_of_ping
#         else:
#             new_rat_probabilities[(x, y)] = rat_probabilities.get((x, y), 1 / len(possible_rat_positions)) * (1 - probability_of_ping)

#     # Normalize the probabilities
#     total_prob = sum(new_rat_probabilities.values())
#     for key in new_rat_probabilities:
#         new_rat_probabilities[key] /= total_prob

#     return new_rat_probabilities

# For dynamic Rat
def markov_filter_update_with_prediction(grid, possible_rat_positions, current_position, alpha, ping_received, state, rat_probabilities):
    new_rat_probabilities = {}
    state.space_rat_detector_actions += 1
    state.actions.append("detection")

    # Predict rat movement based on a simple Markov model (random movement assumed for this example)
    transition_probabilities = {}  # Dictionary to store transition probabilities for each position
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Cardinal directions (right, left, down, up)

    for x, y in possible_rat_positions:
        valid_moves = [(x + dx, y + dy) for dx, dy in directions if 0 <= x + dx < GRID_SIZE and 0 <= y + dy < GRID_SIZE and grid[x + dx][y + dy] == 0]
        num_valid_moves = len(valid_moves)
        if num_valid_moves > 0:
            transition_probabilities[(x, y)] = {move: 1 / num_valid_moves for move in valid_moves}

    # Update probabilities based on ping received and predicted rat movement
    for (x, y), prob in rat_probabilities.items():
        new_prob = prob
        distance = abs(x - current_position[0]) + abs(y - current_position[1])
        probability_of_ping = math.exp(-alpha * (distance - 1)) if distance > 0 else 1

        # Apply ping update
        if ping_received:
            new_prob *= probability_of_ping
        else:
            new_prob *= (1 - probability_of_ping)

        # Predictive update based on transition probabilities
        if (x, y) in transition_probabilities:
            for next_position, transition_prob in transition_probabilities[(x, y)].items():
                if next_position in new_rat_probabilities:
                    new_rat_probabilities[next_position] += new_prob * transition_prob
                else:
                    new_rat_probabilities[next_position] = new_prob * transition_prob

    # Normalize probabilities
    total_prob = sum(new_rat_probabilities.values())
    for key in new_rat_probabilities:
        new_rat_probabilities[key] = (new_rat_probabilities[key] / total_prob) if total_prob > 0 else 1 / len(possible_rat_positions)

    return new_rat_probabilities

def markov_filter_update_with_prediction_for_utility(grid, possible_rat_positions, current_position, alpha, ping_received, rat_probabilities):
    new_rat_probabilities = {}

    # Predict rat movement based on a simple Markov model (random movement assumed for this example)
    transition_probabilities = {}  # Dictionary to store transition probabilities for each position
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Cardinal directions (right, left, down, up)

    for x, y in possible_rat_positions:
        valid_moves = [(x + dx, y + dy) for dx, dy in directions if 0 <= x + dx < GRID_SIZE and 0 <= y + dy < GRID_SIZE and grid[x + dx][y + dy] == 0]
        num_valid_moves = len(valid_moves)
        if num_valid_moves > 0:
            transition_probabilities[(x, y)] = {move: 1 / num_valid_moves for move in valid_moves}

    # Update probabilities based on ping received and predicted rat movement
    for (x, y), prob in rat_probabilities.items():
        new_prob = prob
        distance = abs(x - current_position[0]) + abs(y - current_position[1])
        probability_of_ping = math.exp(-alpha * (distance - 1)) if distance > 0 else 1

        # Apply ping update
        if ping_received:
            new_prob *= probability_of_ping
        else:
            new_prob *= (1 - probability_of_ping)

        # Predictive update based on transition probabilities
        if (x, y) in transition_probabilities:
            for next_position, transition_prob in transition_probabilities[(x, y)].items():
                if next_position in new_rat_probabilities:
                    new_rat_probabilities[next_position] += new_prob * transition_prob
                else:
                    new_rat_probabilities[next_position] = new_prob * transition_prob

    # Normalize probabilities
    total_prob = sum(new_rat_probabilities.values())
    for key in new_rat_probabilities:
        new_rat_probabilities[key] = (new_rat_probabilities[key] / total_prob) if total_prob > 0 else 1 / len(possible_rat_positions)

    return new_rat_probabilities

# Harsha - to spike up actions for high alpha
# def markov_filter_update(grid, possible_rat_positions, current_position, alpha, ping_received, state, rat_probabilities):
#     new_rat_probabilities = {}
#     state.space_rat_detector_actions += 1
#     state.actions.append("detection")

#     # Loop through all possible rat positions to update probabilities based on ping data
#     for x, y in possible_rat_positions:
#         distance = abs(x - current_position[0]) + abs(y - current_position[1])
#         probability_of_ping = math.exp(-alpha * (distance - 1)) if distance > 0 else 1

#         # Lower detection probability interpretation
#         # If the ping probability is low (e.g., below 0.1), reduce its weight in the update
#         if probability_of_ping < 0.1:
#             probability_of_ping *= random.uniform(0.5, 0.8)  # Randomly reduce the weight further

#         # Calculate the new probability based on ping/no-ping and weighted probability of ping
#         if ping_received:
#             # If ping received, increase probability of positions close to the bot
#             new_rat_probabilities[(x, y)] = rat_probabilities.get((x, y), 1 / len(possible_rat_positions)) * probability_of_ping
#         else:
#             # If no ping received, decrease probability of positions close to the bot
#             new_rat_probabilities[(x, y)] = rat_probabilities.get((x, y), 1 / len(possible_rat_positions)) * (1 - probability_of_ping)

#     # Normalize probabilities so they sum to 1
#     total_prob = sum(new_rat_probabilities.values())
#     for key in new_rat_probabilities:
#         new_rat_probabilities[key] = (new_rat_probabilities[key] / total_prob) if total_prob > 0 else 1 / len(possible_rat_positions)

#     return new_rat_probabilities

# def markov_filter_update_for_utility_calc(grid, possible_rat_positions, current_position, alpha, ping_received, rat_probabilities):
#     new_rat_probabilities = {}

#     # Loop through all possible rat positions to update probabilities based on ping data
#     for x, y in possible_rat_positions:
#         distance = abs(x - current_position[0]) + abs(y - current_position[1])
#         probability_of_ping = math.exp(-alpha * (distance - 1)) if distance > 0 else 1

#         # Lower detection probability interpretation
#         # If the ping probability is low (e.g., below 0.1), reduce its weight in the update
#         if probability_of_ping < 0.1:
#             probability_of_ping *= random.uniform(0.5, 0.8)  # Randomly reduce the weight further

#         # Calculate the new probability based on ping/no-ping and weighted probability of ping
#         if ping_received:
#             # If ping received, increase probability of positions close to the bot
#             new_rat_probabilities[(x, y)] = rat_probabilities.get((x, y), 1 / len(possible_rat_positions)) * probability_of_ping
#         else:
#             # If no ping received, decrease probability of positions close to the bot
#             new_rat_probabilities[(x, y)] = rat_probabilities.get((x, y), 1 / len(possible_rat_positions)) * (1 - probability_of_ping)

#     # Normalize probabilities so they sum to 1
#     total_prob = sum(new_rat_probabilities.values())
#     for key in new_rat_probabilities:
#         new_rat_probabilities[key] = (new_rat_probabilities[key] / total_prob) if total_prob > 0 else 1 / len(possible_rat_positions)

#     return new_rat_probabilities

# # def markov_filter_update_for_utility_calc(grid, possible_rat_positions, current_position, alpha, ping_received, rat_probabilities):
#     new_rat_probabilities = {}

#     # Precompute values to avoid recalculating for every position
#     for x, y in possible_rat_positions:
#         distance = abs(x - current_position[0]) + abs(y - current_position[1])
#         base_probability = rat_probabilities.get((x, y), 1 / len(possible_rat_positions))

#         # Calculate the probability of ping based on distance
#         probability_of_ping = math.exp(-alpha * (distance - 1)) if distance > 0 else 1

#         # Adjust probability based on whether a ping was received
#         if ping_received:
#             # Increase probabilities of closer positions
#             new_probability = base_probability * (1 + probability_of_ping)
#         else:
#             # Decrease probabilities of closer positions
#             new_probability = base_probability * (1 - probability_of_ping)
#             # Ensure that the probability does not go negative
#             new_probability = max(new_probability, 0.01 * base_probability)  # Minimum bound to avoid extreme reduction

#         # Store updated probabilities
#         new_rat_probabilities[(x, y)] = new_probability

#     # Normalize probabilities so they sum to 1
#     total_prob = sum(new_rat_probabilities.values())
#     for key in new_rat_probabilities:
#         new_rat_probabilities[key] = (new_rat_probabilities[key] / total_prob) if total_prob > 0 else 1 / len(possible_rat_positions)

#     return new_rat_probabilities

def heuristic_rat(position, target_position, position_weights):
    """
    Heuristic function for A* algorithm that considers both Manhattan distance
    and position weights (probability of finding the rat).
    """
    px, py = position
    tx, ty = target_position
    distance = abs(px - tx) + abs(py - ty)  # Manhattan distance
    weight = position_weights.get(position, 0)
    # The heuristic is a combination of distance and a preference for higher weights
    return distance - weight  # Higher weight positions are prioritized

def move_bot_towards_rat(grid, bot_position, possible_rat_positions, position_weights, visited_positions):
    if not possible_rat_positions:
        return bot_position, False

    bx, by = bot_position
    target_positions = sorted(possible_rat_positions, key=lambda pos: position_weights.get(pos, 0), reverse=True)
    target_position = target_positions[0]  # Choose the highest probability target as the main goal

    # A* Search Initialization
    open_set = [(0, bot_position, [])]  # (f-score, position, path)
    g_scores = {bot_position: 0}  # Cost from start to the current position
    f_scores = {bot_position: heuristic_rat(bot_position, target_position, position_weights)}  # f = g + h
    visited = set()

    while open_set:
        # Get the position with the lowest f-score
        _, current_position, path = min(open_set, key=lambda x: x[0])
        open_set = [entry for entry in open_set if entry[1] != current_position]  # Remove it from the open set

        if current_position in possible_rat_positions and current_position not in visited_positions:
            # If this position is one of the targets and not visited, return the path
            if path:
                next_step = path[0]
                return next_step, True
            else:
                return current_position, True  # Directly at target position

        visited.add(current_position)
        cx, cy = current_position

        # Generate neighbors
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in directions:
            new_x, new_y = cx + dx, cy + dy
            new_position = (new_x, new_y)

            if (
                0 <= new_x < GRID_SIZE and
                0 <= new_y < GRID_SIZE and
                grid[new_x][new_y] == 0 and  # Check if the cell is open
                new_position not in visited
            ):
                tentative_g_score = g_scores[current_position] + 1  # Assume uniform cost for all moves
                if tentative_g_score < g_scores.get(new_position, float('inf')):
                    # This path to new_position is better than any previous one
                    g_scores[new_position] = tentative_g_score
                    f_score = tentative_g_score + heuristic_rat(new_position, target_position, position_weights)
                    f_scores[new_position] = f_score
                    open_set.append((f_score, new_position, path + [new_position]))

    # Fallback: Make a random valid move if stuck
    for dx, dy in random.sample([(0, 1), (0, -1), (1, 0), (-1, 0)], 4):
        new_x, new_y = bx + dx, by + dy
        if (new_x, new_y) not in visited_positions and 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and grid[new_x][new_y] == 0:
            print("Fallback move to avoid getting stuck")
            return (new_x, new_y), True

    return bot_position, False  # No valid movement possible

def move_rat_randomly(grid, rat_position):
    x, y = rat_position
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
    open_directions = [
        (x + dx, y + dy) for dx, dy in directions
        if 0 <= x + dx < GRID_SIZE and 0 <= y + dy < GRID_SIZE and grid[x + dx][y + dy] == 0
    ]
    if open_directions:
        return random.choice(open_directions)
    return rat_position  # If no open directions, stay in the same place

def plot_movement_vs_detection(actions, alpha, x=25):
    """
    Plots the last x actions (movement or detection) performed sequentially over time until the rat is caught.
    Each action is represented by a bar in the bar graph: movement (blue) and detection (red).
    The plot is saved to an image file named 'last_25_actions_<alpha_value>.png'.
    """
    # Take only the last x actions
    actions_to_plot = actions[-x:] if len(actions) > x else actions
    timestamps = range(1, len(actions_to_plot) + 1)  # Generate sequential timestamps from 1 to len(actions_to_plot)

    # Assign colors based on the action type
    colors = ['blue' if action == 'movement' else 'red' for action in actions_to_plot]

    plt.figure(figsize=(10, 6))
    plt.bar(timestamps, [1] * len(actions_to_plot), color=colors, alpha=0.7, edgecolor='black')  # Uniform bar heights of 1
    plt.xlabel('Timestamp')
    plt.ylabel('Action Type (Height Uniform)')
    plt.title(f'Movement vs Rat Detection Over Last {len(actions_to_plot)} Actions for Alpha {alpha}')
    plt.xticks(timestamps)  # Display all timestamps on the x-axis
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add a grid for y-axis
    plt.tight_layout()  # Adjust the layout for better fit

    # Save the plot to a file
    filename = f'last_25_actions_{alpha:.2f}.png'
    plt.savefig(filename)
    plt.close()  # Close the plot to free up memory

def scale_pulse_weight(pulse_rate, alpha):
    # Reduce pulse weight influence when alpha is close to 0 or 1
    if alpha < 0.1 or alpha > 0.9:
        return pulse_rate * 0.5  # Scale down influence (adjust factor as needed)
    return pulse_rate

# Dynamic Latest
def main_phase_2(grid, initial_bot_position, rat_initial_position, alpha, state):
    timeout_limit = 9000
    possible_rat_positions = [(x, y) for x in range(1, GRID_SIZE - 1) for y in range(1, GRID_SIZE - 1) if grid[x][y] == 0]
    bot_position = initial_bot_position
    rat_position = rat_initial_position
    visited_positions = set()
    rat_probabilities = {pos: 1 / len(possible_rat_positions) for pos in possible_rat_positions}

    # UCB-related variables
    action_counts = {'movement': 1, 'sensing': 1}  # Initialize with 1 to avoid division by zero
    action_rewards = {'movement': 0.0, 'sensing': 0.0}
    c = 2  # Exploration parameter
    print(f"c value: {c}")

    while state.movements < timeout_limit:
        # Move the rat randomly in one of the open directions
        rat_position = move_rat_randomly(grid, rat_position)
        bx, by = bot_position
        rx, ry = rat_position
        distance = abs(bx - rx) + abs(by - ry)
        pulse_rate = max(1.0 / (distance + 1), 0.1)

        # Determine if a ping is received based on the detection probability
        if distance == 0:
            ping_received = True
        else:
            probability_of_ping = math.exp(-alpha * (distance - 1))
            ping_received = random.random() < probability_of_ping

        # Calculate UCB values for each action
        total_actions = sum(action_counts.values())
        ucb_movement = (action_rewards['movement'] / action_counts['movement']) + \
                       c * math.sqrt(math.log(total_actions) / action_counts['movement'])
        ucb_sensing = (action_rewards['sensing'] / action_counts['sensing']) + \
                      c * math.sqrt(math.log(total_actions) / action_counts['sensing'])

        # Choose the action with the highest UCB value
        if ucb_movement > ucb_sensing:
            # Perform movement
            most_probable_rat_position = max(rat_probabilities, key=rat_probabilities.get)
            possible_rat_positions = [pos for pos in possible_rat_positions if rat_probabilities.get(pos, 0) > 0]

            if most_probable_rat_position in possible_rat_positions:
                possible_rat_positions = [most_probable_rat_position] + [pos for pos in possible_rat_positions if pos != most_probable_rat_position]

            next_position, moved = move_bot_towards_rat(grid, bot_position, possible_rat_positions, rat_probabilities, visited_positions)

            if moved:
                state.movements += 1
                state.actions.append("movement")
                bot_position = next_position
                visited_positions.add(bot_position)

                # Calculate total probability before movement
                total_probability_before = sum(rat_probabilities.get(pos, 0) for pos in possible_rat_positions)

                # Simulate knowledge base update using a copy
                temp_rat_probabilities = rat_probabilities.copy()

                # Calculate the probability of receiving a ping
                prob_ping = math.exp(-alpha * (distance - 1)) if distance > 0 else 1
                updated_probs_ping = markov_filter_update_with_prediction_for_utility(grid, possible_rat_positions, bot_position, alpha, True, temp_rat_probabilities)
                updated_probs_no_ping = markov_filter_update_with_prediction_for_utility(grid, possible_rat_positions, bot_position, alpha, False, temp_rat_probabilities)

                # Weighted average of probabilities based on the likelihood of receiving a ping
                simulated_rat_probabilities = {
                    pos: prob_ping * updated_probs_ping.get(pos, 0) + (1 - prob_ping) * updated_probs_no_ping.get(pos, 0)
                    for pos in possible_rat_positions
                }

                # Normalize the probabilities
                total_prob = sum(simulated_rat_probabilities.values())
                if total_prob > 0:
                    simulated_rat_probabilities = {pos: p / total_prob for pos, p in simulated_rat_probabilities.items()}

                # Calculate total probability after movement using simulated probabilities
                total_probability_after = sum(simulated_rat_probabilities.get(pos, 0) for pos in possible_rat_positions)

                # Reward based on probability improvement
                max_possible_prob_change = 1
                reward = max(0, total_probability_after - total_probability_before) / max_possible_prob_change

                # reward = max(0, total_probability_after - total_probability_before)
                action_rewards['movement'] += reward
                action_counts['movement'] += 1
        else:
            # Perform sensing
            old_entropy = -sum(p * math.log(p + 1e-9) for p in rat_probabilities.values())  # Calculate entropy
            rat_probabilities = markov_filter_update_with_prediction(grid, possible_rat_positions, bot_position, alpha, ping_received, state, rat_probabilities)
            new_entropy = -sum(p * math.log(p + 1e-9) for p in rat_probabilities.values())  # Calculate new entropy
            max_entropy = math.log(len(possible_rat_positions))  # Maximum possible entropy
            reward = max(0, (old_entropy - new_entropy) / max_entropy)  # Positive reward if entropy decreases
            # reward = max(0, old_entropy - new_entropy)  # Positive reward if entropy decreases
            action_rewards['sensing'] += reward
            action_counts['sensing'] += 1

        # Check if bot has caught the space rat
        if bot_position == rat_position:
            print(f"Bot caught the space rat at {bot_position}", flush=True)
            break

    if state.movements >= timeout_limit:
        state.movements = 0
        state.blocked_cell_actions = 0
        state.space_rat_detector_actions = 0
        print("Failed Simulation for catching rat")
        state.failed_simulations += 1

# UCB - switch between weighted and most probable
# def main_phase_2(grid, initial_bot_position, rat_initial_position, alpha, state, c=5):
#     timeout_limit = 9000
#     possible_rat_positions = [(x, y) for x in range(1, GRID_SIZE - 1) for y in range(1, GRID_SIZE - 1) if grid[x][y] == 0]
#     bot_position = initial_bot_position
#     rat_position = rat_initial_position
#     visited_positions = set()
#     rat_probabilities = {pos: 1 / len(possible_rat_positions) for pos in possible_rat_positions}

#     # UCB-related variables
#     action_counts = {'movement': 1, 'sensing': 1}
#     action_rewards = {'movement': 0.0, 'sensing': 0.0}

#     while state.movements < timeout_limit:
#         bx, by = bot_position
#         rx, ry = rat_position
#         distance = abs(bx - rx) + abs(by - ry)

#         # Determine if a ping is received based on the detection probability
#         if distance == 0:
#             ping_received = True
#         else:
#             probability_of_ping = math.exp(-alpha * (distance - 1))
#             ping_received = random.random() < probability_of_ping

#         # Calculate UCB values for each action
#         total_actions = sum(action_counts.values())
#         ucb_movement = (action_rewards['movement'] / action_counts['movement']) + \
#                        c * math.sqrt(math.log(total_actions) / action_counts['movement'])
#         ucb_sensing = (action_rewards['sensing'] / action_counts['sensing']) + \
#                       c * math.sqrt(math.log(total_actions) / action_counts['sensing'])

#         # Calculate entropy to determine spread of probabilities
#         entropy = -sum(p * math.log(p + 1e-9) for p in rat_probabilities.values())
#         max_entropy = math.log(len(rat_probabilities))
#         normalized_entropy = entropy / max_entropy  # Normalized to [0, 1]

#         # Switch between Most Probable Position and Weighted Expected Position based on entropy
#         if normalized_entropy > 0.5:
#             # High entropy: use Weighted Expected Position for broader exploration
#             total_weight = sum(rat_probabilities.values())
#             weighted_x = sum(pos[0] * prob for pos, prob in rat_probabilities.items()) / total_weight
#             weighted_y = sum(pos[1] * prob for pos, prob in rat_probabilities.items()) / total_weight
#             target_position = (round(weighted_x), round(weighted_y))
#         else:
#             # Low entropy: focus on Most Probable Position
#             target_position = max(rat_probabilities, key=rat_probabilities.get)

#         # Ensure the target position is within bounds
#         target_position = (
#             min(max(1, target_position[0]), GRID_SIZE - 2),
#             min(max(1, target_position[1]), GRID_SIZE - 2)
#         )

#         # Choose the action with the highest UCB value
#         if ucb_movement >= ucb_sensing:
#             # Perform movement towards the chosen target position
#             possible_rat_positions = [pos for pos in possible_rat_positions if rat_probabilities.get(pos, 0) > 0]
#             next_position, moved = move_bot_towards_rat(grid, bot_position, possible_rat_positions, rat_probabilities, visited_positions)

#             if moved:
#                 state.movements += 1
#                 state.actions.append("movement")
#                 bot_position = next_position
#                 visited_positions.add(bot_position)
#                 # Reward for movement (e.g., proximity improvement)
#                 new_distance = abs(bot_position[0] - target_position[0]) + abs(bot_position[1] - target_position[1])
#                 reward = max(0, distance - new_distance)  # Positive reward if closer to the target position
#                 action_rewards['movement'] += reward
#             action_counts['movement'] += 1
#         else:
#             # Perform sensing
#             old_entropy = -sum(p * math.log(p + 1e-9) for p in rat_probabilities.values())
#             rat_probabilities = markov_filter_update(grid, possible_rat_positions, bot_position, alpha, ping_received, state, rat_probabilities)
#             new_entropy = -sum(p * math.log(p + 1e-9) for p in rat_probabilities.values())
#             reward = max(0, old_entropy - new_entropy)  # Positive reward if entropy decreases
#             action_rewards['sensing'] += reward
#             action_counts['sensing'] += 1

#         # Check if bot has caught the space rat
#         if bot_position == rat_position:
#             print(f"Bot caught the space rat at {bot_position}", flush=True)
#             break

#     if state.movements >= timeout_limit:
#         state.movements = 0
#         state.blocked_cell_actions = 0
#         state.space_rat_detector_actions = 0
#         print("Failed Simulation for catching rat")
#         state.failed_simulations += 1

def run_single_simulation(alpha):
    # global MOVEMENTS
    # global BLOCKED_CELL_ACTIONS
    # global SPACE_RAT_DETECTOR_ACTIONS
    # global failed_simulations
    timeout_limit = 9000
    # MOVEMENTS = 0
    # BLOCKED_CELL_ACTIONS = 0
    # SPACE_RAT_DETECTOR_ACTIONS = 0
    # failed_simulations = 0

    state = SimulationState()  # Create an instance of SimulationState
    state.reset()

    # Create initial grid layout
    initial_grid = create_space_vessel_layout()
    grid = convert_grid_to_numpy(initial_grid)

    current_position = start_with_initial_cell(grid)
    sensed_blocked_neighbors = sense_blocked_neighbors(grid, current_position, state)
    eliminated_positions = []  # Track eliminated positions
    previous_position = None
    possible_positions = possible_positions_after_sensing(grid, sensed_blocked_neighbors, eliminated_positions, state)

    # Main loop for phase 1 logic
    print("Begin phase 1", flush=True)
    visited_positions = set()

    temp_movements = 0
    while state.movements < timeout_limit:
        next_position, moved, attempted_direction = move_bot(grid, possible_positions, current_position, previous_position, visited_positions)
        # next_position, moved, attempted_direction = move_bot_bkp(grid, possible_positions, current_position, previous_position)
        previous_position = current_position
        current_position = next_position
        visited_positions.add(current_position)  # Track visited positions

        if moved:
            state.movements += 1
            temp_movements += 1
            dx, dy = attempted_direction
            new_possible_positions = [
                pos for pos in possible_positions
                if (0 <= pos[0] + dx < GRID_SIZE and 0 <= pos[1] + dy < GRID_SIZE and grid[pos[0] + dx, pos[1] + dy] == 0)
            ]
            to_eliminate = [pos for pos in possible_positions if pos not in new_possible_positions]
            eliminated_positions.extend(to_eliminate)
            eliminated_positions = list(set(eliminated_positions))
            sensed_blocked_neighbors = sense_blocked_neighbors(grid, current_position, state)
            possible_positions = possible_positions_after_sensing(grid, sensed_blocked_neighbors, eliminated_positions, state)
            if not possible_positions:
                possible_positions = [current_position]
                break
        else:
            dx, dy = attempted_direction
            new_possible_positions = [
                pos for pos in possible_positions
                if not (0 <= pos[0] + dx < GRID_SIZE and 0 <= pos[1] + dy < GRID_SIZE and grid[pos[0] + dx, pos[1] + dy] == 0)
            ]
            to_eliminate = [pos for pos in possible_positions if pos not in new_possible_positions]
            eliminated_positions.extend(to_eliminate)
            eliminated_positions = list(set(eliminated_positions))
            sensed_blocked_neighbors = sense_blocked_neighbors(grid, current_position, state)
            possible_positions = possible_positions_after_sensing(grid, sensed_blocked_neighbors, eliminated_positions, state)
            if not possible_positions:
                possible_positions = [current_position]
                break
    if state.movements >= timeout_limit:
        print("Failed Simulation")
        state.movements = 0
        state.blocked_cell_actions = 0
        state.space_rat_detector_actions = 0
        state.failed_simulations += 1
        # return MOVEMENTS, BLOCKED_CELL_ACTIONS, SPACE_RAT_DETECTOR_ACTIONS
        return state.movements, state.blocked_cell_actions, state.space_rat_detector_actions, state.failed_simulations, state.actions

        
    print("Exit phase 1", flush=True)
    print("Begin phase 2", flush=True)
    # Phase 2 logic (if applicable)
    if len(possible_positions) == 1:
        print("Bot successfully localized at:", possible_positions[0], flush=True)
        random_rat_position = get_random_open_position(grid)
        if random_rat_position:
            main_phase_2(grid, possible_positions[0], random_rat_position, alpha, state)
        else:
            print("No cell for space rat", flush=True)
            sys.exit(0)
    else:
        print("Bot failed to localize", flush=True)
        sys.exit(0)
    print("Exit phase 2", flush=True)
    # return MOVEMENTS, BLOCKED_CELL_ACTIONS, SPACE_RAT_DETECTOR_ACTIONS
    return state.movements, state.blocked_cell_actions, state.space_rat_detector_actions, state.failed_simulations, state.actions


def run_simulation(alpha):
    # Your existing simulation logic here
    result = run_single_simulation(alpha)
    return result

def run_simulation_with_timeout(alpha, timeout=30):
    with multiprocessing.Pool(10) as pool:
        async_result = pool.apply_async(run_simulation, (alpha,))
        try:
            result = async_result.get(timeout)  # Timeout in seconds
            return result
        except multiprocessing.TimeoutError:
            print(f"Simulation with alpha {alpha} timed out.")
            pool.terminate()  # Forcefully terminate the process
            pool.join()  # Ensure resources are cleaned up
            return (0, 0, 0, 1, [])  # Return a failed simulation result as a placeholder

def main():
    alphas = np.arange(0.0, 1.1, 0.1)
    # alphas = np.arange(0.0, 0.225, 0.025)
    avg_actions = []
    avg_movements = []
    avg_blocked_actions = []
    avg_rat_detector_actions = []
    accuracies = []

    for alpha in alphas:
        print(f"Running simulations for alpha {alpha:.1f}", flush=True)
        num_simulations = 20
        total_movements = 0
        total_blocked_actions = 0
        total_rat_detector_actions = 0
        vs_plot = False
        results = []

        # Run simulations with timeout handling
        for _ in range(num_simulations):
            result = run_simulation_with_timeout(alpha, timeout=60)
            results.append(result)
            if result[4] and result[3] == 0 and not vs_plot:
                plot_movement_vs_detection(result[4], alpha)
                vs_plot = True

        # Aggregate results
        total_movements = sum(result[0] for result in results)
        total_blocked_actions = sum(result[1] for result in results)
        total_rat_detector_actions = sum(result[2] for result in results)
        failed_simulations = sum(result[3] for result in results)

        num_simulations -= failed_simulations
        if num_simulations > 0:  # Avoid division by zero
            total_all_actions = total_movements + total_blocked_actions + total_rat_detector_actions
            avg_actions.append(total_all_actions / num_simulations)
            avg_movements.append(total_movements / num_simulations)
            avg_blocked_actions.append(total_blocked_actions / num_simulations)
            avg_rat_detector_actions.append(total_rat_detector_actions / num_simulations)
            accuracies.append(num_simulations / (failed_simulations + num_simulations))
            print(f"Total movements: {total_movements}", flush=True)
            print(f"Avg movements: {total_movements / num_simulations}", flush=True)
            print(f"Total Blocked Actions: {total_blocked_actions}", flush=True)
            print(f"Avg Blocked Actions: {total_blocked_actions / num_simulations}", flush=True)
            print(f"Total Rat detector actions: {total_rat_detector_actions}", flush=True)
            print(f"Avg Rat detector actions: {total_rat_detector_actions / num_simulations}", flush=True)
            print(f"Total all actions: {total_all_actions}", flush=True)
            print(f"Avg no. of actions: {total_all_actions / num_simulations}", flush=True)
            print(f"Successful Simulation Count: {num_simulations}", flush=True)
            print(f"Failed Simulation Count: {failed_simulations}", flush=True)
            print(f"Accuracy: {num_simulations / (failed_simulations + num_simulations)}", flush=True)
            print("", flush=True)
        else:
            avg_actions.append(0)
            avg_movements.append(0)
            avg_blocked_actions.append(0)
            avg_rat_detector_actions.append(0)
            accuracies.append(0)

    # Plot the results
    plt.figure(figsize=(12, 12))
    # First plot for Average No. of Actions and Average Blocked Cell Actions
    plt.subplot(3, 1, 1)  # Changed from 2 to 3 subplots
    plt.plot(alphas, avg_actions, marker='o', label='Average No. of Actions')
    plt.plot(alphas, avg_blocked_actions, marker='x', label='Average Blocked Cell Actions')
    plt.xlabel('Alpha')
    plt.ylabel('Average No. of Actions')
    plt.title('Average No. of Actions and Average Blocked Cell Actions as a Function of Alpha')
    plt.legend()
    plt.grid(True)
    for i, avg in enumerate(avg_actions):
        plt.annotate(f'{avg:.2f}', (alphas[i], avg), textcoords="offset points", xytext=(0,10), ha='center')
    for i, avg in enumerate(avg_blocked_actions):
        plt.annotate(f'{avg:.2f}', (alphas[i], avg), textcoords="offset points", xytext=(0,-15), ha='center')

    # Second plot for Average Movements and Average Space Rat Detector Actions
    plt.subplot(3, 1, 2)
    plt.plot(alphas, avg_movements, marker='o', label='Average Movements')
    plt.plot(alphas, avg_rat_detector_actions, marker='s', label='Average Space Rat Detector Actions')
    plt.xlabel('Alpha')
    plt.ylabel('Average No. of Actions')
    plt.title('Average Movements and Average Space Rat Detector Actions as a Function of Alpha')
    plt.legend()
    plt.grid(True)
    for i, avg in enumerate(avg_movements):
        plt.annotate(f'{avg:.2f}', (alphas[i], avg), textcoords="offset points", xytext=(0,10), ha='center')
    for i, avg in enumerate(avg_rat_detector_actions):
        plt.annotate(f'{avg:.2f}', (alphas[i], avg), textcoords="offset points", xytext=(0,-15), ha='center')

    # Third plot for Accuracies
    plt.subplot(3, 1, 3)
    plt.plot(alphas, accuracies, marker='^', label='Accuracy', color='green')
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.title('Accuracy as a Function of Alpha')
    plt.legend()
    plt.grid(True)
    for i, acc in enumerate(accuracies):
        plt.annotate(f'{acc:.2f}', (alphas[i], acc), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    current_time_str = datetime.now().strftime("%d_%m_%y-%H_%M_%S")
    plt.savefig('./improved/final_analysis_improved_' + current_time_str + '.png')
    # End the timer
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    plt.show()

if __name__ == "__main__":
    main()