"""Local copy of RefineEvo's initialization prompt assets.

The MCTS-Recipe runner deliberately does not read the separate RefineEvo
repository at runtime.  These strings mirror the initialization assets used by
RefineEvo and are versioned with LLM4AD so experiments are reproducible on both
Windows and Linux.
"""

from __future__ import annotations

import re

from llm4ad.base import SampleTrimmer


SYSTEM_GENERATOR = """You are a world-class expert in optimization algorithms and heuristic design with deep expertise in operations research, combinatorial optimization, and metaheuristic methods.

## YOUR ROLE
Design novel, efficient, and effective heuristic algorithms to solve complex optimization problems.

## PROBLEM CONTEXT
**Function Name:** {func_name}
**Problem Description:** {problem_desc}
**Detailed Specification:**
{func_desc}

## ALGORITHM DESIGN PRINCIPLES
1. **Novelty**: Create algorithms with unique characteristics that differ from existing approaches
2. **Efficiency**: Ensure computational efficiency suitable for the problem scale
3. **Robustness**: Design algorithms that perform well across diverse problem instances

## OUTPUT FORMAT REQUIREMENTS
You must provide your response in exactly two parts:

**Part 1: Algorithm Description**
- Write a concise one-sentence description of your algorithm
- Enclose the description in curly braces: {{your algorithm description here}}
- Focus on the core innovation and key mechanism

**Part 2: Python Implementation**
- Provide complete, runnable Python code
- Enclose code in: ```python ... ```
- Follow the exact function signature specified
- Include necessary imports within the function
- Use clear variable names and efficient data structures
- Ensure code is production-ready without bugs

## IMPORTANT CONSTRAINTS
- Do NOT provide any additional explanations, commentary, or discussion
- Do NOT include example usage or test cases
- Output ONLY the algorithm description in braces and the Python code block
- Ensure the code is syntactically correct and logically sound"""


USER_INIT = """## TASK: Generate Initial Algorithm

You are tasked with designing a novel heuristic algorithm from scratch. This is the initial design phase where you will create the foundation for the evolutionary algorithm development process.

### REFERENCE IMPLEMENTATION
Below is a baseline implementation to illustrate the expected code structure and format:

{seed_func}

### ADDITIONAL DOMAIN KNOWLEDGE
{external_knowledge}
"""


PROFILES = {
    "tsp_constructive": {
        "problem_desc": "Solving Traveling Salesman Problem (TSP) with constructive heuristics. TSP requires finding the shortest path that visits all given nodes and returns to the starting node.",
        "func_desc": "The 'select_next_node' function takes as input the current node, the destination_node, a set of unvisited nodes, and a distance matrix, and returns the next node to visit.",
        "seed_func": """import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    \"\"\"Select the next node to visit from the unvisited nodes.\"\"\"
    threshold = 0.7
    c1, c2, c3, c4 = 0.4, 0.3, 0.2, 0.1
    scores = {}
    for node in unvisited_nodes:
        all_distances = [distance_matrix[node][i] for i in unvisited_nodes if i != node]
        average_distance_to_unvisited = np.mean(all_distances)
        std_dev_distance_to_unvisited = np.std(all_distances)
        score = c1 * distance_matrix[current_node][node] - c2 * average_distance_to_unvisited + c3 * std_dev_distance_to_unvisited - c4 * distance_matrix[destination_node][node]
        scores[node] = score
    next_node = min(scores, key=scores.get)
    return next_node
""",
        "external_knowledge": "- Try look-ahead mechanisms.",
    },
    "cvrp_constructive": {
        "problem_desc": "Solving the Capacitated Vehicle Routing Problem (CVRP) with constructive heuristics. CVRP requires finding routes for a fleet of vehicles with limited capacity to serve all customers while minimizing the total travel distance.",
        "func_desc": "The 'select_next_node' function takes as input the current node ID, the depot ID, an array of unvisited node IDs, the vehicle's remaining capacity, the demand array of all nodes, and the distance matrix, and returns the ID of the next node to visit (or 0 to return to the depot and start a new route).",
        "seed_func": """import numpy as np

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: float, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    \"\"\"Greedy nearest-neighbor: pick the closest feasible unvisited customer.\"\"\"
    best_node = depot
    best_dist = float('inf')
    for node in unvisited_nodes:
        if demands[node] <= rest_capacity:
            d = distance_matrix[current_node][node]
            if d < best_dist:
                best_dist = d
                best_node = node
    return best_node
""",
        "external_knowledge": """- Consider savings-based criteria (Clarke-Wright) to merge routes efficiently.
- Use demand-to-distance ratios to balance load and travel cost.
- Look-ahead mechanisms that estimate the impact of visiting a node on future routing can be effective.
- Returning to the depot early (before capacity is fully used) may sometimes reduce overall tour length.
- Combining distance to the current node, distance back to depot, and remaining demand can yield strong heuristics.""",
    },
    "vrptw_constructive": {
        "problem_desc": "Solving the Vehicle Routing Problem with Time Windows (VRPTW) with constructive heuristics. VRPTW requires finding routes for a fleet of vehicles with limited capacity to serve all customers within their time windows while minimizing total travel distance.",
        "func_desc": "The 'select_next_node' function takes as input the current node ID, the depot ID, an array of feasible unvisited node IDs, the vehicle's remaining capacity, the current time, the demand array of all nodes, the distance matrix, and the time-window array (shape: [n_nodes, 2] with [earliest, latest]), and returns the ID of the next node to visit (or 0 to return to the depot and start a new route).",
        "seed_func": """import numpy as np

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: float, current_time: float, demands: np.ndarray, distance_matrix: np.ndarray, time_windows: np.ndarray) -> int:
    \"\"\"Earliest-deadline-first greedy heuristic for VRPTW.\"\"\"
    best_node = depot
    best_due = float('inf')
    for node in unvisited_nodes:
        if demands[node] > rest_capacity:
            continue
        travel = distance_matrix[current_node][node]
        arrival = current_time + travel
        earliest, latest = time_windows[node]
        if arrival > latest:
            continue
        if latest < best_due:
            best_due = latest
            best_node = node
    return best_node
""",
        "external_knowledge": """- Earliest deadline first (EDF): prioritize customers whose time window closes soonest.
- Combine urgency (time window tightness) with distance to balance travel cost and feasibility.
- Look-ahead: check if visiting a node still allows returning to depot within the depot's time window.
- Consider waiting time: arriving early forces waiting, which costs time but not distance.
- Savings-based criteria: evaluate the marginal cost of inserting a customer vs. opening a new route.
- When no feasible customer exists, return to depot (0) to reset capacity and time.""",
    },
    "kp_constructive": {
        "problem_desc": "Solving the 0/1 Knapsack Problem with constructive heuristics. The goal is to select a subset of items maximizing total value without exceeding the knapsack capacity.",
        "func_desc": "The select_next_item function takes as input the remaining capacity, a numpy array of item values, and a numpy array of item weights, and returns the next item to pick.",
        "seed_func": """import numpy as np

def select_next_item(remaining_capacity, weights, values):
    \"\"\"
    Greedy constructive
    \"\"\"
    feasible_mask = weights <= remaining_capacity
    ratios = values / weights

    feasible_indices = np.where(feasible_mask)[0]
    best_local_idx = np.argmax(ratios[feasible_mask])
    best_index = feasible_indices[best_local_idx]

    return best_index
""",
        "external_knowledge": "",
    },
    "bpp_online": {
        "problem_desc": "Solving online one-dimensional bin packing with constructive heuristics. Items arrive sequentially and must be assigned to bins while minimizing the number of used bins.",
        "func_desc": "The priority function takes as input an item and an array of bins_remain_cap (containing the remaining capacity of each bin) and returns a priority score for each bin. The bin with the highest priority score will be selected for the item.",
        "seed_func": """def priority(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
  \"\"\"Best known heuristics.\"\"\"
  max_bin_cap = max(bins_remain_cap)
  score = (bins_remain_cap - max_bin_cap)**2 / item + bins_remain_cap**2 / (item**2)
  score += bins_remain_cap**2 / item**3
  score[bins_remain_cap > item] = -score[bins_remain_cap > item]
  score[1:] -= score[:-1]
  return score
""",
        "external_knowledge": "",
    },
}


ALIASES = {
    "tsp_construct": "tsp_constructive",
    "cvrp_construct": "cvrp_constructive",
    "vrptw_construct": "vrptw_constructive",
    "knapsack_construct": "kp_constructive",
    "kp_construct": "kp_constructive",
    "bp_1d_construct": "bpp_online",
}


def problem_name(template_module: str) -> str:
    return ALIASES.get(template_module.split(".")[-2], template_module.split(".")[-2])


def initial_messages(method_name: str, template_module: str):
    name = problem_name(template_module)
    profile = PROFILES.get(name)
    if profile is None:
        raise KeyError(f"No local RefineEvo initialization profile for {name!r}")
    system = SYSTEM_GENERATOR.format(
        func_name=method_name,
        problem_desc=profile["problem_desc"],
        func_desc=profile["func_desc"],
    )
    user = USER_INIT.format(
        seed_func=profile["seed_func"],
        external_knowledge=profile["external_knowledge"]
        or "No additional domain knowledge provided.",
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def extract_algorithm_and_code(response: str) -> tuple[str, str]:
    """Parse an initialization response with RefineEvo's exact format rules."""
    if not isinstance(response, str):
        raise ValueError("LLM initialization response must be text.")
    algorithm_match = re.search(r"\{(.*?)\}", response, re.DOTALL)
    if not algorithm_match:
        raise ValueError("LLM response missing algorithm description block.")
    code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if not code_match:
        raise ValueError("LLM response missing python code block.")
    algorithm = algorithm_match.group(1)
    code = code_match.group(1)
    if not algorithm or not code:
        raise ValueError("Failed to extract algorithm or code from LLM response.")
    return code, algorithm


def sample_initial_algorithm(llm, messages, template_program):
    """Generate one initial Function using the local RefineEvo init contract.

    The prompt assets and parser are local to LLM4AD.  The returned code is
    converted to the repository's Function representation only after it has
    passed RefineEvo's strict description/code extraction.
    """
    response = llm.draw_sample(messages)
    code, algorithm = extract_algorithm_and_code(response)
    # RefineEvo keeps the complete function for its standalone evaluator;
    # LLM4AD's Function object stores only the body, so use the same trimming
    # convention as the normal LLM4AD sampler at this boundary.
    function_body = SampleTrimmer.trim_preface_of_function(code)
    function = SampleTrimmer.sample_to_function(function_body, template_program)
    if function is None:
        raise ValueError("Generated initialization code could not be converted.")
    return algorithm, function
