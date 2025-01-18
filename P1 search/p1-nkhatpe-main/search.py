# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    
    from util import Stack

    # Creating a stack for states to explore
    stack = Stack()
    
    ## Creating a set to keep track of explored states
    explored = set()

    # Initialize with the start state and an empty list of actions
    start_state = problem.getStartState()
    stack.push((start_state, []))

    while not stack.isEmpty():
        current_state, actions = stack.pop()

        # Check if the current state is the goal state
        if problem.isGoalState(current_state):
            return actions

        explored.add(current_state)

        # Generate successors and add them to the stack
        for next_state, action, _ in problem.getSuccessors(current_state):
            if next_state not in explored:
                stack.push((next_state, actions + [action]))

    return None # Return None if no path is found   


    util.raiseNotDefined()



def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    
    from util import Queue
    
    # Create a queue for states to explore
    queue = Queue()

    # Initialize with the start state
    queue.push((problem.getStartState(), []))

    # Create a set to keep track of explored states
    explored = set()

    # Create a set to keep track of seen states
    seen = set()

    while not queue.isEmpty():
        current_state, path = queue.pop()

        # Check if the current state is the goal state
        if problem.isGoalState(current_state):
            return path

        # Mark the current state as explored
        explored.add(current_state)

        # Generate successors and enqueue unexplored and unseen ones
        for next_state, action, _ in problem.getSuccessors(current_state):
            if next_state not in explored and next_state not in seen:
                next_path = path + [action]
                queue.push((next_state, next_path))
                seen.add(next_state)
                
    return None # Return None if no path is found  

    util.raiseNotDefined()


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    
    p_queue = PriorityQueue()
    
    # Use a dictionary to store explored states and their costs
    explored = {} 

    # Initialize with the start state and a cost of 0
    start_state = problem.getStartState()
    p_queue.push((start_state, [], 0), 0)

    while not p_queue.isEmpty():
        current_state, path, current_cost = p_queue.pop()

        # Skip this state if it has already been explored with a lower cost
        if current_state in explored and current_cost > explored[current_state]:
            continue

        # Mark the current state as explored with the current cost
        explored[current_state] = current_cost

        # Checking if the current state is the goal state
        if problem.isGoalState(current_state):
            return path

        # Generate successors and add them to the priority queue
        for next_state, action, step_cost in problem.getSuccessors(current_state):
            if next_state not in explored or current_cost + step_cost < explored[next_state]:
                next_path = path + [action]
                next_cost = current_cost + step_cost
                p_queue.push((next_state, next_path, next_cost), next_cost)

    return None  # Return None if no path is found


    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    from util import PriorityQueue
    
    # Creating a priority queue for states to explore
    as_pqueue = PriorityQueue()

    # Initializing with the start state and a cost of 0
    start_state = problem.getStartState()
    as_pqueue.push((start_state, [], 0), 0)

    # Creating a dictionary to keep track of explored states and their costs
    explored = {}

    while not as_pqueue.isEmpty():
        current_state, path, current_cost = as_pqueue.pop()

        # Check if the current state is in the explored one with a lower cost
        if current_state in explored and current_cost >= explored[current_state]:
            continue

        # Update the explored with the current cost
        explored[current_state] = current_cost

        # Checking if the current state is the goal state
        if problem.isGoalState(current_state):
            return path

        # Generating successors and add them to the priority queue
        for next_state, action, step_cost in problem.getSuccessors(current_state):
            if next_state not in explored or current_cost + step_cost < explored[next_state]:
                next_path = path + [action]
                g_cost = current_cost + step_cost
                h_cost = heuristic(next_state, problem)
                f_cost = g_cost + h_cost
                as_pqueue.push((next_state, next_path, g_cost), f_cost)

    return None  # Return None if no path is found



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
