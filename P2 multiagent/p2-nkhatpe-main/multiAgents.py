# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # Get the current position of Pacman in the current game state
        currentPos = currentGameState.getPacmanPosition()

        # Initialize the evaluation score with the score of the successor game state
        score = successorGameState.getScore()

        # Calculate the distance to the nearest ghost from Pacman's new position
        ghost_distances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        min_ghost_distance = min(ghost_distances)

        # If Pacman is about to get eaten by a ghost and there's no invulnerability, give a very negative score
        if min_ghost_distance == 1 and sum(newScaredTimes) == 0:
            score -= 1000

        # Calculate the reciprocal of the distance to the nearest food pellet from Pacman's new position
        food_distances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if food_distances:
            min_food_distance = min(food_distances)
            score += 1.0 / (min_food_distance + 1)

        # Return the final evaluation score for the successor game state
        return score

        return successorGameState.getScore()



def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
              
        # Initialize a variable to keep track of the best action and its value
        bestAction, _ = self.minimax(gameState, self.depth, 0)
        return bestAction

    def minimax(self, gameState, depth, agentIndex):
        # Check if we've reached the end of the search or a terminal state
        if depth == 0 or gameState.isWin() or gameState.isLose():
            # Return None as there is no specific action at this point, and evaluate the game state
            return None, self.evaluationFunction(gameState)

        # Get the list of legal actions for the current agent
        legalActions = gameState.getLegalActions(agentIndex)

        if agentIndex == 0:  # Max (Pacman's) turn
            bestAction, bestValue = None, float('-inf')
            for action in legalActions:
                # Generate a successor state by applying the action
                successorState = gameState.generateSuccessor(agentIndex, action)
                # Recursively call minimax for the next agent (Ghost) and get their best value
                _, value = self.minimax(successorState, depth, 1)
                # Update the best action and value if a better action is found
                if value > bestValue:
                    bestValue = value
                    bestAction = action
            return bestAction, bestValue
        else:  # Min (Ghost's) turn
            bestAction, bestValue = None, float('inf')
            for action in legalActions:
                # Generate a successor state by applying the action
                successorState = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    # If the last ghost has taken a turn, decrement the depth
                    _, value = self.minimax(successorState, depth - 1, 0)
                else:
                    # Otherwise, continue with the next ghost's turn
                    _, value = self.minimax(successorState, depth, agentIndex + 1)
                # Update the best action and value if a better action is found
                if value < bestValue:
                    bestValue = value
                    bestAction = action
            return bestAction, bestValue
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Call the alpha-beta pruning function and return the best action
        bestAction, _ = self.alphaBetaPruning(gameState, self.depth, 0, float('-inf'), float('inf'))
        return bestAction

    def alphaBetaPruning(self, gameState, depth, agentIndex, alpha, beta):
        """
        Alpha-beta pruning algorithm for selecting actions with pruning
        """
        if depth == 0 or gameState.isWin() or gameState.isLose():
            # Return None as there is no specific action at this point, and evaluate the game state
            return None, self.evaluationFunction(gameState)

        legalActions = gameState.getLegalActions(agentIndex)

        if agentIndex == 0:  # Max (Pacman's) turn
            bestAction, bestValue = None, float('-inf')
            for action in legalActions:
                # Generate a successor state by applying the action
                successorState = gameState.generateSuccessor(agentIndex, action)
                # Recursively call alpha-beta pruning for the next agent (Ghost) and get their best value
                _, value = self.alphaBetaPruning(successorState, depth, 1, alpha, beta)
                # Update the best action and value if a better action is found
                if value > bestValue:
                    bestValue = value
                    bestAction = action
                # Update alpha with the maximum value
                alpha = max(alpha, bestValue)
                # Perform pruning if the best value is greater than beta
                if bestValue > beta:
                    break
            return bestAction, bestValue
        else:  # Min (Ghost's) turn
            bestAction, bestValue = None, float('inf')
            for action in legalActions:
                # Generate a successor state by applying the action
                successorState = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    # If the last ghost has taken a turn, decrement the depth
                    _, value = self.alphaBetaPruning(successorState, depth - 1, 0, alpha, beta)
                else:
                    # Otherwise, continue with the next ghost's turn
                    _, value = self.alphaBetaPruning(successorState, depth, agentIndex + 1, alpha, beta)
                # Update the best action and value if a better action is found
                if value < bestValue:
                    bestValue = value
                    bestAction = action
                # Update beta with the minimum value
                beta = min(beta, bestValue)
                # Perform pruning if the best value is less than alpha
                if bestValue < alpha:
                    break
            return bestAction, bestValue


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Call the expectimax function and return the best action
        bestAction, _ = self.expectimax(gameState, self.depth, 0)
        return bestAction

    def expectimax(self, gameState, depth, agentIndex):
        """
        Expectimax algorithm for selecting actions
        """
        # Base case: If we've reached the maximum depth or a terminal state, return the evaluation function value
        if depth == 0 or gameState.isWin() or gameState.isLose():
            # Return None as there is no specific action at this point, and evaluate the game state
            return None, self.evaluationFunction(gameState)

        # Get the legal actions for the current agent
        legalActions = gameState.getLegalActions(agentIndex)

        if agentIndex == 0:  # Max player's (Pacman's) turn
            bestAction, bestValue = None, float('-inf')
            # Iterate over legal actions and compute the best action and its associated value
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                # Recursively call expectimax for the next agent and get their value
                _, value = self.expectimax(successorState, depth, 1)
                # Update the best action and best value based on the max player's logic
                if value > bestValue:
                    bestValue = value
                    bestAction = action
            return bestAction, bestValue
        else:  # Expected value (Ghost's) turn
            expectedValue = 0.0
            numActions = len(legalActions)
            # Iterate over legal actions and calculate the expected value
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    # If the last ghost has taken a turn, decrement the depth
                    _, value = self.expectimax(successorState, depth - 1, 0)
                else:
                    # Otherwise, continue with the next ghost's turn
                    _, value = self.expectimax(successorState, depth, agentIndex + 1)
                expectedValue += value
            expectedValue /= numActions  # Calculate the expected value
            return None, expectedValue
        util.raiseNotDefined()

        
        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <This function evaluates the state based on the following factors:
    1. The score of the current game state.
    2. The reciprocal of the distance to the nearest food pellet.
    3. The reciprocal of the distance to the nearest capsule.
    4. The distance to the nearest ghost when Pacman is not invulnerable.

    The evaluation function assigns a higher score to states with higher scores,
    states where Pacman is closer to food and capsules, and states where Pacman is far from ghosts.

    :param currentGameState: The current game state (GameState object).
    :return: A numeric score representing the evaluation of the state.
    In this evaluation function, I have adjusted the weights and added additional conditions to ensure 
    that Pacman avoids ghosts when they are nearby and prioritize eating food and capsules.>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    foodGrid = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()

    # Calculate the reciprocal of the distance to the nearest food pellet
    food_distances = [manhattanDistance(pacmanPos, food) for food in foodGrid.asList()]
    if food_distances:
        min_food_distance = min(food_distances)
    else:
        min_food_distance = 0  # No food left

    # Calculate the reciprocal of the distance to the nearest capsule
    capsule_distances = [manhattanDistance(pacmanPos, capsule) for capsule in capsules]
    if capsule_distances:
        min_capsule_distance = min(capsule_distances)
    else:
        min_capsule_distance = float('inf')  # No capsules left

    # Calculate the distance to the nearest non-scared ghost
    ghost_distances = [manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghostStates if not ghost.scaredTimer]
    if ghost_distances:
        min_ghost_distance = min(ghost_distances)
    else:
        min_ghost_distance = float('inf')  # No non-scared ghosts

    # Define weights for each factor
    score_weight = 1.0
    food_distance_weight = 3.0  # Maximize food proximity
    capsule_distance_weight = 2.0  # Maximize capsule proximity
    ghost_distance_weight = -4.0  # Strongly avoid non-scared ghosts

    # Calculate the final evaluation score
    evaluation = (
        score_weight * score +
        food_distance_weight / (min_food_distance + 1) +
        capsule_distance_weight / (min_capsule_distance + 1) +
        ghost_distance_weight / (min_ghost_distance + 1)
    )

    return evaluation
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
