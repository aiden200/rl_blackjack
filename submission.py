# Originally created by Percy Liang, modified by Dave Musicant


import util, math, random
from collections import defaultdict, namedtuple
from util import FixedRLAlgorithm, ValueIteration, State, PossibleResult, Feature
from typing import List, Callable, Tuple, Any

############################################################

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues: List[int], multiplicity: int, threshold: int, peekCost: int):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self) -> State:
        return State(0, None, (self.multiplicity,) * len(self.cardValues))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state: Tuple) -> List[str]:
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    # Note: The grader script expects the outputs follow the same order as the cards.
    # For example, if the deck has face values: 1, 2, 3. You should order your corresponding
    # tuples in the same order.
    def succAndProbReward(self, state: State, action: str) -> List[PossibleResult]:
        # BEGIN_YOUR_CODE HERE; I'VE ADDED SOME LIMITED STARTING CODE
        if state.handTotal > self.threshold:
            return []

        if state.deckCounts == None:
            return []

        if action == 'Quit':
            newState = State(handTotal=state.handTotal,
                             nextCard=None,
                             deckCounts=None)  # specified in assignment
            return [PossibleResult(successor=newState,
                                   probability=1,
                                   reward=state.handTotal)]

        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions: Callable, discount: float, featureExtractor: Callable, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state: Tuple, action: Any) -> float:
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state: Tuple) -> Any:
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self) -> float:
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state: State, action: Any, reward: int, newState: State) -> None:
        # BEGIN_YOUR_CODE; I'VE STARTED IT, BUT WITH NOT MUCH

        featureValues = self.featureExtractor(state, action)
        for featureKey, featureValue in featureValues:
            self.weights[featureKey] += 10


        # END_YOUR_CODE

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state: Tuple, action: Any) -> List[Feature]:
    featureKey = (state, action)
    featureValue = 1
    return [Feature(featureKey, featureValue)]

############################################################
# (This was an exercise in the original version, but I just did it)
#
# As noted in the comments/documentation, util.simulate() is a function that takes as inputs an MDP and a particular RL algorithm you wish to run on the MDP.
# The RL algorithm will be an instance of the RLAlgorithm abstract class defined in util.py. 
# In this case, you’ll want to use the Q-learning algorithm that you implemented in 4(a). 
# Once you’re done calling simulate, your RL will have explored and learned a policy from the MDP. 
# You will also want to run value iteration on the same MDP to get a policy pi
# Now that you have your trained Q-learning policy and value iteration policy, you can examine/explore the two and see where/how they differ. 
# You’ll want to think about how you can extract/query the policy from your trained Q-learning algorithm object. 
# Note that you should be careful that when you’re examining the policy, this is the final, “optimal” policy (i.e. your algorithm should only exploit, not explore). 

# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

def simulate_QL_over_MDP(mdp: BlackjackMDP, featureExtractor: Callable):
    print()
    print("Doing Value Iteration and Q-learning...")
    qlearn = QLearningAlgorithm(mdp.actions, mdp.discount(), featureExtractor)
    util.simulate(mdp, qlearn, 30000)
    qlearn.explorationProb = 0

    alg = util.ValueIteration()
    alg.solve(mdp, .0001)

    mdp.computeStates()

    totalCount = 0
    correctCount = 0
    totalError = 0
    for state in mdp.states:
        totalCount += 1
        if alg.pi[state] == qlearn.getAction(state):
            correctCount += 1
        totalError += math.fabs(alg.V[state] - qlearn.getQ(state, qlearn.getAction(state)))
        # print(state, alg.V[state], qlearn.getQ(state, qlearn.getAction(state)))
        # print(state, alg.pi[state], qlearn.getAction(state))
    print("Total number of states:", totalCount)
    print("Number of states where Q-learning action matches value iteration action:", correctCount)
    print("Percent of matching states =", correctCount/totalCount*100)

    print()
    print("A random selection of states, the optimal action, and what qlearning says:")
    for state in random.choices(list(mdp.states), k=10):
        print(state, alg.pi[state], qlearn.getAction(state))

    # Now just try comparing simulations of each
    print()
    print("Now simulating average rewards from both approaches...")
    valueIterationAlgorithm = FixedRLAlgorithm(alg.pi)
    numIters = 10000
    valueIterationRewards = util.simulate(mdp, valueIterationAlgorithm, numIters)
    qlearnRewards = util.simulate(mdp, qlearn, numIters)
    print()
    print("Avg rewards from value iteration result = ", sum(valueIterationRewards)/numIters)
    print("Avg rewards from q-learning result = ", sum(qlearnRewards)/numIters)



############################################################
# Features for Q-learning.

# You should return a list of Features, where a Feature is a named tuple containing a (feature key, feature value).
# (See identityFeatureExtractor() above for a simple example.)
# Include only the following features in the list you return:
# -- Indicator for the action and the current total (1 feature).
#       The feature should be (('total', totalCardValueInHand, action),1). Feel free to use a different name.
# -- Indicator for the action and the presence/absence of each face value in the deck.
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the presence of each card is (1, 1, 0, 1)
#       The feature will be (('bitmask', (1, 1, 0, 1), action), 1). Feel free to use a different name. 
#       Note: only add this feature if the deck is not None.
# -- Indicators for the action and the number of cards remaining with each face value.
#       Example: if the deck is (3, 4, 0, 2), you should have four features (one for each face value).
#       The first feature will be ((0, 3, action), 1)
#       Note: only add these features if the deck is not None.
def blackjackFeatureExtractor(state: State, action: str) -> List[Feature]:

    # BEGIN_YOUR_CODE; I'VE WRITTEN A SMALL AMOUNT TO GET YOU STARTED.
    features: List[Feature] = []

    if state == None:
        return features

    features.append(Feature(featureKey=('total', state.handTotal, action), featureValue=1))

    return features



    # END_YOUR_CODE
