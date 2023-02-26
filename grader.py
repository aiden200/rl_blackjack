#!/usr/bin/env python3

# Originally created by Percy Liang, modified by Dave Musicant

import random, util, collections
import graderUtil
from submission import State, PossibleResult

random.seed(12345)
grader = graderUtil.Grader()
submission = grader.load('submission')

############################################################
# Setting up the MDP

def test3a0():
    mdp1 = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                   threshold=10, peekCost=1)
    startState = mdp1.startState()
    preBustState = State(6, None, (1, 1))
    postBustState = State(11, None, None)

    mdp2 = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                   threshold=15, peekCost=1)
    preEmptyState = State(11, None, (1,0))

    mdp3 = submission.BlackjackMDP(cardValues=[1, 2, 3], multiplicity=3,
                                   threshold=8, peekCost=1)
    mdp3_startState = mdp3.startState()
    mdp3_preBustState = State(6, None, (1, 1, 1))
    mdp3_unevenState = State(1, None, (2, 3, 3))

    # Make sure the succAndProbReward function is implemented correctly.
    tests = [
        ([PossibleResult(State(1, None, (1, 2)), 0.5, 0),
          PossibleResult(State(5, None, (2, 1)), 0.5, 0)], mdp1, startState, 'Take'),
        ([PossibleResult(State(0, 0, (2, 2)), 0.5, -1),
          PossibleResult(State(0, 1, (2, 2)), 0.5, -1)], mdp1, startState, 'Peek'),
        ([PossibleResult(State(0, None, None), 1, 0)], mdp1, startState, 'Quit'),
        ([PossibleResult(State(7, None, (0, 1)), 0.5, 0),
          PossibleResult(State(11, None, None), 0.5, 0)], mdp1, preBustState, 'Take'),
        ([], mdp1, postBustState, 'Take'),
        ([], mdp1, postBustState, 'Peek'),
        ([], mdp1, postBustState, 'Quit'),
        ([PossibleResult(State(12, None, None), 1, 12)], mdp2, preEmptyState, 'Take'),
        ([PossibleResult(State(1, None, (2, 3, 3)), 1/3, 0),
          PossibleResult(State(2, None, (3, 2, 3)), 1/3, 0),
          PossibleResult(State(3, None, (3, 3, 2)), 1/3, 0)], mdp3, mdp3_startState, 'Take'),
        ([PossibleResult(State(7, None, (0, 1, 1)), 1/3, 0),
          PossibleResult(State(8, None, (1, 0, 1)), 1/3, 0),
          PossibleResult(State(9, None, None), 1/3, 0)], mdp3, mdp3_preBustState, 'Take'),
        ([PossibleResult(State(2, None, (1, 3, 3)), 2/8, 0),
          PossibleResult(State(3, None, (2, 2, 3)), 3/8, 0),
          PossibleResult(State(4, None, (2, 3, 2)), 3/8, 0)], mdp3, mdp3_unevenState, 'Take'),
        ([PossibleResult(State(6, None, None), 1, 6)], mdp3, mdp3_preBustState, 'Quit'),
        ([], mdp1, State(0, None, None), 'Take')
    ]
    for gold, mdp, state, action in tests:
        if not grader.require_is_equal(gold,
                                       mdp.succAndProbReward(state, action)):
            print(('   state: {}, action: {}'.format(state, action)))
grader.add_basic_part('mdp-basic', test3a0, 5, description="Basic test for succAndProbReward() that covers several edge cases.")

def test3a1():
    mdp = submission.BlackjackMDP(cardValues=[1, 3, 5, 7, 9], multiplicity=3,
                                  threshold=30, peekCost=1)
    startState = mdp.startState()
    alg = util.ValueIteration()
    alg.solve(mdp, .0001)

    grader.require_is_less_than(.001, abs(26.00303030303 - alg.V[startState]))
    grader.require_is_less_than(.001, abs(26.12676767676 - alg.V[(11, 3, (2, 2, 3, 2, 3))]))
    grader.require_is_less_than(.001, abs(24.37499999999 - alg.V[(21, 1, (1, 3, 1, 3, 2))]))

grader.add_basic_part('mdp-values', test3a1, 5, description="Test for running ValueIteration on BlackjackMDP.")

############################################################
# Test Q-learning

def test4a():
    mdp = util.NumberLineMDP()
    mdp.computeStates()
    rl = submission.QLearningAlgorithm(mdp.actions, mdp.discount(),
                                       submission.identityFeatureExtractor,
                                       0)
    # We call this here so that the stepSize will be 1
    rl.numIters = 1

    rl.incorporateFeedback(0, 1, 0, 1)
    grader.require_is_equal(0, rl.getQ(0, -1))
    grader.require_is_equal(0, rl.getQ(0, 1))

    rl.incorporateFeedback(1, 1, 1, 2)
    grader.require_is_equal(0, rl.getQ(0, -1))
    grader.require_is_equal(0, rl.getQ(0, 1))
    grader.require_is_equal(0, rl.getQ(1, -1))
    grader.require_is_equal(1, rl.getQ(1, 1))

    rl.incorporateFeedback(2, -1, 1, 1)
    grader.require_is_equal(1.9, rl.getQ(2, -1))
    grader.require_is_equal(0, rl.getQ(2, 1))

grader.add_basic_part('qlearn-basic', test4a, 5, max_seconds=10, description="Basic test for incorporateFeedback() using NumberLineMDP.")



def qlearnIdSmall():
    submission.simulate_QL_over_MDP(submission.smallMDP, submission.identityFeatureExtractor)
grader.add_basic_part('qlearn-id-small', qlearnIdSmall, 0, max_seconds=60, description="Q-learning simulations for small MDP, id extractor.")

def qlearnIdLarge():
    submission.simulate_QL_over_MDP(submission.largeMDP, submission.identityFeatureExtractor)
grader.add_basic_part('qlearn-id-large', qlearnIdLarge, 0, max_seconds=60, description="Q-learning simulations for large MDP, id extractor.")

def qlearnBjSmall():
    submission.simulate_QL_over_MDP(submission.smallMDP, submission.blackjackFeatureExtractor)
grader.add_basic_part('qlearn-bj-small', qlearnBjSmall, 0, max_seconds=120, description="Q-learning simulations for small MDP, blackjack extractor.")

def qlearnBjLarge():
    submission.simulate_QL_over_MDP(submission.largeMDP, submission.blackjackFeatureExtractor)
grader.add_basic_part('qlearn-bj-large', qlearnBjLarge, 0, max_seconds=120, description="Q-learning simulations for large MDP, blackjack extractor.")


grader.grade()
