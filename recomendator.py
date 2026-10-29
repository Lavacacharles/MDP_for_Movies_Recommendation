from bs4 import BeautifulSoup
from tqdm import tqdm  
from tqdm import tqdm
from datetime import datetime
from scipy.stats import uniform
from dateutil.parser import parse
from scipy.interpolate import UnivariateSpline

import itertools
import requests
import numpy as np 
import pandas as pd 
import os
import pickle

class MDP:
    def __init__(
            self, 
            invertedStatesDict, 
            statesDict, 
            probTransitions, 
            categoriasDict, 
            invertedCategoriasDict, 
            arrTransitions, 
            mdpTransitions=None,
            values=None,
            policy=None,
            preInitialized=False,
            gamma=0.9,
            theta=1e-6, 
            user_weight=0.6
            ):
        self.invertedStatesDict = invertedStatesDict
        self.statesDict = statesDict

        
        self.probTransitions = probTransitions
        self.categoriasDict = categoriasDict
        self.invertedCategoriasDict = invertedCategoriasDict
        
        rewards = np.array([
            1.0 if like == "like" and beh == "slowSkip" else
            0.5 if like == "noReaction" and beh == "fastSkip" else
            -1.0 if beh == "outApp" else
            0.0
            for like, cat, beh in list(invertedStatesDict.keys())
        ])
        self.rewards = rewards
        self.arrTransitions = arrTransitions
        

        self.num_actions = len(categoriasDict)
        self.num_states = probTransitions.shape[0] 

        self.gamma = gamma
        self.theta = theta
        self.user_weight = user_weight
        if not preInitialized:
            self.P = np.zeros((self.num_actions, self.num_states, self.num_states))
            self.policy = np.zeros(self.num_states, dtype=int)
            self.V = np.zeros(self.num_states)
            self._build_transition_tensor()
            self.policy_iteration()
        else:
            self.P = mdpTransitions
            self.policy = policy
            self.V = values
            
    def _build_transition_tensor(self):
        P_single = self.arrTransitions
        for s in range(self.num_states):
            for s_next in range(self.num_states):
                cat_next = self.statesDict[s_next][1]
                a = self.invertedCategoriasDict[cat_next]
                self.P[a][s][s_next] = P_single[s][s_next]

        for a in range(self.num_actions):
            row_sums = self.P[a].sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            self.P[a] = self.P[a] / row_sums

        N,_ = self.P[0].shape
        soft_prob = 0.9
        rest_prob = (1 - soft_prob)/(N-1)
        for a in range(self.num_actions):
            sumRow = self.P[a].max(axis=1, keepdims=True)
            for idx, i in enumerate(sumRow):
                if i >= 1 :
                    ind_j = np.argsort(self.P[a][idx,:])[-1]
                    self.P[a][idx,:] = np.repeat(rest_prob, N)
                    self.P[a][idx,ind_j] = soft_prob
                
    def getMetrics(self, state):
        idxState = self.invertedStatesDict[state]
        return {
            "rewards":float(self.rewards[idxState]),
            "policy":self.categoriasDict[self.policy[idxState]],
            "value":float(self.V[idxState]),
            }

    def policy_iteration(self):
        P, R, gamma, theta = self.P, self.rewards, self.gamma, self.theta
        policy = np.zeros(self.num_states, dtype=int)
        V = np.zeros(self.num_states)

        while True:
            while True:
                delta = 0
                for s in range(self.num_states):
                    a = policy[s]
                    v_new = sum(P[a][s, s_next] * (R[s_next] + gamma * V[s_next])
                                for s_next in range(self.num_states))
                    delta = max(delta, abs(v_new - V[s]))
                    V[s] = v_new
                if delta < theta:
                    break

            policy_stable = True
            for s in range(self.num_states):
                old_action = policy[s]
                q_vals = [sum(P[a][s, s_next] * (R[s_next] + gamma * V[s_next])
                              for s_next in range(self.num_states))
                          for a in range(self.num_actions)]
                new_action = int(np.argmax(q_vals))
                policy[s] = new_action
                if old_action != new_action:
                    policy_stable = False

            if policy_stable:
                break

        self.V = V
        self.policy = policy
        return V, policy

    def update(self,
               arrTransitions=None,
               rewards=None,
               gamma=None,
               theta=None):
        if arrTransitions is not None:
            self.arrTransitions = arrTransitions
            self.probTransitions = arrTransitions
        if rewards is not None:
            self.rewards = rewards
        if gamma is not None:
            self.gamma = gamma
        if theta is not None:
            self.theta = theta
        self._build_transition_tensor()
        return self.policy_iteration()

    def fit_user(self, visits):
        counts = np.zeros((self.num_states, self.num_states))
        indices = [self.invertedStatesDict[v] for v in visits]
        for i in range(len(indices) - 1):
            s, s_next = indices[i], indices[i+1]
            counts[s, s_next] += 1
        
        counts = counts + self.arrTransitions
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        P_user = counts / row_sums
        P_combined = ((1 - self.user_weight) * self.probTransitions
                      + self.user_weight * P_user)
        
        rs = P_combined.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        P_combined = P_combined / rs
        
        return self.update(arrTransitions=P_combined)
    
    
    def recommend(self, state_index, n_steps=1, method='optimal'):
        path = []
        current = state_index
        for _ in range(n_steps):
            if method == 'optimal':
                action = int(self.policy[current])
            elif method == 'random':
                action = np.random.choice(self.num_actions)
            else:
                raise ValueError("method debe ser 'optimal' o 'random'")
            path.append((self.statesDict[current], self.categoriasDict[action]))
            # sample next state según P[action]
            probs = self.P[action][current]
            current = int(np.random.choice(self.num_states, p=probs))
        return path

# Ejecutar Policy Iteration
# V, pi = policy_iteration(P, rewards)


# Ejecutar Policy Iteration
# V, pi = policy_iteration(P, rewards)


# def getRecomendation(model :MDP, current_state): # Explore True, Exploit False
    # pass 

# Ejecutar Policy Iteration
# V, pi = policy_iteration(P, rewards)
