import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class PokeNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PokeNet, self).__init__()
        self.network == nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)


class Agent:
    
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = []
        self.batch_size = 64
        
        self.model = PokeNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def get_state(self, state):
        
        flattened = []
        
        # Flatten the dictionary into a 1D vector with values from 0 to 1.
        def _flatten(d, prefix=""):
            
            for key in sorted(d.keys()):
                value = d[key]
                
                if isinstance(value, dict):
                    _flatten(value, prefix+key+"_")
                elif isinstance(value, list):
                    for i in range(len(value)):
                        _flatten(value[i], prefix+key+str(i)+"_")
                else:
                    # Normalisation. Max values are found from the data. Improve this by making it actually pull those maxes.
                    if "request" in key:
                        value = value/3
                    elif "pokeid" in key:
                        value = value/1293
                    elif "type" in key or "teraType" in key:
                        value = value/19
                    elif "ability" in key:
                        value = value/307
                    elif "item" in key:
                        value = value/276
                    elif "moveid" in key:
                        value = value/919
                    elif "pp" in key:
                        value = value/64
                    elif "category" in key:
                        value = value/2
                    elif "power" in key:
                        value = value/300
                    elif "Mod" in key:
                        value = 0.5+(value/12) # 0 means the stat is at it's minimum. 1 means it's at it's max, while 0.5 is neutral.
                    elif key in ["atk", "def", "spa", "spd", "spe"]:
                        value = value/2192 # Theoretical max stat.
                    elif "status" in key:
                        value = value/7
                    elif "baseSpeed" in key:
                        value = value/300
                    elif "toxicspikes" in key:
                        value = value/2
                    elif "spikes" in key:
                        value = value/3
                    elif "weather" in key:
                        value = value/4

                    flattened.append(float(value))
        
        _flatten(state)
        return torch.FloatTensor(flattened) # tensor of the flattened state. This is something the agent can interpret.
    
    # Make an action from a list of valid ones and the state. 
    def act(self, state, valid_actions):
        state_tensor = self.get_state(state)
        
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        q_values = self.model(state_tensor)
        valid_q = [q_values[i] for i in valid_actions]
        return valid_actions(np.argmax(valid_q))

    # Append what's just occured and the result of that action to memory.
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    # Replay part of memory to learn from it
    def replay(self):
        # If there's not enough memory, don't bother.
        if len(self.memory) < self.batch_size:
            return
        
        # Take a random sample of the memory to learn from.
        batch = random.sample(self.memory, self.batch_size)
        
        # Iterate over everything in the batch, and learn from it.
        # This is done by calculating the target value, and then updating the model based on the difference between the current Q value and the target.
        # This is the Bellman equation. The target is the reward plus the discounted future reward.
        for state, action, reward, next_state, done in batch:
            state_tensor = self.get_state(state)
            next_state_tensor = self.get_state(next_state)
            
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state_tensor))
                
            current_q = self.model(state_tensor)[action]
            
            loss = self.criterion(current_q, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # This is the epsilon decay. It's a way to make the agent less random over time.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        


    #Select randomly for now        
    def getAction(self, state, tag):

        if 'active' in state:
            invalidMove = True

            # Pokémon can be locked into moves, so we need to check that the chosen move hasn't been disabled.
            while invalidMove:
                randomMove = random.choice(state["active"][0]["moves"])
                if("disabled" not in randomMove.keys()):
                    invalidMove = False
                    return tag+'|/choose move '+randomMove["id"]+"|"+str(state["rqid"])
                elif (not randomMove["disabled"]):
                    invalidMove = False
                    return tag+'|/choose move '+randomMove["id"]+"|"+str(state["rqid"])
            
        if 'forceSwitch' in state:
            invalidMon = True
            
            while invalidMon:
                randomMon = random.randint(0,5)
                # Don't send in a fainted mon or try to send in the current one.
                if(state["side"]["pokemon"][randomMon]["condition"] != "0 fnt") and (state["side"]["pokemon"][randomMon]["active"] == False):
                    invalidMon = False
                    #print(tag+'|/choose switch '+str(randomMon+1)+"|"+str(state["rqid"]))
                    return tag+'|/choose switch '+str(randomMon+1)+"|"+str(state["rqid"])