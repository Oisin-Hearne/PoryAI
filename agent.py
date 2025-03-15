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
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)


class Agent:
    
    def __init__(self, state_dim, action_dim, device, possible_actions, lr=0.001, gamma=0.99, epsilon=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = []
        self.batch_size = 64
        self.device = device
        
        # Receives a list of actions and creates a dictionary to map them to indices.
        self.possible_actions = possible_actions
        self.action_index = {action: i for i, action in enumerate(possible_actions)}
        self.index_action = {i: action for i, action in enumerate(possible_actions)}
        
        self.model = PokeNet(state_dim, action_dim).to(device)
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
        print(f"State: {state}")
        print(f"State length {len(flattened)}")

        return torch.FloatTensor(flattened).to(self.device) # tensor of the flattened state. This is something the agent can interpret.
    
    # Make an action from a list of valid ones and the state. 
    def act(self, state, valid_actions):
        state_tensor = self.get_state(state)
        
        # Select randomly from the list of valid actions.
        
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(valid_actions)
            return action
        
        valid_actions = [self.action_index[action] for action in valid_actions]
        

        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor)
        valid_q = [q_values[i] for i in valid_actions]
        print(f"Decision: {valid_actions[np.argmax(valid_q)]}")
        decision = valid_actions[np.argmax(valid_q)]
        return self.index_action[decision]

    # Append what's just occured and the result of that action to memory.
    def remember(self, state, action, reward, next_state, done):
        print("Remembering")
        index_of_action = self.action_index[action] # Action translated to index for storage
        self.memory.append((state, index_of_action, reward, next_state, done))
        
    # Replay part of memory to learn from it
    def replay(self):
        print("Replaying")
        # If there's not enough memory, don't bother.
        if len(self.memory) < self.batch_size:
            return
        
        # Take a random sample of the memory to learn from.
        batch = random.sample(self.memory, self.batch_size)
        
        self.model.train()
        
        states = torch.stack([self.get_state(state) for state, _, _, _, _ in batch])
        next_states = torch.stack([self.get_state(next_state) for _, _, _, next_state, _ in batch])
        actions = torch.tensor([index_of_action for _, index_of_action, _, _, _ in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([reward for _, _, reward, _, _ in batch], dtype=torch.float).to(self.device)
        dones = torch.tensor([done for _, _, _, _, done in batch], dtype=torch.float).to(self.device)
        
        current_q_vals = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_vals = self.model(next_states).max(1)[0]
            target_q_vals = rewards + (1- dones) * self.gamma * next_q_vals
            
        loss = self.criterion(current_q_vals, target_q_vals)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # This is the epsilon decay. It's a way to make the agent less random over time.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    
    def saveModel(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
    def loadModel(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']


    # #Select randomly for now        
    # def getAction(self, state, tag):

    #     if 'active' in state:
    #         invalidMove = True

    #         # Pokémon can be locked into moves, so we need to check that the chosen move hasn't been disabled.
    #         while invalidMove:
    #             randomMove = random.choice(state["active"][0]["moves"])
    #             if("disabled" not in randomMove.keys()):
    #                 invalidMove = False
    #                 return tag+'|/choose move '+randomMove["id"]+"|"+str(state["rqid"])
    #             elif (not randomMove["disabled"]):
    #                 invalidMove = False
    #                 return tag+'|/choose move '+randomMove["id"]+"|"+str(state["rqid"])
            
    #     if 'forceSwitch' in state:
    #         invalidMon = True
            
    #         while invalidMon:
    #             randomMon = random.randint(0,5)
    #             # Don't send in a fainted mon or try to send in the current one.
    #             if(state["side"]["pokemon"][randomMon]["condition"] != "0 fnt") and (state["side"]["pokemon"][randomMon]["active"] == False):
    #                 invalidMon = False
    #                 #print(tag+'|/choose switch '+str(randomMon+1)+"|"+str(state["rqid"]))
    #                 return tag+'|/choose switch '+str(randomMon+1)+"|"+str(state["rqid"])
                
