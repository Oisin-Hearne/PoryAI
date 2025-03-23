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
        self.lstm = nn.LSTM(state_dim, 128, batch_first=True)
        self.network = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x.unsqueeze(1), hidden)
        return self.network(lstm_out.squeeze(1)), hidden


class Agent:
    
    def __init__(self, state_dim, action_dim, device, possible_actions, lr=0.001, gamma=0.99, epsilon=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.memory = []
        self.battles = []
        self.currentBattle = []
        self.batch_size = 128
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
            
            for key in d.keys():
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

        return torch.FloatTensor(flattened).unsqueeze(0).to(self.device) # tensor of the flattened state. This is something the agent can interpret.
    
    def getStatesBatch(self, states):
        flattenedBatch = []
        
        for state in states:
            flattened = []
            
            def _flatten(d, prefix=""):
            
                
                for key in d.keys():
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
                        
                return flattened
            
            _flatten(state)
            flattenedBatch.append(flattened)

        return torch.FloatTensor(flattenedBatch).to(self.device)
    
    # Make an action from a list of valid ones and the state. 
    def act(self, state, valid_actions):
        state_tensor = self.get_state(state)
        
        # Count how many moves & switches
        moveCount = sum(1 for a in valid_actions if a.startswith("/choose move"))
        switchCount = sum(1 for a in valid_actions if a.startswith("/choose switch"))
        move_actions = [a for a in valid_actions if a.startswith("/choose move")]
        switch_actions = [a for a in valid_actions if a.startswith("/choose switch")]
        
        # Select randomly from the list of valid actions.
        
        if np.random.rand() <= self.epsilon:
            # Forcing it to pick lower options sometimes.
            if hasattr(self, 'action_counts') and moveCount > 0 and switchCount > 0:
                moveTotal = sum(self.action_counts.get(a, 0) for a in valid_actions if a.startswith("/choose move"))
                switchTotal = sum(self.action_counts.get(a, 0) for a in valid_actions if a.startswith("/choose switch"))
                
                if moveTotal < switchTotal * 0.7 and moveCount > 0:
                    
                    print(f"Forcing move: {move_actions}")
                    return np.random.choice(move_actions)
                elif any(self.action_counts.get(m, 0) > 3 * self.action_counts.get(move_actions[0], 1) for m in move_actions[1:]) and len(move_actions) > 1:
                    overused = max(move_actions, key=lambda m: self.action_counts.get(m, 0))
                    return np.random.choice([m for m in move_actions if m != overused])
            return np.random.choice(valid_actions)
        
        valid_indices = [self.action_index[action] for action in valid_actions]
        

        self.model.eval()
        with torch.no_grad():
            q_values, _ = self.model(state_tensor)
        valid_q = [q_values[0, i].item() - (i * 0.1) for i in valid_indices]
        
        # move bias
        if moveCount > 0 and switchCount > 0:
            for i, action in enumerate(valid_actions):
                if action.startswith("/choose move"):
                    valid_q[i] += 0.3
                    
                    if "move 1" in action:
                        valid_q[i] += 0.05
                    elif "move 2" in action:
                        valid_q[i] += 0.03
                    elif "move 3" in action:
                        valid_q[i] += 0.01
                
        
        if not hasattr(self, "action_counts"):
            self.action_counts = {}
        
        bestActionIndex = np.argmax(valid_q)
        decision = valid_indices[bestActionIndex]
        chosenAction = self.index_action[decision]
        self.action_counts[chosenAction] = self.action_counts.get(chosenAction, 0) + 1
        print(chosenAction)
        return chosenAction

    # Append what's just occured and the result of that action to memory.
    def remember(self, state, action, reward, next_state, done):
        print("Remembering")
        index_of_action = self.action_index[action] # Action translated to index for storage
        battleResults = (state, index_of_action, reward, next_state, done)
        self.memory.append(battleResults)
        self.currentBattle.append(battleResults)
        
        if done:
            self.battles.append(self.currentBattle)
            self.currentBattle = []
            
            if len(self.battles) > 10000:
                self.battles.pop(0)
        
    # Replay part of memory to learn from it
    def replay(self):
        print("Replaying")
        # If there's not enough memory, don't bother.
        if len(self.memory) < self.batch_size:
            return
        
        # Take a random sample of the memory to learn from.
        batch = random.sample(self.memory, self.batch_size // 2)
        battleBatch = []
        if len(self.battles) > 0:
            sampleBattles = random.sample(self.battles, min(10, len(self.battles)))
            for battle in sampleBattles:
                if len(battle) <= 12:
                    battleBatch.extend(battle)
                else:
                    start = random.randint(0, len(battle) - 12)
                    battleBatch.extend(battle[start:start+12])
                    
        batch = batch + battleBatch
        if (len(batch) > self.batch_size):
            batch = random.sample(batch, self.batch_size)
        
        statesBatch = [state for state, _, _, _, _ in batch]
        nextStatesBatch = [nextState for _, _, _, nextState, _ in batch]
        
        self.model.train()
        
        states = self.getStatesBatch(statesBatch)
        next_states = self.getStatesBatch(nextStatesBatch)
        actions = torch.tensor([index_of_action for _, index_of_action, _, _, _ in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([reward for _, _, reward, _, _ in batch], dtype=torch.float).to(self.device)
        dones = torch.tensor([done for _, _, _, _, done in batch], dtype=torch.float).to(self.device)
        
        q_vals, _ = self.model(states)
        current_q_vals = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q, _ = self.model(next_states)
            next_q_vals = next_q.max(1)[0]
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
            'epsilon': self.epsilon,
            'battles': self.battles[:10000]
        }, path)
        
    def saveMemory(self, path):
        torch.save(self.memory, path)
    
    def loadMemory(self, path):
        self.memory = torch.load(path)
        
    def loadModel(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        if 'battles' in checkpoint:
            self.battles = checkpoint['battles']


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
                
