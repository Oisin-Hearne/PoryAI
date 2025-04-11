import showdown
import agent
import asyncio
import torch
import json
import numpy as np
from datetime import datetime
from IPython.display import clear_output
from matplotlib import pyplot as plt

class Trainer:

    def __init__(self, agents, showdowns, stateSize, actionSize, battles=10000, loadModel=None):
        self.agents = agents
        self.showdowns = showdowns
        self.battles = battles
        self.loadModel = loadModel
        self.stateSize = stateSize
        self.actionSize = actionSize
        
        if len(agents) < 1:
            self.mode = "random"
        elif len(agents) < 2:
            self.mode = "single"
        else:
            self.mode = "self-play"
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.loadModel:
            for agent in agents:
                agent.loadModel(f"data/models/model_{self.loadModel}.pt")
                agent.loadMemory(f"data/memory/memory_{self.loadModel}.json")
            
        with open('data/rewards.json') as f:
            self.rewardScheme = json.load(f)
        
    def analyseBattle(self, showdown, battleLength, winner):
        # Reward is limited by battle length and switchy behaviour.
        actionRatio = showdown.inter.actionRatio
        repeatMoves = showdown.inter.repeatMoves
        
        battleReward = -self.rewardScheme["win"] if not winner else self.rewardScheme["win"]
        if winner and battleLength < self.rewardScheme["lengthThreshold"]:
            battleReward += self.rewardScheme["shortBattle"]
        if actionRatio < self.rewardScheme["actionThreshold"]:
            battleReward -= self.rewardScheme["switchyBattle"]
        if repeatMoves > self.rewardScheme["repeatThreshold"]:
            battleReward -= self.rewardScheme["repeatBattle"]
            
        return battleReward
        

    async def agent_battle(self, agent, showdown):
        await showdown.restart()
        done = False
        totalReward = 0
        battleProgress = []
        
        
        while not done:
            state = showdown.getState()
            validActions = showdown.getValidActions()
            
            action = agent.act(state, validActions)
            nextState, reward, done, winner = await showdown.executeAction(action)
            battleProgress.append((state, action, reward, nextState, done))
        
        battleReward = self.analyseBattle(self.showdowns[0], len(battleProgress), winner)
        for i, (state, action, reward, nextState, done) in enumerate(battleProgress):
            adjustedReward = reward + (battleReward / (len(battleProgress)))
            totalReward += adjustedReward
            agent.remember(state, action, adjustedReward, nextState, done)
        
        print("Finishing battle...")
        return winner, totalReward
    
    async def random_battle(self, showdown):
        await showdown.restart()
        done = False
        
        while not done:
            validActions = showdown.getValidActions()
            
            action = np.random.choice(validActions)
            nextState, reward, done, winner = await showdown.executeAction(action)
        
        return winner

    def makePlot(self, x, y, battle, timestamp, winrate, ratio):
            plt.plot(x, y)
            plt.xlabel('Battles')
            plt.ylabel('Rewards')
            plt.title(f'Learning Curve - Winrate: {winrate} - Best Ratio: {ratio}')
            plt.savefig(f"data/logs/plots/plot-{battle}-{timestamp}.png")
            plt.show()
            
    async def trainingLoopSelf(self):
        agent1Wins = 0
        latestWins = 0
        currentBestRatio = 0
        rewards1 = 0
        plotX = []
        plotY = []
        currentBestModel = 0

        for battle in range(self.battles):
            
            # Concurrently execute both agents and get the results from agent_battle
            results = await asyncio.gather(self.agent_battle(self.agents[0], self.showdowns[0]), self.agent_battle(self.agents[1], self.showdowns[1]))
            winner = results[0][0]
            print(results)
            if winner == 1:
                agent1Wins += 1
                latestWins += 1
                rewards1 += results[0][1]
                plotY.append(results[0][1])
            else:
                rewards1 += results[0][1]
                plotY.append(results[0][1])
                
            self.agents[0].replay()
            
            plotX.append(battle)
            
            # Every 10 battles, output the current state and clear the old output.
            # Notebooks are so laggy.
            if battle % 10 == 0 and battle > 0:
                clear_output(wait=True)
                
                timestamp = datetime.now().strftime("%Y_%m%d-%p%I_%M_%S")
                # Save output to file
                with open(f"data/logs/outputs/output-{battle}-{timestamp}.txt", "w") as file:
                    file.write(f"Current Stats: \n Wins This Cycle: {agent1Wins} \n Battles: {battle} \n Epsilon: {self.agents[0].epsilon}")
                
                print(f"Cleared Output! Current Stats: \n Wins This Cycle: {agent1Wins} \n  Battles: {battle} \n Epsilon: {self.agents[0].epsilon}, \n Latest Wins: {latestWins} \n Stats: {self.showdowns[0].inter.getStats()}")
            
            # Every 50 battles, save the model and memory.
            if battle % 50 == 0 and battle > 0:
                

                
                # Save model and memory
                self.agents[0].saveModel(f"data/models/model_{battle}.pt")
                self.agents[0].saveMemory(f"data/memory/memory_{battle}.json")
                
                # Save plot
                self.makePlot(plotX, plotY, battle, timestamp, winRatio, currentBestRatio)
                


                f = open(f"data/stats/{battle}.json", "w")
                f.write(json.dumps({"wins": agent1Wins, "rewards": rewards1, "winsThisCycle": latestWins, "epsilon": self.agents[0].epsilon, "stats": self.showdowns[0].inter.getStats()}))
                f.close()
                latestWins = 0
                
            if battle % 100 == 0 and battle > 0:
                self.agents[0].loadTargetModel()
                # Set agent 2's weights to agent 1's.
                self.agents[1].model.load_state_dict(self.agents[0].model.state_dict())
                
            if battle % 500 == 0 and battle > 0:
                # If the previous 500 battles went worse than the current 500, revert to the previous model.
                self.showdowns[0].inter.resetStats()

                # Reset epsilon according to win ratio
                winRatio = agent1Wins / battle
                
                # If win ratio is too uneven, something's gone wrong. Reset epsilon.
                if float(winRatio) < 0.7 and float(winRatio) > 0.3:
                    
                    # Reset Epsilon
                    self.agents[0].epsilon = max(self.agents[0].epsilon, 0.5)

                if winRatio > currentBestRatio:
                    print("Noting best model")
                    currentBestModel = f"data/models/model_{battle}.pt"
                    currentBestRatio = winRatio
                else:
                    print("Reloading Model...!")
                    self.agents[0].loadModel(currentBestModel)
                    self.agents[0].epsilon = 0.3
                    
                    
                if self.showdowns[0].inter.getStats()["repeatMoves"] > 200 or self.showdowns[0].inter.getStats()["switched"] > 200:
                    print("Resetting Epsilon")
                    self.agents[0].epsilon = 0.7
            
    async def trainingLoopExpert(self):
        agent1Wins = 0
        latestWins = 0
        currentBestRatio = 0
        rewards = 0
        plotX = []
        plotY = []
        currentBestModel = 0
        
        for battle in range(self.battles):
            
            # Concurrently execute both agents and get the results from agent_battle
            winner, reward = await self.agent_battle(self.agents[0], self.showdowns[0])
            if winner == 1:
                agent1Wins += 1
                latestWins += 1
                rewards += reward
                plotY.append(reward)
            else:
                rewards += reward
                plotY.append(reward)
                
            self.agents[0].replay()
            
            plotX.append(battle)
            
            # Every 10 battles, output the current state and clear the old output.
            # Notebooks are so laggy.
            if battle % 10 == 0 and battle > 0:
                clear_output(wait=True)
                
                timestamp = datetime.now().strftime("%Y_%m%d-%p%I_%M_%S")
                # Save output to file
                with open(f"data/logs/outputs/output-{battle}-{timestamp}.txt", "w") as file:
                    file.write(f"Current Stats: \n Wins This Cycle: {agent1Wins} \n Battles: {battle} \n Epsilon: {self.agents[0].epsilon}")
                
                print(f"Cleared Output! Current Stats: \n Wins This Cycle: {agent1Wins} \n  Battles: {battle} \n Epsilon: {self.agents[0].epsilon}, \n Latest Wins: {latestWins} \n Stats: {self.showdowns[0].inter.getStats()}")
            
            # Every 50 battles, save the model and memory.
            if battle % 50 == 0 and battle > 0:
            
                
                # Save model and memory
                self.agents[0].saveModel(f"data/models/model_{battle}.pt")
                self.agents[0].saveMemory(f"data/memory/memory_{battle}.json")
                
                # Save plot
                print(plotY)
                self.makePlot(plotX, plotY, battle, timestamp, winRatio, currentBestRatio)
                


                f = open(f"data/stats/{battle}.json", "w")
                f.write(json.dumps({"wins": agent1Wins, "rewards": rewards, "winsThisCycle": latestWins, "epsilon": self.agents[0].epsilon, "stats": self.showdowns[0].inter.getStats()}))
                f.close()
                latestWins = 0
                
            if battle % 100 == 0 and battle > 0:
                self.agents[0].loadTargetModel()
                
            if battle % 500 == 0 and battle > 0:
                # If the previous 500 battles went worse than the current 500, revert to the previous model.
                self.showdowns[0].inter.resetStats()
                
                # Reset epsilon according to win ratio
                winRatio = agent1Wins / battle
                
                if float(winRatio) < 0.1:
                    
                    # Reset Epsilon
                    self.agents[0].epsilon = min(0.7, self.agents[0].epsilon +0.2)
                    
                if winRatio > currentBestRatio:
                    print("Noting best model")
                    currentBestModel = f"data/models/model_{battle}.pt"
                    currentBestRatio = winRatio
                else:
                    print("Reloading Model...!")
                    self.agents[0].loadModel(currentBestModel)
                    self.agents[0].epsilon = 0.3 
                    
                winRatio = 0.0       
                    
    async def trainingLoopRandom(self):
        agentWins = 0
        
        for battle in range(self.battles):
            winner = await self.random_battle(self.showdowns[0])
            if winner == 1:
                agentWins += 1
                
            if battle % 10 == 0 and battle > 0:
                clear_output(wait=True)
                print(f"Current Wins: {agentWins} \n Battles: {battle}")            

    
    async def run(self):
        if self.mode == "self-play":
            await self.trainingLoopSelf()
        elif self.mode == "single":
            await self.trainingLoopExpert()
        elif self.mode == "random":
            await self.trainingLoopRandom()