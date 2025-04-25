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
            if self.showdowns[0].ladder:
                self.mode = "human"
            else:
                self.mode = "single"
        else:
            self.mode = "self-play"
            
        print(self.mode)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.loadModel:
            for agent in agents:
                agent.loadModel(f"data/models/model_{self.loadModel}.pt")
                agent.loadMemory(f"data/memory/memory_{self.loadModel}.json")
                agent.epsilon = 0.2
            
        with open('data/rewards.json') as f:
            self.rewardScheme = json.load(f)
        
    def analyseBattle(self, showdown, battleLength, winner):
        # Reward is limited by battle length and switchy behaviour.
        actionRatio = showdown.inter.actionRatio
        repeatMoves = showdown.inter.repeatMoves
        
        battleReward = self.rewardScheme["win"] if (winner == 1) else 0
        print(f"Winner: {winner}, Battle Length: {battleLength}, Action Ratio: {actionRatio}, Repeat Moves: {repeatMoves}")
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
            if(battleReward > 3):
                for j in range(4): # Remember winning battles more.
                    print("Won! Remembering more...")
                    agent.remember(state, action, adjustedReward, nextState, done)
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

    def makePlot(self, x, y, battle, timestamp, winrate, ratio, col):
            plt.scatter(x, y, s=2, c=col, alpha=1, label="Win or Loss")
            plt.plot(x, y, c='blue', alpha=0.5, label="Rewards")
            plt.xlabel('Battles')
            plt.ylabel('Rewards')
            plt.title(f'Learning Curve - Winrate: {winrate} - Best Ratio: {ratio}')
            plt.savefig(f"data/logs/plots/plot-{battle}-{timestamp}.png")
            plt.show()
            
    def rewardsPlot(self, x, y, battle, timestamp):
        plt.scatter(x, y, s=2, c="blue", alpha=1, label="Average Rewards")
        plt.plot(x, y)
        plt.xlabel('Battles')
        plt.ylabel('Rewards')
        plt.title(f'Average Rewards over Time')
        plt.savefig(f"data/logs/plots/plot-{battle}-{timestamp}-avg.png")
        plt.show()
            
    async def trainingLoopSelf(self):
        agent1Wins = 0
        latestWins = 0
        currentBestRatio = 0
        rewards1 = 0
        plotX = []
        plotY = []
        plotAvg = []
        plotCol = []
        currentBestModel = 0
        currentBestRewards = -50

        for battle in range(self.battles):
            
            # Concurrently execute both agents and get the results from agent_battle
            results = await asyncio.gather(self.agent_battle(self.agents[0], self.showdowns[0]), self.agent_battle(self.agents[1], self.showdowns[1]))
            winner = results[0][0]
            print(results)
            if winner == 1:
                agent1Wins += 1
                latestWins += 1
                rewards1 += results[0][1]
                plotCol.append("green")
                plotY.append(results[0][1])
            else:
                rewards1 += results[0][1]
                plotCol.append("red")
                plotY.append(results[0][1])
                
            self.agents[0].replay()
            
            plotX.append(battle)
            plotAvg.append(sum(plotY)/len(plotY))
            # Every 10 battles, output the current state and clear the old output.
            # Notebooks are so laggy.
            if battle % 10 == 0 and battle > 0:
                clear_output(wait=True)
                
                timestamp = datetime.now().strftime("%Y_%m%d-%p%I_%M_%S")
                # Save output to file
                with open(f"data/logs/outputs/output-{battle}-{timestamp}.txt", "w") as file:
                    file.write(f"Current Stats: \n Wins This Cycle: {agent1Wins} \n Battles: {battle} \n Epsilon: {self.agents[0].epsilon}, \n Average Reward: {sum(plotY)/len(plotY)}")
                
                print(f"Cleared Output! Current Stats: \n Wins This Cycle: {agent1Wins} \n  Battles: {battle} \n Epsilon: {self.agents[0].epsilon}, \n Latest Wins: {latestWins} \n Stats: {self.showdowns[0].inter.getStats()}\n Average Reward: {sum(plotY)/len(plotY)}")
            
            # Every 50 battles, save the model and memory.
            if battle % 50 == 0 and battle > 0:
                winRatio = agent1Wins / battle


                
                # Save model and memory
                self.agents[0].saveModel(f"data/models/model_{battle}.pt")
                self.agents[0].saveMemory(f"data/memory/memory_{battle}.json")
                
                # Save plot
                self.makePlot(plotX, plotY, battle, timestamp, winRatio, currentBestRatio, plotCol)
                self.rewardsPlot(plotX, plotAvg, battle, timestamp)
                


                f = open(f"data/stats/{battle}.json", "w")
                f.write(json.dumps({"wins": agent1Wins, "rewards": rewards1, "winsThisCycle": latestWins, "epsilon": self.agents[0].epsilon, "stats": self.showdowns[0].inter.getStats()}))
                f.close()
                latestWins = 0
                
            if battle % 100 == 0 and battle > 0:
                self.agents[0].loadTargetModel()
                # Set agent 2's weights to agent 1's.
                self.agents[1].model.load_state_dict(self.agents[0].model.state_dict())
                if self.showdowns[0].inter.getStats()["repeatMoves"] > 2000 or self.showdowns[0].inter.getStats()["switched"] > 2000:
                    print("Resetting Epsilon")
                    self.agents[0].epsilon = 0.2
                self.showdowns[0].inter.resetStats()
            if battle % 500 == 0 and battle > 0:
                # If the previous 500 battles went worse than the current 500, revert to the previous model.
                
                self.agents[0].epsilon += 0.2
                
                # Get cumulative rewards for last 500 battles from plotY.
                currentAverage = sum(plotY[-500:]) / 500
                if currentAverage > currentBestRewards:
                    print("Noting best model")
                    currentBestModel = f"data/models/model_{battle}.pt"
                    currentBestRewards = currentAverage
                else:
                    print("Reloading Model...!")
                    self.agents[0].loadModel(currentBestModel)
                    self.agents[0].epsilon = 0.3
                    
                    

            
    async def trainingLoopExpert(self):
        agent1Wins = 0
        latestWins = 0
        currentBestRatio = 0
        currentWins = 0
        rewards = 0
        plotX = []
        plotY = []
        plotAvg = []
        plotCol = []
        currentBestModel = 0
        currentBestRewards = -50
        
        for battle in range(self.battles):
            
            # Concurrently execute both agents and get the results from agent_battle
            winner, reward = await self.agent_battle(self.agents[0], self.showdowns[0])
            if winner == 1:
                agent1Wins += 1
                latestWins += 1
                currentWins += 1
                rewards += reward
                plotCol.append("green")
                plotY.append(reward)
            else:
                rewards += reward
                plotCol.append("red")
                plotY.append(reward)
                
            plotAvg.append(sum(plotY)/len(plotY))
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
                
                print(f"Cleared Output! Current Stats: \n Wins This Cycle: {agent1Wins} \n  Battles: {battle} \n Epsilon: {self.agents[0].epsilon}, \n Latest Wins: {latestWins} \n Stats: {self.showdowns[0].inter.getStats()}, \n: Average Reward: {sum(plotY)/len(plotY)}")
            
            # Every 50 battles, save the model and memory.
            if battle % 50 == 0 and battle > 0:
                winRatio = agent1Wins / battle
                
                # Save model and memory
                self.agents[0].saveModel(f"data/models/model_{battle}.pt")
                self.agents[0].saveMemory(f"data/memory/memory_{battle}.json")
                
                # Save plot
                print(plotY)
                self.makePlot(plotX, plotY, battle, timestamp, winRatio, currentBestRatio, plotCol)
                self.rewardsPlot(plotX, plotAvg, battle, timestamp)
                


                f = open(f"data/stats/{battle}.json", "w")
                f.write(json.dumps({"wins": agent1Wins, "rewards": rewards, "winsThisCycle": latestWins, "epsilon": self.agents[0].epsilon, "stats": self.showdowns[0].inter.getStats()}))
                f.close()
                latestWins = 0
                
            if battle % 100 == 0 and battle > 0:
                self.agents[0].loadTargetModel()
                
                if self.showdowns[0].inter.getStats()["repeatMoves"] > 2000 or self.showdowns[0].inter.getStats()["switched"] > 2000:
                    print("Resetting Epsilon")
                    self.agents[0].epsilon = 0.2
                self.showdowns[0].inter.resetStats()
                
            if battle % 500 == 0 and battle > 0:
                # If the previous 500 battles went worse than the current 500, revert to the previous model.
                self.agents[0].epsilon += 0.2
                
                # Get cumulative rewards for last 500 battles from plotY.
                currentAverage = sum(plotY[-500:]) / 500
                if currentAverage > currentBestRewards:
                    print(f"Noting best model {battle}...")
                    print(f"Current Average: {currentAverage}, Current Best: {currentBestRewards}")
                    currentBestModel = f"data/models/model_{battle}.pt"
                    currentBestRewards = currentAverage
                else:
                    print(f"Reloading Model {currentBestModel}...!")
                    print(f"Current Average: {currentAverage}, Current Best: {currentBestRewards}")
                    self.agents[0].loadModel(currentBestModel)
                    self.agents[0].epsilon = 0.3
                    
    async def trainingLoopRandom(self):
        agentWins = 0
        
        for battle in range(self.battles):
            winner = await self.random_battle(self.showdowns[0])
            if winner == 1:
                agentWins += 1
                
            if battle % 10 == 0 and battle > 0:
                clear_output(wait=True)
                print(f"Current Wins: {agentWins} \n Battles: {battle}")
                
                timestamp = datetime.now().strftime("%Y_%m%d-%p%I_%M_%S")

                with open(f"data/logs/outputs/output-RANDOM-{battle}-{timestamp}.txt", "w") as file:
                    file.write(f"Current Stats: \n Wins This Cycle: {agentWins} \n Battles: {battle}, \n Stats: {self.showdowns[0].inter.getStats()}")
              
    
    
    # Training is not necessarily the goal when playing against humans, so a smaller training loop is used.
    async def trainingLoopHuman(self):
        print("Starting loop!")
        agentWins = 0
        latestWins = 0
        currentBestRatio = 0
        rewards = 0
        plotX = []
        plotY = []
        plotAvg = []
        plotCol = []
        
        for battle in range(self.battles):
            winner, reward = await self.agent_battle(self.agents[0], self.showdowns[0])
            if winner == 1:
                agentWins += 1
                latestWins += 1
                rewards += reward
                plotCol.append("green")
                plotY.append(reward)
            else:
                rewards += reward
                plotCol.append("red")
                plotY.append(reward)
            
            plotAvg.append(sum(plotY)/len(plotY))
            plotX.append(battle)
            
            if battle % 10 == 0 and battle > 0:
                clear_output(wait=True)
                
                timestamp = datetime.now().strftime("%Y_%m%d-%p%I_%M_%S")
                # Save output to file
                with open(f"data/logs/outputs/output-VSHUMAN-{battle}-{timestamp}.txt", "w") as file:
                    file.write(f"Current Stats: \n Wins This Cycle: {agentWins} \n Battles: {battle} \n Epsilon: {self.agents[0].epsilon}")
                
                print(f"Cleared Output! Current Stats: \n Wins This Cycle: {agentWins} \n  Battles: {battle} \n Epsilon: {self.agents[0].epsilon}, \n Latest Wins: {latestWins} \n Stats: {self.showdowns[0].inter.getStats()}, \n: Average Reward: {sum(plotY)/len(plotY)}")
            
            if battle % 50 == 0 and battle > 0:
                winRatio = agentWins / battle
                
                # Save model and memory
                self.agents[0].saveModel(f"data/models/model_vsHUMAN_{battle}.pt")
                self.agents[0].saveMemory(f"data/memory/memory_vsHUMAN_{battle}.json")
                
                # Save plot
                print(plotY)
                self.makePlot(plotX, plotY, battle, timestamp, winRatio, currentBestRatio, plotCol)
                self.rewardsPlot(plotX, plotAvg, battle, timestamp)
                


                f = open(f"data/stats/{battle}.json", "w")
                f.write(json.dumps({"wins": agentWins, "rewards": rewards, "winsThisCycle": latestWins, "epsilon": self.agents[0].epsilon, "stats": self.showdowns[0].inter.getStats()}))
                f.close()
                latestWins = 0
                 

    
    async def run(self):
        if self.mode == "self-play":
            await self.trainingLoopSelf()
        elif self.mode == "single":
            await self.trainingLoopExpert()
        elif self.mode == "random":
            await self.trainingLoopRandom()
        elif self.mode == "human": 
            await self.trainingLoopHuman()