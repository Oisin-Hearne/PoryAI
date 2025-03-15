import showdown
import agent
import asyncio
import torch
import json

async def training_loop(agent, showdown, numBattles=1000):
    for battle in range(numBattles):
        
        
        await showdown.restart()
        battleDone = False
        totalReward = 0
        
        while not battleDone:
            state = showdown.getState()
            
            validActions = showdown.getValidActions()
            action = agent.act(state, validActions)
            nextState, reward, battleDone = await showdown.executeAction(action)
            print(f"Action: {action}, Reward: {reward}")
            
            agent.remember(state, action, reward, nextState, battleDone)
            agent.replay()
            totalReward += reward
            
            if battleDone:
                print(f"Battle {battle} done. Total reward: {totalReward}")
                
        
    if battle % 100 == 0:
        agent.saveModel(f"model_{battle}.pt")
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stateSize = 671

possibleActions = json.load(open("data/possible_actions.json", "r"))
actionSize = len(possibleActions)

newAgent = agent.Agent(stateSize, actionSize, device, possibleActions)

sd = showdown.Showdown("https://play.pokemonshowdown.com/action.php", "PoryAI-0", "password", "ws://localhost:8000/showdown/websocket", "gen9randombattle")

asyncio.run(sd.connectNoSecurity())
asyncio.run(training_loop(newAgent, sd, 1000))