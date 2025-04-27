import showdown
import agent
import asyncio
import json
import os
import training
import torch
        

# This file allows users to run the bot in different modes based on the config.
async def run():
    agents = config["agents"]
    mode = config["mode"]

    
    #Check if the combination of agents and mode is valid
    if len(agents) > 2:
        print("Error: Only two agents are allowed.")
        return
    elif len(agents) < 1 and mode != "RANDOM":
        print("Error: For non-random modes, at least one agent is required.")
        return
    elif len(agents) > 1 and mode == "SELFPLAY":
        print("Error: Selfplay mode only allows one agent.")
        return
    
    # Execute the correct mode.
    if mode == "RANDOM":
        await beginRandomMode(config)
    elif mode == "SELFPLAY":
        await beginSelfPlayMode(config)
    elif mode == "EXPERT":
        await beginExpertMode(config)
    elif mode == "QUEUE":
        await beginHumanMode(config)
    else:
        print("Error: Invalid mode. Please choose RANDOM, SELFPLAY, EXPERT, or QUEUE.")
        return

# In random mode, the bot will select a random action every turn against a designated opponent.
async def beginRandomMode(config):
    sd = showdown.Showdown(config["uri"], config["agents"][0]["username"], config["agents"][0]["password"], config["websocket"], config["format"], config["agents"][0]["challenger"], config["agents"][0]["verbose"], config["agents"][0]["opponent"])
    
    if config["offline"]:
        await sd.connectNoSecurity()
    else:
        await sd.connectToShowdown()
        
    print("Running Random Mode")
    trainer = training.Trainer([], [sd], stateSize, actionSize, config["iterations"])
    await trainer.run()
    
# In expert mode, the bot will play & learn against a designated opponent.
async def beginExpertMode(config):
    sd = showdown.Showdown(config["uri"], config["agents"][0]["username"], config["agents"][0]["password"], config["websocket"], config["format"], config["agents"][0]["challenger"], config["agents"][0]["verbose"], config["agents"][0]["opponent"])

    if config["offline"]:
        await sd.connectNoSecurity()
    else:
        await sd.connectToShowdown()
    
    agent = agent.Agent(stateSize, actionSize, device, possibleActions)
    
    model = None if not modelPresent else config['lastModel']
    print(f"Running Expert Mode with model {model}")
    trainer = training.Trainer([agent], [sd], stateSize, actionSize, config["iterations"], model)
    await trainer.run()

# In Selfplay mode, the bot will play against itself and learn from the experience.
async def beginSelfPlayMode(config):
    sd1 = showdown.Showdown(config["uri"], config["agents"][0]["username"], config["agents"][0]["password"], config["websocket"], config["format"], config["agents"][0]["challenger"], config["agents"][0]["verbose"], config["agents"][0]["opponent"])
    sd2 = showdown.Showdown(config["uri"], config["agents"][1]["username"], config["agents"][1]["password"], config["websocket"], config["format"], config["agents"][1]["challenger"], config["agents"][1]["verbose"], config["agents"][1]["opponent"])
    
    if config["offline"]:
        await sd1.connectNoSecurity()
        await sd2.connectNoSecurity()
    else:
        await sd1.connectToShowdown()
        await sd2.connectToShowdown()
    
    agent1 = agent.Agent(stateSize, actionSize, device, possibleActions)
    agent2 = agent.Agent(stateSize, actionSize, device, possibleActions)
    
    model = None if not modelPresent else config['lastModel']
    print(f"Running Self-Play Mode with model {model}")
    trainer = training.Trainer([agent1, agent2], [sd1, sd2], stateSize, actionSize, config["iterations"], model)
    await trainer.run()
    
# In human mode, the bot will enter the designated queue and play against human opponents.
async def beginHumanMode(config):
    sd = showdown.Showdown(config["uri"], config["agents"][0]["username"], config["agents"][0]["password"], config["websocket"], config["format"], config["agents"][0]["challenger"], config["agents"][0]["verbose"], config["agents"][0]["opponent"], True)
    
    if config["offline"]:
        await sd.connectNoSecurity()
    else:
        await sd.connectToShowdown()
    
    agent = agent.Agent(stateSize, actionSize, device, possibleActions)
    
    model = None if not modelPresent else config['lastModel']
    print(f"Running Human Mode with model {model}")
    trainer = training.Trainer([agent], [sd], stateSize, actionSize, config["iterations"], model)
    await trainer.run()


if __name__ == "__main__":
    fopen = open("config.json", "r")
    config = json.load(fopen)
    fopen.close()
    "data/models/model_{battle}.pt"
    stateSize = config["stateSize"]
    possibleActions = json.load(open("data/possible_actions.json", "r"))
    actionSize = len(possibleActions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelPresent = os.path.isfile(f"data/models/model_{config['lastModel']}.pt") and os.path.isfile(f"data/memory/memory_{config['lastModel']}.json")
    print(f"Model present: {modelPresent}")
    
    try:
        asyncio.run(run())
        print("Finished running...")
    except Exception as e:
        print(f"An error occured: {e}")