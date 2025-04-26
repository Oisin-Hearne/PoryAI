# PoryAI
PoryAI is an attempt at training Reinforcement Learning agents to play competitive Pokémon, training against itself, an expert system, and eventually against human players on [Pokémon Showdown](https://play.pokemonshowdown.com/).

## Overview
This repository consists of several python classes for interacting with Pokémon Showdown, gathering state, assigning reward and training a DNN Reinforcement Learning agent.

It features several modes, allowing for different methods of training, and outputs a variety of information on the agent's performance over the course of the battle.

There is an included model, the latest version trained (a compilation of self-play, expert training and human training) which can be imported back into the program to examine how the training worked. 

This project is highly configurable. It can be run with a locally hosted server or the main Showdown server, and features a modifiable reward structure in order to change how the model is trained.

<a href="https://gyazo.com/06caed1311bc605185cfd044dde1b1c7"><img src="https://i.gyazo.com/06caed1311bc605185cfd044dde1b1c7.gif" alt="Unfortunate for PoryAI." width="700"/></a>


## Running
- Ensure Python is installed. Version 3.11.7 was used for this project.
- Install all the requirements in requirements.txt by running `pip install -r requirements.txt`.
- Modify config.json as necessary. See the [wiki](https://github.com/Oisin-Hearne/PoryAI/wiki) for the options available.
- Ensure that related programs are running. For locally hosted instances, this means that your Pokémon Showdown server needs to be running. Any experts being used should be active too.
    - The default config expects a local copy of Pokémon Showdown running. To avoid this, an alternate config is included in the resources folder of this repository. Using this, just modify the username and password to a unique one.
    - This alternate config starts in vs Human mode, instead of self-play.
- Run the command `python runner.py` in the root folder of the project.

Alternatively, `modularTesting.ipynb` in the root folder can be used for quickly changing modes and what agents are used. Note that notebooks outputting this much text can be somewhat laggy, so verbose mode being turned off may help. 

## Output
Various statistics and plots are output in the /data/ folder. These can be used to monitor how the program is performing. This includes
- **data/logs/outputs/** -  Simple statistics outputted every 10 battles, showcasing the number of wins/losses, supereffective moves used, and so on.
- **data/logs/battles/** - Full recounts of the events of every battle. This was implemented as replays cannot be captured on local servers.
- **data/logs/plots/** - Images output by all modes but random. Showcases how the agent is learning, with a reward plot and an average reward plot, with the latter showing a clearer learning curve.
- **data/stats** - The stats taken every 50 battles of the most recently ran model. Includes super effective hits, as well as negative behaviours like move and switch repetition.
- **data/memory** & **data/models** - Models and memory are stored every 50 iterations for backup and restoration purposes.

## Resources
In the **resources** folder of this repository, the related **minor dissertation** for this project can be found as well as a sample config for vs Human training.

The config requires some modification for a username and password in order to correctly log in.

Additionally, [here](https://drive.google.com/file/d/1GQT1EglpOhPajA422-FyoEinaFqHxi5G/view?usp=sharing) is a link to the latest and best trained model. This would've been included in the repository, but it is too large to store on Github effectively. 

The memory and model can be used in the Running section in order to run a pre-trained agent. Place the trained model and memory in **data/models** and **data/memory** respectively.
