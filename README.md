# PoryAI
PoryAI is an attempt at a reinforcement-learning model trained to play Pokémon Showdown.

## Overview


## Running
- Install all the requirements in requirements.txt
- Modify env as necessary. By default, the program will start training in Self-Play mode looking for a locally ran copy of [Showdown](https://github.com/smogon/pokemon-showdown/blob/master/server/README.md). If one is not present, it will fail.
    - The WEBSOCKET field can be modified with the official server's websocket, `link here`. For the sake of the PS developers, do not run the self-play mode on the official server.
    - The FORMAT field can be set to any valid format. It is gen9randombattle by default, as that was what was used for this project.
    - The MODE field can be updated between SELFPLAY, EXPERT and QUEUE.
        - In self-play mode, the agent battles against itself to learn the best ways to battle.
        - In expert mode, the agent continously challenges a given expert, which should accept challenges. Experts are used to create better strategies in the agent. An expert is not included, but pmariglia's [Foul Play](https://github.com/pmariglia/foul-play) was used previously.
        - In queue mode, the agent enters the ladder for the given format. This is used to measure how the agent performs against real players on the official PS server.
    - The USER1, USER2, PASS1, PASS2 fields are to set the details for the two agents. USER2 and PASS2 can be left blank if the agent is running in Expert or Queue mode.
    - The OFFLINE field determines whether to login without security measures. This is for locally hosted servers that are purely for testing. This will fail if attempting to log in to the official server. FALSE by default.
    - The ITERATIONS field determines how many battles the agent should perform before stopping.

## Resources
In the `resources` folder of this repository, the related **minor dissertation** for this project can be found as well as the best-trained model and memory of PoryAI as of xxxxxx. The memory and model can be used in the Running section in order to run a pre-trained agent.

