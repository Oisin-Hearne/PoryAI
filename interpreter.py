import json
import copy
from difflib import get_close_matches

class Interpreter:

    def __init__(self):
        self.loadData()
        self.knownOpponents = []

    # Load in previously fetched data for use in state.
    def loadData(self):
        with open('data/sample-state.json') as f:
            self.state = json.load(f)
        with open('data/abilitydata.json') as f:
            self.abilityData = json.load(f)
        with open('data/movedata.json') as f:
            self.moveData = json.load(f)
        with open('data/pokedata.json') as f:
            self.pokeData = json.load(f)
        with open('data/dictionary.json') as f:
            self.miscData = json.load(f)
        with open('data/itemdata.json') as f:
            self.itemData = json.load(f)


    # Update state w/ Active Request
    def updateStateActive(self, newState):
        # Extract information from newState
        porySide = newState["side"]["pokemon"]
        activeMoves = newState["active"][0]["moves"]
        self.state["request"] = self.miscData["request"][list(newState.keys())[0]] #Set request state
        
        for poke in range(len(porySide)):
            pokeName = porySide[poke]["details"].split(",")[0].replace("-", "").replace(" ", "").lower()
            pokeName = get_close_matches(pokeName, self.pokeData.keys(), n=1, cutoff=0.5)[0]
            pokeHp = eval(porySide[poke]["condition"].split(" ")[0])
            pokeStatus = porySide[poke]["condition"].split(" ")[1] if len(porySide[poke]["condition"].split(" ")) > 1 else "none"
            pokeAbility = porySide[poke]["ability"]
            pokeMoves = porySide[poke]["moves"]
            pokeStats = porySide[poke]["stats"]
            pokeItem = porySide[poke]["item"]
            pokeTeraType = porySide[poke]["teraType"].lower()

            if poke == 0: # The first mon in the array is always the active one.
                            # Pokemon type, id, stats fetched from pokedata
                self.state["playerSide"]["activeMon"]["id"] = self.pokeData[pokeName]["id"]
                self.state["playerSide"]["activeMon"]["stats"]["baseSpeed"] = self.pokeData[pokeName]["baseSpeed"]
                self.state["playerSide"]["activeMon"]["type1"] = self.pokeData[pokeName]["type1"]
                self.state["playerSide"]["activeMon"]["type2"] = self.pokeData[pokeName]["type2"]

                # Condition and TeraType from showdown.
                self.state["playerSide"]["activeMon"]["condition"]["hp"] = pokeHp
                self.state["playerSide"]["activeMon"]["condition"]["status"] = self.miscData["conditions"][pokeStatus]
                self.state["playerSide"]["activeMon"]["teraType"] = self.miscData["types"][pokeTeraType]

                if("canTerastallize" in newState["active"][0]):
                    self.state["playerSide"]["activeMon"]["terrastillized"] = 0 if (not newState["active"][0]["canTerastallize"]) else 1

                # Ability and item fetched from abilityData and itemData
                self.state["playerSide"]["activeMon"]["ability"] = self.abilityData[pokeAbility]
                self.state["playerSide"]["activeMon"]["item"] = self.itemData[pokeItem]

                if((len(pokeMoves) > 1)):
                    for move in range(len(activeMoves)): # Moves fetched from moveData
                        self.state["playerSide"]["activeMon"]["moves"][move] = self.moveData[pokeMoves[move]]

                        if("pp" in activeMoves[move].keys()):
                            self.state["playerSide"]["activeMon"]["moves"][move]["pp"] = activeMoves[move]["pp"]
                            self.state["playerSide"]["activeMon"]["moves"][move]["disabled"] = 0 if not activeMoves[move]["disabled"] else 1
                        self.state["playerSide"]["activeMon"]["moves"][move]["locked"] = 0
                        self.state["playerSide"]["activeMon"]["condition"]["struggling"] = 0
                elif(pokeMoves[0] != "Struggle"):
                    self.state["playerSide"]["activeMon"]["moves"][pokeMoves[0]]["locked"] = 1
                else:
                    self.state["playerSide"]["activeMon"]["condition"]["struggling"] = 1
                for stat in pokeStats.keys():
                    self.state["playerSide"]["activeMon"]["stats"][stat] = pokeStats[stat]
            else:
                # Pokemon type, id, stats fetched from pokedata
                self.state["playerSide"]["reserves"][poke-1]["id"] = self.pokeData[pokeName]["id"]
                self.state["playerSide"]["reserves"][poke-1]["stats"]["baseSpeed"] = self.pokeData[pokeName]["baseSpeed"]
                self.state["playerSide"]["reserves"][poke-1]["type1"] = self.pokeData[pokeName]["type1"]
                self.state["playerSide"]["reserves"][poke-1]["type2"] = self.pokeData[pokeName]["type2"]

                # Condition and TeraType from showdown.
                self.state["playerSide"]["reserves"][poke-1]["condition"]["hp"] = pokeHp
                self.state["playerSide"]["reserves"][poke-1]["condition"]["status"] = self.miscData["conditions"][pokeStatus]
                self.state["playerSide"]["reserves"][poke-1]["teraType"] = self.miscData["types"][pokeTeraType]
                self.state["playerSide"]["reserves"][poke-1]["terrastillized"] = 0 if not porySide[poke]["terastallized"] else 1


                # Ability and item fetched from abilityData and itemData
                self.state["playerSide"]["reserves"][poke-1]["ability"] = self.abilityData[pokeAbility]
                self.state["playerSide"]["reserves"][poke-1]["item"] = self.itemData[pokeItem]

                for move in range(len(pokeMoves)): # Moves fetched from moveData
                    self.state["playerSide"]["reserves"][poke-1]["moves"][move] = self.moveData[pokeMoves[move]]
                for stat in pokeStats.keys():
                    self.state["playerSide"]["reserves"][poke-1]["stats"][stat] = pokeStats[stat]

    def updateStateNoActive(self, newState):

        # Extract information from newState
        porySide = newState["side"]["pokemon"]
        self.state["request"] = self.miscData["request"][list(newState.keys())[0]] #Set request state

        
        for poke in range(len(porySide)):
            pokeName = porySide[poke]["details"].split(",")[0].replace("-", "").replace(" ", "").lower()
            pokeName = get_close_matches(pokeName, self.pokeData.keys(), n=1, cutoff=0.5)[0]
            pokeHp = eval(porySide[poke]["condition"].split(" ")[0])
            pokeStatus = porySide[poke]["condition"].split(" ")[1] if len(porySide[poke]["condition"].split(" ")) > 1 else "none"
            pokeAbility = porySide[poke]["ability"]
            pokeMoves = porySide[poke]["moves"]
            pokeStats = porySide[poke]["stats"]
            pokeItem = porySide[poke]["item"]
            pokeTeraType = porySide[poke]["teraType"].lower()

            
            # Pokemon type, id, stats fetched from pokedata
            self.state["playerSide"]["reserves"][poke-1]["id"] = self.pokeData[pokeName]["id"]
            self.state["playerSide"]["reserves"][poke-1]["stats"]["baseSpeed"] = self.pokeData[pokeName]["baseSpeed"]
            self.state["playerSide"]["reserves"][poke-1]["type1"] = self.pokeData[pokeName]["type1"]
            self.state["playerSide"]["reserves"][poke-1]["type2"] = self.pokeData[pokeName]["type2"]

            # Condition and TeraType from showdown.
            self.state["playerSide"]["reserves"][poke-1]["condition"]["hp"] = pokeHp
            self.state["playerSide"]["reserves"][poke-1]["condition"]["status"] = self.miscData["conditions"][pokeStatus]
            self.state["playerSide"]["reserves"][poke-1]["teraType"] = self.miscData["types"][pokeTeraType]
            self.state["playerSide"]["reserves"][poke-1]["terrastillized"] = 0 if not porySide[poke]["terastallized"] else 1


            # Ability and item fetched from abilityData and itemData
            self.state["playerSide"]["reserves"][poke-1]["ability"] = self.abilityData[pokeAbility]
            self.state["playerSide"]["reserves"][poke-1]["item"] = self.itemData[pokeItem]

            for move in range(len(pokeMoves)): # Moves fetched from moveData
                self.state["playerSide"]["reserves"][poke-1]["moves"][move] = self.moveData[pokeMoves[move]]
            for stat in pokeStats.keys():
                self.state["playerSide"]["reserves"][poke-1]["stats"][stat] = pokeStats[stat]


    def updateTurnState(self, turnData, turnCount):
        # Extract information from Turn Data.
        for line in turnData:
            splitData = line.split("|")

            # Check for a new pokemon
            if "move" in splitData[1]:
                # If the opponnet has used a move, and their pokémon hasn't been recorded yet, mark it as the active pokemon.
                if "p2a:" in splitData[2]:
                    if self.state["opposingSide"]["activeMon"]["id"] == 0: # Likely start of battle
                        self.recordActiveMon(splitData[2])
                    # Check if opponent is struggling
                    if "Struggle" in splitData[2]:
                        self.state["opposingSide"]["activeMon"]["condition"]["struggling"] = 1
                    else:
                        self.state["opposingSide"]["activeMon"]["condition"]["struggling"] = 0
                    
                    # Record move used by opponent, if it hasn't been recorded before.
                    moveUsed = splitData[3].replace(" ", "").replace("-", "").lower()
                    for move in range(len(self.state["opposingSide"]["activeMon"]["moves"])):
                        if self.state["opposingSide"]["activeMon"]["moves"][move]["id"] == self.moveData[moveUsed]["id"]:
                            break # Exit early if it's been recorded already.
                        if self.state["opposingSide"]["activeMon"]["moves"][move]["id"] == 0: # Look for empty move slot
                            self.state["opposingSide"]["activeMon"]["moves"][move]["id"] = self.moveData[moveUsed]["id"]
                            self.state["opposingSide"]["activeMon"]["moves"][move]["type"] = self.moveData[moveUsed]["type"]
                            self.state["opposingSide"]["activeMon"]["moves"][move]["category"] = self.moveData[moveUsed]["category"]
                            self.state["opposingSide"]["activeMon"]["moves"][move]["power"] = self.moveData[moveUsed]["power"]
                            break

            # When a switch occurs, add the current active mon to reserves and record the new one.
            if 'switch' in splitData[1] and "p2" in splitData[2] and turnCount > 0:
                self.addReserves(self.state["opposingSide"]["activeMon"])
                self.recordActiveMon(splitData[2])
                
            if 'switch' in splitData[1] and "p2" in splitData[2] and turnCount is 0: # No need for reserves on first turn.
                self.recordActiveMon(splitData[2])
                
            # Record any changes in HP or status.
            if '-damage' in splitData[1] or '-heal' in splitData[1]:
                status = splitData[3].split(" ")

                if "p1" in splitData[2]:
                    self.state["playerSide"]["activeMon"]["condition"]["hp"] = eval(status[0])

                    if len(status) > 1:
                        print("status check", status[1])
                        self.state["playerSide"]["activeMon"]["condition"]["status"] = self.miscData["conditions"][status[1]]

                elif "p2" in splitData[2]:
                    self.state["opposingSide"]["activeMon"]["condition"]["hp"] = eval(status[0])

                    if len(status) > 1:
                        print("status check", status[1])
                        self.state["opposingSide"]["activeMon"]["condition"]["status"] = self.miscData["conditions"][status[1]]
        
            
        print("Turn State: ", self.state)

    def recordActiveMon(self, opponentMon):
        opponentMon = opponentMon.split(":")[1].replace(" ", "").replace("-", "").lower()
        opponentMon = get_close_matches(opponentMon, self.pokeData.keys(), n=1, cutoff=0.5)[0]

        self.state["opposingSide"]["activeMon"]["id"] = self.pokeData[opponentMon]["id"]
        self.state["opposingSide"]["activeMon"]["type1"] = self.pokeData[opponentMon]["type1"]
        self.state["opposingSide"]["activeMon"]["type2"] = self.pokeData[opponentMon]["type2"]
        self.state["opposingSide"]["activeMon"]["stats"]["baseSpeed"] = self.pokeData[opponentMon]["baseSpeed"]
        print("New Opponent Mon: ", self.state["opposingSide"]["activeMon"])

    # Add a new reserve to the opposingside's reserves list. If the mon is already in the reserves, do nothing. 
    def addReserves(self, newReserve):
        
        # A painful way to learn about python's pass by reference.
        opponentMon = copy.deepcopy(newReserve)
            
        for mon in range(len(self.state["opposingSide"]["reserves"])):
            if self.state["opposingSide"]["reserves"][mon]["id"] == opponentMon["id"]:
                return
            
            # Can overwrite a mon that's currently active as well.
            if self.state["opposingSide"]["reserves"][mon]["id"] == 0 or self.state["opposingSide"]["activeMon"]["id"] == self.state["opposingSide"]["reserves"][mon]["id"]:
                for stat in opponentMon["stats"]:
                    if stat != "baseSpeed":
                        opponentMon["stats"][stat] = 0
                for condition in opponentMon["condition"]:
                    if condition not in ["hp", "status"]:
                        opponentMon["condition"][condition] = 0
                        
                self.state["opposingSide"]["reserves"][mon] = opponentMon

                return
        

    def countTurn(self, turnData):
        pass

    def reset(self):
        self.knownOpponents = []

