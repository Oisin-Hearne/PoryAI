import json

class Interpreter:

    def __init__(self):
        self.loadData()

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
        
        for poke in range(len(porySide)):
            pokeName = porySide[poke]["details"].split(",")[0].replace("-", "").replace(" ", "").lower()
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
                self.state["playerSide"]["activeMon"]["terrastillized"] = 0 if not newState["active"][0]["canTerastallize"] else 1

                # Ability and item fetched from abilityData and itemData
                self.state["playerSide"]["activeMon"]["ability"] = self.abilityData[pokeAbility]
                self.state["playerSide"]["activeMon"]["item"] = self.itemData[pokeItem]

                for move in range(len(pokeMoves)): # Moves fetched from moveData
                    self.state["playerSide"]["activeMon"]["moves"][move] = self.moveData[pokeMoves[move]]
                    self.state["playerSide"]["activeMon"]["moves"][move]["pp"] = activeMoves[move]["pp"]
                    self.state["playerSide"]["activeMon"]["moves"][move]["disabled"] = 0 if not activeMoves[move]["disabled"] else 1
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
        pass

    def updateStateTeamPreview(self, newState):
        pass

    def countTurn(self, turnData):
        pass

