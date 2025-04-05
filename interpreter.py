import json
import math
import copy
from difflib import get_close_matches

class Interpreter:

    def __init__(self):
        self.loadData()
        self.stats = {"superEffective": 0, "resisted": 0, "fainted": 0, "boosted": 0, "damaged": 0, "setHazards": 0, "clearedHazards": 0, "switched": 0, "repeatMoves": 0}
        self.actionRatio = 0.0
        self.repeatMoves = 0

    # Load in previously fetched data for use in state.
    def loadData(self):
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
        with open('data/rewards.json') as f:
            self.rewards = json.load(f)
            
    def resetState(self):
        self.prevSelfHp = 1
        self.prevOppHp = 1
        self.prevAction = ""
        self.opponentHalved = False
        self.action_counts = {'move1': 0, 'move2': 0, 'move3': 0, 'move4':0, 'switch': 0}
        self.actionRatio = 0
        self.repeatMoves = 0
        
        with open('data/sample-state.json') as f:
            self.state = json.load(f)

    def resetStats(self):
        self.stats = {"superEffective": 0, "resisted": 0, "fainted": 0, "boosted": 0, "damaged": 0, "setHazards": 0, "clearedHazards": 0, "switched": 0, "repeatMoves": 0}
    def getStats(self):
        return self.stats

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
            pokeAbility = get_close_matches(pokeAbility, self.abilityData.keys(), n=1, cutoff=0.5)[0]
            pokeMoves = porySide[poke]["moves"]
            pokeStats = porySide[poke]["stats"]
            pokeItem = porySide[poke]["item"]
            pokeTeraType = porySide[poke]["teraType"].lower()

            if poke == 0: # The first mon in the array is always the active one.
                            # Pokemon type, id, stats fetched from pokedata
                self.state["playerSide"]["activeMon"]["pokeid"] = self.pokeData[pokeName]["pokeid"]
                self.state["playerSide"]["activeMon"]["stats"]["baseSpeed"] = self.pokeData[pokeName]["baseSpeed"]
                self.state["playerSide"]["activeMon"]["type1"] = self.pokeData[pokeName]["type1"]
                self.state["playerSide"]["activeMon"]["type2"] = self.pokeData[pokeName]["type2"]

                # Condition and TeraType from showdown.
                self.state["playerSide"]["activeMon"]["condition"]["hp"] = pokeHp
                self.state["playerSide"]["activeMon"]["condition"]["status"] = self.miscData["conditions"][pokeStatus]
                self.state["playerSide"]["activeMon"]["teraType"] = self.miscData["types"][pokeTeraType]

                if("canTerastallize" in newState["active"][0]):
                    self.state["playerSide"]["activeMon"]["terrastillized"] = 0

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
                elif(pokeMoves[0] != "Transform"):
                    print("Transform detected")
                    self.state["playerSide"]["activeMon"]["moves"][0] = self.moveData[pokeMoves[0]]
                elif(pokeMoves[0] != "Struggle"):
                    self.state["playerSide"]["activeMon"]["moves"][pokeMoves[0]]["locked"] = 1
                else:
                    self.state["playerSide"]["activeMon"]["condition"]["struggling"] = 1
                for stat in pokeStats.keys():
                    self.state["playerSide"]["activeMon"]["stats"][stat] = pokeStats[stat]
            else:
                # Pokemon type, id, stats fetched from pokedata
                self.state["playerSide"]["reserves"][poke-1]["pokeid"] = self.pokeData[pokeName]["pokeid"]
                self.state["playerSide"]["reserves"][poke-1]["stats"]["baseSpeed"] = self.pokeData[pokeName]["baseSpeed"]
                self.state["playerSide"]["reserves"][poke-1]["type1"] = self.pokeData[pokeName]["type1"]
                self.state["playerSide"]["reserves"][poke-1]["type2"] = self.pokeData[pokeName]["type2"]

                # Condition and TeraType from showdown.
                self.state["playerSide"]["reserves"][poke-1]["condition"]["hp"] = pokeHp
                self.state["playerSide"]["reserves"][poke-1]["condition"]["status"] = self.miscData["conditions"][pokeStatus]
                self.state["playerSide"]["reserves"][poke-1]["teraType"] = self.miscData["types"][pokeTeraType]


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
            pokeAbility = get_close_matches(pokeAbility, self.abilityData.keys(), n=1, cutoff=0.5)[0]
            pokeMoves = porySide[poke]["moves"]
            pokeStats = porySide[poke]["stats"]
            pokeItem = porySide[poke]["item"]
            pokeTeraType = porySide[poke]["teraType"].lower()

            
            # Pokemon type, id, stats fetched from pokedata
            self.state["playerSide"]["reserves"][poke-1]["pokeid"] = self.pokeData[pokeName]["pokeid"]
            self.state["playerSide"]["reserves"][poke-1]["stats"]["baseSpeed"] = self.pokeData[pokeName]["baseSpeed"]
            self.state["playerSide"]["reserves"][poke-1]["type1"] = self.pokeData[pokeName]["type1"]
            self.state["playerSide"]["reserves"][poke-1]["type2"] = self.pokeData[pokeName]["type2"]

            # Condition and TeraType from showdown.
            self.state["playerSide"]["reserves"][poke-1]["condition"]["hp"] = pokeHp
            self.state["playerSide"]["reserves"][poke-1]["condition"]["status"] = self.miscData["conditions"][pokeStatus]
            self.state["playerSide"]["reserves"][poke-1]["teraType"] = self.miscData["types"][pokeTeraType]


            # Ability and item fetched from abilityData and itemData
            self.state["playerSide"]["reserves"][poke-1]["ability"] = self.abilityData[pokeAbility]
            self.state["playerSide"]["reserves"][poke-1]["item"] = self.itemData[pokeItem]

            for move in range(len(pokeMoves)): # Moves fetched from moveData
                self.state["playerSide"]["reserves"][poke-1]["moves"][move] = self.moveData[pokeMoves[move]]
            for stat in pokeStats.keys():
                self.state["playerSide"]["reserves"][poke-1]["stats"][stat] = pokeStats[stat]

    def updateTurnState(self, turnData, startOfBattle, player):
        self.prevSelfHp = copy.deepcopy(self.state["playerSide"]["activeMon"]["condition"]["hp"])
        self.prevOppHp = copy.deepcopy(self.state["opposingSide"]["activeMon"]["condition"]["hp"])
        opponent = "p1" if player == "p2" else "p2"
        
        # Extract information from Turn Data.
        for line in turnData:
            splitData = line.split("|")

            # Check for a new pokemon
            if "move" in splitData[1]:
                # If the opponnet has used a move, and their pokémon hasn't been recorded yet, mark it as the active pokemon.
                if opponent+"a:" in splitData[2]:
                    #if self.state["opposingSide"]["activeMon"]["id"] == 0: # Likely start of battle
                    #    self.recordActiveMon(splitData[3:])
                    # Check if opponent is struggling
                    if "Struggle" in splitData[2]:
                        self.state["opposingSide"]["activeMon"]["condition"]["struggling"] = 1
                    else:
                        self.state["opposingSide"]["activeMon"]["condition"]["struggling"] = 0
                    
                    # Record move used by opponent, if it hasn't been recorded before.
                    moveUsed = splitData[3].replace(" ", "").replace("-", "").lower()
                    for move in range(len(self.state["opposingSide"]["activeMon"]["moves"])):
                        if self.state["opposingSide"]["activeMon"]["moves"][move]["moveid"] == self.moveData[moveUsed]["moveid"]:
                            break # Exit early if it's been recorded already.
                        if self.state["opposingSide"]["activeMon"]["moves"][move]["moveid"] == 0: # Look for empty move slot
                            self.state["opposingSide"]["activeMon"]["moves"][move]["moveid"] = self.moveData[moveUsed]["moveid"]
                            self.state["opposingSide"]["activeMon"]["moves"][move]["type"] = self.moveData[moveUsed]["type"]
                            self.state["opposingSide"]["activeMon"]["moves"][move]["category"] = self.moveData[moveUsed]["category"]
                            self.state["opposingSide"]["activeMon"]["moves"][move]["power"] = self.moveData[moveUsed]["power"]
                            break

            # When a switch occurs, add the current active mon to reserves and record the new one.
            if 'switch' in splitData[1] and opponent in splitData[2] and not startOfBattle:
                self.addReserves(self.state["opposingSide"]["activeMon"])
                self.recordActiveMon(splitData[3:])
                
            if 'switch' in splitData[1] and opponent in splitData[2] and startOfBattle: # No need for reserves on first turn.
                self.recordActiveMon(splitData[3:])
                
            # Record any changes in HP or status.
            if '-damage' in splitData[1] or '-heal' in splitData[1]:
                status = splitData[3].split(" ")
                side = "playerSide" if player in splitData[2] else "opposingSide"
                print(f"Changing HP of {side} by {status[0]}. Current hp is {self.state[side]['activeMon']['condition']['hp']}")
                self.state[side]["activeMon"]["condition"]["hp"] = round(eval(status[0]), 2)

                if len(status) > 1:
                    print(f"Changing status of {side} to {status[1]}. Current status is {self.state[side]['activeMon']['condition']['status']}")
                    self.state[side]["activeMon"]["condition"]["status"] = self.miscData["conditions"][status[1]]
            if "-curestatus" in splitData[1]:
                side = "playerSide" if player in splitData[2] else "opposingSide"
                self.state[side]["activeMon"]["condition"]["status"] = 0
                        
            if '-boost' in splitData[1] or '-unboost' in splitData[1]:
                boost = splitData[4] if '-boost' in splitData[1] else "-" + splitData[4] # Positive or negative boost
                stat = splitData[3]+"Mod"
                side = "playerSide" if player in splitData[2] else "opposingSide"
                
                print(f"Boosting {stat} by {boost}. Current stat is {self.state[side]['activeMon']['stats'][stat]}, new stat will be {eval(str(self.state[side]['activeMon']['stats'][stat])+'+'+boost)}")

                self.state[side]["activeMon"]["stats"][stat] = eval(str(self.state[side]["activeMon"]["stats"][stat])+"+"+boost)
            
            # Clear all boosts below zero.
            if "-clearnegativeboost" in splitData[1]:
                side = "playerSide" if player in splitData[2] else "opposingSide"
                for stat in self.state[side]["activeMon"]["stats"]:
                    if self.state[side]["activeMon"]["stats"][stat] < 0:
                        self.state[side]["activeMon"]["stats"][stat] = 0
                        
            if "switch" in splitData[1] or "clearboost" in splitData[1]:
                side = "playerSide" if player in splitData[2] else "opposingSide"
                for stat in self.state[side]["activeMon"]["stats"]:
                    if "Mod" in stat:
                        self.state[side]["activeMon"]["stats"][stat] = 0
            
            if "-clearallboost" in splitData[1]:
                for stat in self.state["playerSide"]["activeMon"]["stats"]:
                    if "Mod" in stat:
                        self.state["playerSide"]["activeMon"]["stats"][stat] = 0
                        self.state["opposingSide"]["activeMon"]["stats"][stat] = 0
                        
            # Clear volatile conditions
            if 'switch' in splitData[1] or 'faint' in splitData[1]:
                side = "playerSide" if player in splitData[2] else "opposingSide"
                for condition in self.state[side]["activeMon"]["condition"]:
                    if condition not in ["hp", "status"]:
                        self.state[side]["activeMon"]["condition"][condition] = 0
                        
            # Record opponent abilities as they're activated.
            if "-ability" in splitData[1] and opponent in splitData[2]:
                ability = splitData[3].replace(" ", "").replace("-", "").lower()
                ability = get_close_matches(ability, self.abilityData.keys(), n=1, cutoff=0.5)[0]
                self.state["opposingSide"]["activeMon"]["ability"] = self.abilityData[ability]
            
            if len(splitData) > 4:
                if opponent+"a:" in splitData[2] and "[from] item" in splitData[4]:
                    item = splitData[4].split(" ")[2:]
                    item = ''.join(item).replace(" ", "").replace("-", "").lower()
                    self.state["opposingSide"]["activeMon"]["item"] = self.itemData[item]
            if len(splitData) > 5:
                if opponent+"a:" in splitData[5] and "[from] item" in splitData[4]:
                    item = splitData[4].split(" ")[2:]
                    item = ''.join(item).replace(" ", "").replace("-", "").lower()
                    self.state["opposingSide"]["activeMon"]["item"] = self.itemData[item]
                
            if "-item" in splitData[1] and opponent in splitData[2]:
                item = splitData[3].replace(" ", "").replace("-", "").lower()
                self.state["opposingSide"]["activeMon"]["item"] = self.itemData[item]
                
            if "-start" in splitData[1] or "-end" in splitData[1]:
                side = "playerSide" if player in splitData[2] else "opposingSide"
                newStatus = 0 if "-end" in splitData[1] else 1
                condition = splitData[3]
                if "Encore" in condition:
                    self.state[side]["activeMon"]["condition"]["encored"] = newStatus
                elif "Leech Seed" in condition:
                    self.state[side]["activeMon"]["condition"]["leechSeed"] = newStatus
                elif "Perish Song" in condition:
                    self.state[side]["activeMon"]["condition"]["perishSong"] = newStatus
                elif "Taunt" in condition:
                    self.state[side]["activeMon"]["condition"]["taunted"] = newStatus
                elif "confusion" in condition:
                    self.state[side]["activeMon"]["condition"]["confusion"] = newStatus
                elif "Substitute" in condition:
                    self.state[side]["activeMon"]["condition"]["substitute"] = newStatus
                    
            if "-sidestart" in splitData[1] or "-sideend" in splitData[1]:
                side = "playerSide" if player in splitData[2] else "opposingSide"
                newStatus = 0 if "-sideend" in splitData[1] else 1
                
                # Showdown sends side effects weirdly, occasionally having move: sideffect, so we need to parse it.
                sideEffect = splitData[3].split(" ")
                if "move:" in sideEffect[0]:
                    sideEffect = "".join(sideEffect[1:])
                else:
                    sideEffect = "".join(sideEffect[0:])
                
                sideEffect = sideEffect.replace(" ", "").replace("-", "").lower()
                
                self.state[side]["effects"][sideEffect] = newStatus
                
            if "-weather" in splitData[1]:
                weather = splitData[2].lower()
                self.state["universal"]["weather"] = self.miscData["weather"][weather]
                
            if "-fieldstart" in splitData[1] or "-fieldend" in splitData[1]:
                newStatus = -1 if "-fieldend" in splitData[1] else 1
                
                if "Terrain" in splitData[2]:
                    fieldEffect = splitData[2].split(" ")[1:]
                    fieldEffect = "".join(fieldEffect).replace(" ", "").replace("-", "").lower()
                elif "Trick Room" in splitData[2]:
                    fieldEffect = splitData[2].split(" ")[1:]
                    fieldEffect = "".join(fieldEffect).replace(" ", "").replace("-", "").lower()
                else:
                    fieldEffect = splitData[3].split(" ")[1:]
                    fieldEffect = "".join(fieldEffect).replace(" ", "").replace("-", "").lower()
                
                self.state["universal"][fieldEffect] = self.state["universal"][fieldEffect] + newStatus
            
            if "-terastallize" in splitData[1]:
                side = "playerSide" if player in splitData[2] else "opposingSide"
                self.state[side]["activeMon"]["terrastillized"] = 1
                self.state[side]["activeMon"]["teraType"] = self.miscData["types"][splitData[3].lower()]
            
            
        print(f"Player: {player}, State: {self.state}")
        return self.state

    # Set the active mon state to the new opponent.
    def recordActiveMon(self, opponentMon):
        opponentName = opponentMon[0].split(",")[0].replace(" ", "").replace("-", "").lower()
        opponentName = get_close_matches(opponentName, self.pokeData.keys(), n=1, cutoff=0.5)[0]
        opponentCondition = opponentMon[1].split(" ")
        print(f"Recording Active Mon: {opponentName}, with condition: {opponentCondition}")
        
        self.state["opposingSide"]["activeMon"]["pokeid"] = self.pokeData[opponentName]["pokeid"]
        self.state["opposingSide"]["activeMon"]["type1"] = self.pokeData[opponentName]["type1"]
        self.state["opposingSide"]["activeMon"]["type2"] = self.pokeData[opponentName]["type2"]
        self.state["opposingSide"]["activeMon"]["stats"]["baseSpeed"] = self.pokeData[opponentName]["baseSpeed"]
        self.state["opposingSide"]["activeMon"]["condition"]["status"] = self.miscData["conditions"][opponentCondition[1]] if len(opponentCondition) > 1 else 0
        self.state["opposingSide"]["activeMon"]["condition"]["hp"] = eval(opponentCondition[0])

    # Add a new reserve to the opposingside's reserves list. If the mon is already in the reserves, do nothing. 
    def addReserves(self, newReserve):
    
        # A painful way to learn about python's pass by reference.
        opponentMon = copy.deepcopy(newReserve)
        print(f"Adding Reserve: {opponentMon}")        
        for mon in range(len(self.state["opposingSide"]["reserves"])):
            if self.state["opposingSide"]["reserves"][mon]["pokeid"] == opponentMon["pokeid"]:
                return
            
            # Can overwrite a mon that's currently active as well.
            if self.state["opposingSide"]["reserves"][mon]["pokeid"] == 0 or self.state["opposingSide"]["activeMon"]["pokeid"] == self.state["opposingSide"]["reserves"][mon]["pokeid"]:
                for stat in opponentMon["stats"]:
                    if stat != "baseSpeed":
                        opponentMon["stats"][stat] = 0
                for condition in opponentMon["condition"]:
                    if condition not in ["hp", "status"]:
                        opponentMon["condition"][condition] = 0
                        
                self.state["opposingSide"]["reserves"][mon] = opponentMon

                return
        

    def countTurn(self, turnData, lastAction, player):
        turnPoints = 0
        opponent = "p1" if player == "p2" else "p2"
    
        # Penalise consecutive switches
        if len(self.prevAction) > 0 and len(lastAction) > 0:
            if "switch" in lastAction[0] and "switch" in self.prevAction[0]:
                #print("Consecutive Switches Detected")
                turnPoints -= self.rewards["switchDecentive"]
        # Slight incentive to attacking
        if "move 1" in lastAction[0]:
            self.action_counts['move1'] += 1
            turnPoints += self.rewards["attackIncentive"]
        elif "move 2" in lastAction[0]:
            self.action_counts['move2'] += 1
            turnPoints += self.rewards["attackIncentive"]
        elif "move 3" in lastAction[0]:
            self.action_counts['move3'] += 1
            turnPoints += self.rewards["attackIncentive"]
        elif "move 4" in lastAction[0]:
            self.action_counts['move4'] += 1
            turnPoints += self.rewards["attackIncentive"]
        elif "switch" in lastAction[0]:
            self.action_counts['switch'] += 1
            
        move_actions = sum([self.action_counts[key] for key in self.action_counts.keys() if "move" in key])
        self.actionRatio = max(0.1, min(move_actions / max(1, self.action_counts["switch"]), 10))
        print(f"Action Ratio: {self.actionRatio}")
        print(f"Action Counts: {self.action_counts}")
        
        #Decentivize too much switching
        if self.actionRatio < 0.5 and "switch" in lastAction[0]:
            print("Detecting too many switches...")
            turnPoints -= self.rewards["ratioPunishment"]
            self.stats["switched"] += 1
            # Do it again if it's really bad
            if self.actionRatio < 0.3:
                turnPoints -= self.rewards["ratioPunishment"]
            if self.actionRatio < 0.1:
                turnPoints -= self.rewards["ratioPunishment"]
        
        # Decentivize using the same move slot too often
        moveCount = sum([self.action_counts[key] for key in self.action_counts.keys() if "move" in key])
        averageMoveCount = moveCount / 4
        #TODO: When a move is locked, don't let it effect this.
                
        if "switch" not in lastAction[0] and self.actionRatio > 0.5 and moveCount > 8:
            for key in self.action_counts.keys():
                if "move" in key and self.action_counts[key] > (averageMoveCount*self.rewards["moveLeeway"]) and key in lastAction[0].replace(" ", ""):
                    print(f"Detecting too many moves in slot {key}... {self.action_counts[key]} > {averageMoveCount*self.rewards['moveLeeway']}")
                    turnPoints -= self.rewards["movePunishment"]
                    self.repeatMoves += 1
                    self.stats["repeatMoves"] += 1
                    
                    # If the move is used too many times, punish it more
                    if self.action_counts[key] > 1.5*(averageMoveCount*self.rewards["moveLeeway"]):
                        turnPoints -= self.rewards["movePunishment"]
                    if self.action_counts[key] > 2*(averageMoveCount*self.rewards["moveLeeway"]):
                        turnPoints -= self.rewards["movePunishment"]
                        
        
        for line in turnData:
            splitData = line.split("|")
            
            
            
            # Action - Dealing/Taking Damage
            if "damage" in splitData[1]:
                side = "playerSide" if player in line else "opposingSide"
                newHp = self.state[side]["activeMon"]["condition"]["hp"]
                
                if side == "playerSide":
                    damage = self.prevSelfHp - newHp
                    self.prevSelfHp = newHp
                else:
                    damage = self.prevOppHp - newHp
                    self.prevOppHp = newHp
                    self.stats["damaged"] += 1
                
                damage = max(0, round(damage, 3))
                #print(f"Damage Dealt: {damage}, Side: {side}, Old Player HP: {self.prevSelfHp}, Old Opp HP: {self.prevOppHp}, New HP: {newHp}")
                # Dealing damage above a certain amount is rewarded, but punished if too little damage is done.
                # Receiving very little damage is rewarded, receiving too much is punished.
                #turnPoints += damageBase if (damage > damageThreshold and side == "opposingSide") or (damage < damageThreshold and side == "playerSide") else -damageBase
                
                # Damage calculation is seemingly off, going with something simpler for now.
                turnPoints += self.rewards["damageBase"] if side == "opposingSide" else -self.rewards["damageBase"]
               
                if self.prevOppHp < 0.5 and not self.opponentHalved:
                    self.opponentHalved = True
                    turnPoints += self.rewards["progressBase"]
                if self.prevOppHp > 0.5 and self.opponentHalved:
                    self.opponentHalved = False
                
            # Action - Healing
            if "heal" in splitData[1]:
                side = "playerSide" if player in line else "opposingSide"
                newHp = self.state[side]["activeMon"]["condition"]["hp"]
                
                if side == "playerSide":
                    heal = newHp - self.prevSelfHp
                    self.prevSelfHp = newHp
                else:
                    heal = newHp - self.prevOppHp
                    self.prevOppHp = newHp  
                         
                heal = max(0, round(heal, 3))
                
                #print(f"Healing Done: {heal}, Side: {side}")   
                # Healing is rewarded for the player and punished for the opponent.
                turnPoints += self.rewards["healBase"] if side == "playerSide" else -self.rewards["healBase"]
                # Additional points for above a certain threshold
                turnPoints += self.rewards["healBase"] if side == "playerSide" and heal > self.rewards["healThreshold"] else -self.rewards["healBase"] if heal > self.rewards["healThreshold"] else 0
                               
            # Action - Knockout
            if "faint" in splitData[1]:
                side = "playerSide" if player in line else "opposingSide"
                turnPoints += self.rewards["koBase"] if side == "opposingSide" else -(self.rewards["koBase"] /2)
                if side == "opposingSide":
                    self.stats["fainted"] += 1
                
            # Action - Status Change
            if "-status" in splitData[1]:
                side = "playerSide" if player in line else "opposingSide"
                status = splitData[3]
                turnPoints += self.rewards["statusBase"] if side == "opposingSide" else -self.rewards["statusBase"]
                turnPoints += self.rewards["statusBonus"] if status in ["tox, slp"] and side == "opposingSide" else -self.rewards["statusBonus"] if status in ["tox, slp"] and side == "playerSide" else 0

                
            if "-curestatus" in splitData[1]:
                side = "playerSide" if player in line else "opposingSide"
                turnPoints += self.rewards["statusBase"] if side == "playerSide" else -self.rewards["statusBase"]
                
            # Action - Boosts/Unboosts
            if "-boost" in splitData[1] or "-unboost" in splitData[1]:
                side = "playerSide" if player in line else "opposingSide"
                modifier = 1 if "-boost" in splitData[1] else -1
                boostAmnt = int(splitData[4])
                turnPoints += (modifier*self.rewards["boostBase"]*boostAmnt) if side == "playerSide" else -(modifier*self.rewards["boostBase"]*boostAmnt)
                
                if side == "playerSide":
                    self.stats["boosted"] += 1        
                        
            # Action - Supereffective & Resisted Hits
            if "supereffective" in splitData[1]:
                side = "playerSide" if player in line else "opposingSide"
                turnPoints += self.rewards["effectiveBase"] if side == "opposingSide" else -self.rewards["effectiveBase"]
                if side == "opposingSide":
                    self.stats["superEffective"] += 1
                
            if "resisted" in splitData[1] or "immune" in splitData[1]:
                side = "playerSide" if player in line else "opposingSide"
                turnPoints += -self.rewards["effectiveBase"] if side == "opposingSide" else self.rewards["effectiveBase"]
                if side == "playerSide":
                    self.stats["resisted"] += 1
                
            # Action - Field Effects
            if "sidestart" in splitData[1]:
                side = "playerSide" if player in line else "opposingSide"
                positives = ["Reflect", "Light Screen", "Aurora Veil", "Tailwind"]
                negatives = ["|Spikes", "Toxic Spikes", "Stealth Rock", "Sticky Web"]
                
                if any(x in line for x in positives):
                    turnPoints += self.rewards["fieldBase"] if side == "playerSide" else -self.rewards["fieldBase"]
                elif any(x in line for x in negatives):
                    turnPoints += self.rewards["hazardSetBase"] if side == "opposingSide" else -self.rewards["hazardSetBase"]
                    if side == "opposingSide":
                        self.stats["setHazards"] += 1
                    
            if "sideend" in splitData[1]:
                side = "playerSide" if player in splitData[2] else "opposingSide"
                
                # Positive and negative field effects listed.
                positives = ["Reflect", "Light Screen", "Aurora Veil", "Tailwind"]
                negatives = ["|Spikes", "Toxic Spikes", "Stealth Rock", "Sticky Web"]
                                
                # If upkeep is in the line, the effect ended naturally and should not be counted.
                if any(x in line for x in positives) and len(splitData) > 4:
                    turnPoints += -self.rewards["fieldBase"] if side == "playerSide" else self.rewards["fieldBase"]
                elif any(x in line for x in negatives):
                    turnPoints += -self.rewards["hazardClearBase"] if side == "opposingSide" else self.rewards["hazardClearBase"]
                    if side == "playerSide":
                        self.stats["clearedHazards"] += 1
            
            # Action - Setting Weather
            if "weather" in splitData[1]:
                side = "playerSide" if player+"a" in line else "opposingSide" if opponent+"a" in line else "universal"
                turnPoints += self.rewards["weatherBase"] if side == "playerSide" else -self.rewards["weatherBase"] if side == "opposingSide" else 0
            
            # Action - Failed Move
            if "-fail" in splitData[1] and player+"a" in line:
                turnPoints -= self.rewards["failBase"]
                
            #if "|win|" in line:
             #   turnPoints += self.rewards["winBase"] if "PoryAI" in line else 0
        
        self.prevAction = lastAction
        
        max_reward = sum([self.rewards[key] for key in self.rewards.keys()])
        actualReward = math.tanh(turnPoints / max_reward )
        turnString = "".join(turnData)+": "+str(actualReward)
        #print(f"Player: {player}, \n Turn String: {turnData}, \n Reward: {actualReward}")
        return actualReward, turnString 



                    
                    
                