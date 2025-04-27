import websockets
import requests
import json
import random
import interpreter
import agent
import concurrent.futures
import timeit
import time
from datetime import datetime
class Showdown:
    def __init__(self, uri, user, password, websocket, format, challenger, verbose, opponent, ladder=False):
        self.uri = uri
        self.user = user
        self.password = password
        self.websocket = websocket
        self.inter = interpreter.Interpreter()
        self.format = format
        self.challenger = challenger
        self.opponent = opponent
        self.player = ""
        self.verbose = verbose
        self.ladder = ladder

    # Logging in to a registered showdown instance.
    # (Registered showdown instances require passwords to log in and do validation.)
    async def connectToShowdown(self):
        self.socket = await websockets.connect(self.websocket)
            # challstr and chall id represent the current user token.
            # we need to retrieve it from the showdown server. When not logged in,
            # the server will send the user their challstr and challid whcih we receive below.
        challid = 0
        challstr = ""
        received = ""

        while(challstr == ""):
            received = await self.socket.recv()
            msgs = received.split("|")
            if(msgs[1] == "challstr"):
                challid = msgs[2]
                challstr = msgs[3]
                challstr = "|".join([challid, challstr]) # Join these together to form the token to be sent to the server

        # HTTP Request 
        loggedIn = requests.post(
            self.uri,
            data={
                "act": "login",
                "name": self.user,
                "pass": self.password,
                "challstr": challstr,
            }
        )

        # Finish logging in by grabbing the "assertion" given to us by showdown in JSON.
        jsonOut = json.loads(loggedIn.text[1:])
        await self.sendMessage(f"|/trn {self.user},0,{jsonOut['assertion']}|")

    # When we're on a local server, just set the username.
    async def connectNoSecurity(self):
        self.socket = await websockets.connect(self.websocket)
        await self.sendMessage(f"|/trn {self.user}")

    # Helper method for sending messages to the showdown server.
    # May be refactored in the future. At the moment, this has caused more trouble than help.
    async def sendMessage(self, message):
        await self.socket.send([message])
        return await self.socket.recv()
    
    # Join the queue for the specified format, returning the battle tag.
    async def joinQueue(self, format):
        recv = await self.sendMessage(f"|/search {format}|")

        while True:
            recv = recv.split("|")
            if self.verbose:
                print(recv)
            if 'battle' in recv[0]:
                return recv[0][1:].strip()
            elif len(recv) > 4:
                if 'challenge' in recv[4]:
                    await self.sendMessage(f"|/accept {recv[2]}|")
            recv = await self.socket.recv()

    # Challenge a specified user to a battle and return the battle tag.
    async def challengeUser(self, format, user):
        
        recv = await self.sendMessage(f"|/challenge {user}, {format}|")
        while True:
            recv = recv.split("|")
            if 'battle' in recv[0] and 'init' in recv[1] and not 'deinit' in recv[1]:
                return recv[0][1:].strip()
            
            recv = await self.socket.recv()
            
    # Wait for a challenge from a specified user and return the battle tag.
    async def waitForChallenge(self, user):
        while True:
            recv = await self.socket.recv()
            recv = recv.split("|")
            if len(recv) > 4 and '/challenge gen9randombattle' in recv[4]:
                await self.socket.send(f"|/accept {user}")
            if len(recv) > 1 and 'battle' in recv[0] and 'init' in recv[1] and not 'deinit' in recv[1]:
                return recv[0][1:].strip()
            

    # Runs the logic for the given battle tag
    # Communicates with the Interpreter for state,
    # and the agent for actions.
    async def manageBattle(self):
        self.currentRewards = 0


        while True:
            recv = await self.socket.recv()
            msgs = recv.split("|")
            if self.verbose:
                print(recv)
            battleStarted = False
            
            
            # This big long messy if-else chain is somewhat necessary but could likely be cleaned up quite a bit.
            # This formed mostly through trial and error, as it was difficult to form a set of conditions that would work for all cases.
            # Check what's received from the server, and perform different actions accordingly.
            
            # p1 and p2 are determined alphabetically, so we need to check which one the user is.
            if f"player|p1|{self.user}" in recv:
                self.player = "p1"
            elif f"player|p2|{self.user}" in recv:
                self.player = "p2"
                
            # New battle, set a new tag.
            if ">battle" in msgs[0]:
                chatTag = msgs[0].strip().replace(">", "")
                if chatTag != self.currentTag:
                    self.currentTag = chatTag
                    print(f"New tag: {self.currentTag}")
                    await self.socket.send([f"{self.currentTag}|{self.currentAction}|{self.latestRequest['rqid']}"])
            if 'start\n' in recv and not battleStarted: # Reset previous info.
                battleStarted = True
                _, _, firstTurn = recv.partition("start\n")
                self.inter.updateTurnState(firstTurn.split("\n"), True, self.player)
                await self.socket.send([f"{self.currentTag}|/timer on"]) # Make sure stalling doesn't happen.
                await self.socket.send([f"{self.currentTag}|I'm PoryAI, a RL-Bot. I learn by playing!"]) # Inform users that they are battling a bot.
            
            elif '/choose - must be used in a chat room' in recv: # If the bot sends a decision too fast, it may need to wait and send it again.
                time.sleep(1)
                await self.socket.send(self.currentCommand)
                
            elif "Can't switch: You have to pass to a fainted Pokémon" in recv: # Handling Revival Blessing, a niche move that causes issues.
                await self.socket.send([f"{self.currentTag}|/choose default|{self.latestRequest['rqid']}"])
                return False, 0
        
            # Chat messages. Need to be covered before the other statements.
            elif msgs[1] in ["l", "deinit", "c", "j", "J", "L"]:
                if self.verbose:
                    print(msgs)
            
            # Requests for the user to do something. These should be sent to the interpreter.
            elif 'request' in msgs[1] and len(msgs[2]) > 2:
                requestOutput = json.loads(msgs[2])
                
                if "active" in requestOutput:
                    self.inter.updateStateActive(requestOutput)
                elif "forceSwitch" in requestOutput or "wait" in requestOutput:
                    self.inter.updateStateNoActive(requestOutput)
                
                if not "wait" in requestOutput:
                    self.latestRequest = requestOutput
                    return False, 0 # Battle not done

            # Typically means a turn has ended, so take all the data from the turn and send it to the interpreter.
            elif ('t:' in msgs[2]) and (len(recv.split("\n")) > 3) and ("Time left" not in recv):
                # Send turn content to interpreter here, then reset it.
                turnContent = recv.split("\n")[3:]
                
                self.state = self.inter.updateTurnState(turnContent, self.turnCount, self.player)

                self.currentRewards, battleContent = self.inter.countTurn(turnContent, self.currentCommand, self.player)
                self.battleLog += battleContent + "\n"
                
                if self.verbose:
                    print(f"Turn {self.turnCount}:{turnContent} \n Rewards: {self.currentRewards} \n State: {self.state} \n")
                
                self.turnCount += 1
                turnContent = []

            # Turn has been updated, append it.
            elif self.turnCount > 0 and (msgs[1] in ["switch", "move", "faint"] or msgs[1][1] == "-"):
                turnContent.append(recv)
            
            # Don't count forfeits or timeouts as wins.
            if 'forfeited.' in recv or "due to inactivity" in recv:
                print("Opponent forfeited!")
                await self.socket.send([f"|/leave {self.currentTag}"])
                with open(f"data/logs/battles/forfeit-{self.currentTag}.txt", "a") as f:
                    f.write(self.battleLog)
                print(f"Opponent forfeited the battle!\n=====================================================================")
                return True, 0
            
            if '|win|' in recv: # Battle is over.
                
                await self.socket.send([f"|/leave {self.currentTag}"])
                
                result = "Won" if ("win|"+self.user) in recv else "Lost"
                timestamp = datetime.now().strftime("%Y_%m%d-%p%I_%M_%S")
                # Write battle to file
                if self.verbose:
                    with open(f"data/logs/battles/{result}-{self.currentTag}-{timestamp}.txt", "a") as f:
                        f.write(self.battleLog)
                    print(f"User {self.user} {result} the battle!\n=====================================================================")
                
                if f'win|{self.user}' in recv:
                    return True, 1
                else:
                    return True, -1


    async def run(self):
        await self.connectNoSecurity()
        while True:
            battleTag = await self.challengeFoulPlay(self.format)
            await self.manageBattle(battleTag)
    
    # Reset the state, turn count and battle log and start a new battle based on the current mode.
    async def restart(self):
        self.inter.resetState()
        self.turnCount = 0
        self.battleLog = ""
        if self.challenger:
            self.currentTag = await self.challengeUser(self.format, self.opponent)
        elif self.ladder:
            self.currentTag = await self.joinQueue(self.format)
        else: 
            self.currentTag = await self.waitForChallenge(self.opponent)

        await self.manageBattle()
    
    def getState(self):
        return self.inter.state
        
    # Send an action and get the results of it.
    async def executeAction(self, action):
        self.currentAction = action
        self.currentCommand = [f"{self.currentTag}|{action}|{self.latestRequest['rqid']}"]
        await self.socket.send(self.currentCommand)
        battleDone, winner = await self.manageBattle()
        newState = self.inter.state
        rewards = self.currentRewards
        
        return newState, rewards, battleDone, winner
        

    # A list of possible valid actions is needed for the agent to choose from.
    # This would perhaps fit better in the interpreter, but for now it's here.
    def getValidActions(self):
        valid_actions = []
        
        # Showdown requesting a move/switch.
        if 'active' in self.latestRequest:
            
            # Check all the moves, and add the ones that can be used.
            
            # If there's only one move in latestRequest, they can only use that move.
            if len(self.latestRequest['active'][0]['moves']) == 1:
                valid_actions.append("/choose move 1")
                return valid_actions
            
            # If the agent can terastallize, let them.
            for move in range(len(self.latestRequest['active'][0]['moves'])):
                if not self.latestRequest["active"][0]["moves"][move]["disabled"]:
                    valid_actions.append(f"/choose move {move+1}")
                    if "canTerastallize" in self.latestRequest["active"][0] and self.latestRequest["active"][0]["canTerastallize"]:
                        valid_actions.append(f"/choose move {move+1} terastal")
                        
            # Player can't switch if they're trapped.
            if not ("trapped" in self.latestRequest["active"][0] and self.latestRequest["active"][0]["trapped"]):
                for mon in range(len(self.latestRequest['side']['pokemon'])):
                
                # Big ugly chain so it's more readable.
                    if self.latestRequest['side']['pokemon'][mon]['active'] == False: # If the mon isn't active...
                        if self.latestRequest['side']['pokemon'][mon]['condition'] != "0 fnt": # And it's not fainted...
                            valid_actions.append(f"/choose switch {mon+1}") # Add the switch.
        
        # Force switch - a mon has just fainted or something.
        if "forceSwitch" in self.latestRequest:
            for mon in range(len(self.latestRequest['side']['pokemon'])):
                if self.latestRequest['side']['pokemon'][mon]['active'] == False:
                    if self.latestRequest['side']['pokemon'][mon]['condition'] != "0 fnt":
                        valid_actions.append(f"/choose switch {mon+1}")
                        
        # If an action isn't found, something's gone wrong. 
        if valid_actions == []:
            print("No options found! Going with default...")
            valid_actions.append("/choose default")
            
        return valid_actions