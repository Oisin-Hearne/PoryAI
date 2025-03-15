import websockets
import requests
import json
import random
import interpreter
import agent
import concurrent.futures
import timeit
import time

class Showdown:
    def __init__(self, uri, user, password, websocket, format):
        self.uri = uri
        self.user = user
        self.password = password
        self.websocket = websocket
        self.inter = interpreter.Interpreter()
        self.format = format

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

    async def connectNoSecurity(self):
        self.socket = await websockets.connect(self.websocket)
        await self.sendMessage(f"|/trn {self.user}")


    async def sendMessage(self, message):
        await self.socket.send([message])
        return await self.socket.recv()
    
    # Join the queue for the specified format, returning the battle tag.
    async def joinQueue(self, format):
        recv = await self.sendMessage(f"|/search {format}|")

        while True:
            #print(recv)
            recv = recv.split("|")
            if 'battle' in recv[0]:
                return recv[0][1:].strip()
            elif len(recv) > 4:
                if 'challenge' in recv[4]:
                    await self.sendMessage(f"|/accept {recv[2]}|")
            recv = await self.socket.recv()

    async def challengeFoulPlay(self, format):
        foulPlayUser = self.user.replace("PoryAI", "FoulPlay")
        recv = await self.sendMessage(f"|/challenge {foulPlayUser}, {format}|")
        while True:
            recv = recv.split("|")
            if 'battle' in recv[0] and 'init' in recv[1]:
                return recv[0][1:].strip()
            
            recv = await self.socket.recv()

    # Runs the logic for the given battle tag
    # Communicates with the Interpreter for state,
    # and the agent for actions.
    async def manageBattle(self):


        while True:
            recv = await self.socket.recv()
            print(recv)
            msgs = recv.split("|")
            battleStarted = False
            
            if 'start\n' in recv and not battleStarted:
                battleStarted = True
                _, _, firstTurn = recv.partition("start\n")
                self.inter.updateTurnState(firstTurn.split("\n"), True)
                self.currentRewards = 0
                print(f"Current Rewards: {self.currentRewards}")

            
            # Requests for the user to do something. These should be sent to the interpreter.
            elif 'request' in msgs[1] and len(msgs[2]) > 2:
                requestOutput = json.loads(msgs[2])
                
                
                if not "wait" in requestOutput:
                    self.latestRequest = requestOutput
                    return False # Battle not done

            elif 't:' in msgs[2] and "Time left" not in msgs[2]:
                # Send turn content to interpreter here, then reset it.
                turnContent = recv.split("\n")[3:]
                print(f"TURN {turnContent[-1:]}"+self.user)
                print(turnContent)
                
                self.currentRewards = self.inter.countTurn(turnContent)
                self.state = self.inter.updateTurnState(turnContent, self.turnCount)
                
                
                self.turnCount += 1
                turnContent = []

            elif self.turnCount > 0 and (msgs[1] in ["switch", "move", "faint"] or msgs[1][1] == "-"):
                turnContent.append(recv)
            

            if '|win|' in recv: # Battle is over.
                time.sleep(3)
                return True


    async def run(self):
        await self.connectNoSecurity()
        while True:
            print("looking for battle "+self.user)
            battleTag = await self.challengeFoulPlay(self.format)
            await self.manageBattle(battleTag)
    
    async def restart(self):
        self.inter.resetState()
        self.turnCount = 0
        print(f"{self.user} looking for a battle...")
        battleTag = await self.challengeFoulPlay(self.format)
        self.currentTag = battleTag
        await self.manageBattle()
    
    def getState(self):
        return self.inter.state
        
    async def executeAction(self, action):
        print(f"{self.currentTag}|{action}|{self.latestRequest['rqid']}")
        await self.socket.send([f"{self.currentTag}|{action}|{self.latestRequest['rqid']}"])
        battleDone = await self.manageBattle()
        newState = self.inter.state
        rewards = self.currentRewards
        
        return newState, rewards, battleDone
        

    def getValidActions(self):
        valid_actions = []
        print(f"Latest request: {self.latestRequest}")
        
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
                    #if self.latestRequest["active"][0]["canTerastallize"]:
                        #valid_actions.append(f"/choose move {move+1} tera")
                        
            # Player can't switch if they're trapped.
            if not self.inter.state["playerSide"]["activeMon"]["condition"]["trapped"] or not self.latestRequest["active"][0]["trapped"]:
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
            
        print(valid_actions)
        return valid_actions