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
        self.agent = agent.Agent()
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
    async def manageBattle(self, battleTag):
        self.inter.resetState()
        battle_started = False
        turnContent = []
        turnCount = 0

        while True:
            recv = await self.socket.recv()
            msgs = recv.split("|")
            
            if 'start\n' in recv and not battle_started: # Start of battle
                battle_started = True
                _, _, firstTurn = recv.partition("start\n")
                self.inter.updateTurnState(firstTurn.split("\n"), turnCount)
                turnCount += 1
            
            # Identify the side of the player, for use in the interpreter.
            elif 'player' in msgs[1] and self.user in msgs[3]:
                if 'p1' in msgs[2]:
                    self.player = 1
                else:
                    self.player = 2

            # Requests for the user to do something. These should be sent to the interpreter.
            elif 'request' in msgs[1] and len(msgs[2]) > 2:
                requestOutput = json.loads(msgs[2])
                if 'active' in requestOutput:
                    self.inter.updateStateActive(requestOutput)
                elif 'wait' in requestOutput or 'forceSwitch' in requestOutput:
                    self.inter.updateStateNoActive(requestOutput)
                elif 'teampreview' in requestOutput:
                    self.inter.updateStateTeamPreview(requestOutput)

                # Make decision down here
                if not "wait" in requestOutput:
                    await self.socket.send([self.agent.getAction(requestOutput, battleTag)])

            elif 't:' in msgs[2] and "Time left" not in msgs[2]:
                # Send turn content to interpreter here, then reset it.
                turnContent = recv.split("\n")[3:]
                print(f"TURN {turnContent[-1:]}"+self.user)
                print(turnContent)
                self.inter.updateTurnState(turnContent, turnCount)
                turnCount += 1
                turnContent = []

            elif turnCount > 0 and (msgs[1] in ["switch", "move", "faint"] or msgs[1][1] == "-"):
                turnContent.append(recv)
            

            if '|win|' in recv: # Battle is over.
                print("battle over"+self.user)
                time.sleep(3)
                break


    async def run(self):
        await self.connectNoSecurity()
        while True:
            print("looking for battle "+self.user)
            battleTag = await self.challengeFoulPlay(self.format)
            await self.manageBattle(battleTag)