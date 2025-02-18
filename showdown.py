import websockets
import requests
import json
import random
import interpreter

class Showdown:
    async def __init__(self, uri, user, password, websocket):
        self.uri = uri
        self.user = user
        self.password = password
        self.socket = await websockets.connect(websocket)
        self.inter = interpreter.Interpreter()

    async def connectToShowdown(self):
            # challstr and chall id represent the current user token.
            # we need to retrieve it from the showdown server. When not logged in,
            # the server will send the user their challstr and challid whcih we receive below.
        challid = 0
        challstr = ""

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


    async def sendMessage(self, message):
        await self.socket.send([message])
        return await self.socket.recv()
    
    # Join the queue for the specified format, returning the battle tag.
    async def joinQueue(self, format):
        recv = await self.sendMessage(f"|/join {format}|")

        while True:
            recv = recv.split("|")
            if 'battle' in recv[0]:
                return recv[0][1:].strip()
            recv = await self.socket.recv()

    # Runs the logic for the given battle tag
    # Communicates with the Interpreter for state,
    # and the agent for actions.
    async def manageBattle(self, battleTag):
        battle_started = False
        turnContent = []

        while True:
            recv = await self.socket.recv()
            msgs = recv.split("|")

            if 'start' in msgs[1]:
                battle_started = True

            # Identify the side of the player, for use in the interpreter.
            if 'player' in msgs[1] and self.user in msgs[3]:
                if 'p1' in msgs[2]:
                    self.player = 1
                else:
                    self.player = 2

            # Requests for the user to do something. These should be sent to the interpreter.
            if 'request' in msgs[1] and len(msgs) > 2:
                requestOutput = json.loads(msgs[2])
                if 'active' in requestOutput:
                    self.inter.updateStateActive(requestOutput)
                elif 'wait' in requestOutput or 'forceSwitch' in requestOutput:
                    self.inter.updateStateNoActive(requestOutput)
                elif 'teampreview' in requestOutput:
                    self.inter.updateStateTeamPreview(requestOutput)

                # Make decision down here


            if 'turn' in msgs[1]:
                # Send turn content to interpreter here, then reset it.
                print(f"TURN {msgs[2]}")
                print(turnContent)
                turnContent = []

            if battle_started:
                turnContent.append(recv)

            if 'win' in msgs[1]: # Battle is over.
                return
