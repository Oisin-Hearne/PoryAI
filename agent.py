import random

class Agent:
    def __init__(self):
        pass

    #Select randomly for now        
    def getAction(self, state, tag):

        if 'active' in state:
            invalidMove = True

            # Pokémon can be locked into moves, so we need to check that the chosen move hasn't been disabled.
            while invalidMove:
                randomMove = random.choice(state["active"][0]["moves"])
                if not randomMove["disabled"]:
                    invalidMove = False
                    return tag+'|/choose move '+randomMove["id"]+"|"+str(state["rqid"])
            
        if 'forceSwitch' in state:
            invalidMon = True
            
            while invalidMon:
                randomMon = random.randint(0,5)
                # Don't send in a fainted mon or try to send in the current one.
                if(state["side"]["pokemon"][randomMon]["condition"] != "0 fnt") and (state["side"]["pokemon"][randomMon]["active"] == False):
                    invalidMon = False
                    return tag+'|/choose switch '+str(randomMon+1)+"|"+str(state["rqid"])