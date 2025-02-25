import showdown
import concurrent.futures
import asyncio

sd = showdown.Showdown("https://play.pokemonshowdown.com/action.php", "thisisatest12345", "password", "ws://localhost:8000/showdown/websocket", "gen9randombattle")
async def main():
    async with asyncio.TaskGroup() as tg:
        for i in range(500):
            sd = showdown.Showdown("https://play.pokemonshowdown.com/action.php", "PoryAI-"+str(i), "password", "ws://localhost:8000/showdown/websocket", "gen9randombattle")
            tg.create_task(sd.run())

    await asyncio.gather(*tg.tasks)

if __name__ == "__main__":
    asyncio.run(main())