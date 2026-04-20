import asyncio
from miniclaw.bot import miniclaw
import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

async def agent_command():
    try:
        bot = miniclaw.from_config()
        print("miniclaw initialized. Type 'exit' to quit.")

        while True:
            user_input = input("User:")
            if user_input.lower() == "exit":
                break
            response = await bot.run(user_input)
            print(f"Miniclaw: {response}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    asyncio.run(agent_command())
