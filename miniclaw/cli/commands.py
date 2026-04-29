import asyncio
import sys

from miniclaw.bot import miniclaw

try:
    from loguru import logger
except ImportError:  # pragma: no cover - fallback for minimal environments
    import logging

    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    logger = logging.getLogger(__name__)
else:
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
