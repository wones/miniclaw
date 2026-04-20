from typing import List, Callable, Any,Awaitable
import asyncio

class MessageBus:
    def __init__(self):
        self._inbound_queue = asyncio.Queue()
        self._outbound_queue = asyncio.Queue()

    async def publish_inbound(self,message):
        await self._inbound_queue.put(message)
    
    async def consum_inbound(self):
        return await self._inbound_queue.get()

    async def publish_outbound(self,message):
        await self._outbound_queue.put(message)
    
    async def consum_outbound(self):
        return await self._outbound_queue.get()