from typing import List, Any
import asyncio

class BatchProcessor:
    def __init__(self, batch_size: int = 10, max_concurrent: int = 5):
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._queue = []
        self._results = {}
        
    async def add(self, items: List[Any]) -> List[Any]:
        """Efficiently batch process items"""
        if len(items) <= self.batch_size:
            async with self.semaphore:
                return await self._process_batch(items)
                
        batches = [items[i:i + self.batch_size] 
                  for i in range(0, len(items), self.batch_size)]
                  
        results = await asyncio.gather(*[
            self._process_batch(batch) for batch in batches
        ])
        
        return [item for batch in results for item in batch] 