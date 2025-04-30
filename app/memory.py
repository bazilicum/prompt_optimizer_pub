import heapq
import itertools

class TopNMemory:
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.entries = []
        self.counter = itertools.count()

    def add(self, prompt: str, prompt_output: list, score: int, input_text: str):
        entry_id = next(self.counter)
        heapq.heappush(self.entries, (score, entry_id, {
            "prompt": prompt,
            "outputs": prompt_output,
            "score": score,
            "input": input_text
        }))
        if len(self.entries) > self.capacity:
            heapq.heappop(self.entries)

    def get_top(self):
        return [entry[2] for entry in sorted(self.entries, key=lambda x: -x[0])] 
    
    def reset(self):
        self.entries = []