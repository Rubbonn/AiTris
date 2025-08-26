from typing import Mapping, Any
import torch.nn as nn
import torch.optim as optim
import torch
import random
from collections import deque
from copy import deepcopy

class Agent(nn.Module):
	def __init__(self, epsilon: float = 0.01, batchSize: int = 128, memorySize: int = 20000, targetUpdateFreq: int = 1000):
		super().__init__()
		self._model: nn.Sequential = nn.Sequential(
			nn.Linear(9, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 9),
		)
		self.epsilon: float = epsilon
		self._batchSize: int = batchSize
		self._memory: deque = deque([], memorySize)
		self._optimizer: optim.Adam = optim.Adam(self.parameters())
		self._lossFn: nn.SmoothL1Loss = nn.SmoothL1Loss()

		self._target: nn.Sequential = deepcopy(self._model).requires_grad_(False)
		self._learnStep: int = 0
		self._updateTargetFreq: int = targetUpdateFreq
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self._model(x)
	
	def chooseMove(self, table: torch.Tensor) -> int:
		if torch.rand(1).item() < self.epsilon:
			while True:
				output = int(torch.randint(0, 9, (1,)).item())
				if table[output] == 0:
					return output
		
		with torch.no_grad():
			output = self.forward(table)
			for i, v in enumerate(output):
				output[i] = -1e9 if table[i] != 0 else v
			
			return int(torch.argmax(output).item())
	
	def learn(self) -> None:
		if len(self._memory) < self._batchSize:
			return
		
		self._learnStep += 1
		if self._learnStep % self._updateTargetFreq == 0:
			self._target.load_state_dict(self._model.state_dict())
		
		miniBatch = random.sample(self._memory, self._batchSize)
		states = torch.stack([s for s, a, r, ns, d in miniBatch])
		actions = torch.tensor([a for s, a, r, ns, d in miniBatch])
		rewards = torch.tensor([r for s, a, r, ns, d in miniBatch])
		nextStates = torch.stack([ns for s, a, r, ns, d in miniBatch])
		dones = torch.tensor([d for s, a, r, ns, d in miniBatch])

		invalidMask = (nextStates != 0)
		q_predicted = self.forward(states).masked_fill((states != 0), -1e9).gather(1, actions.unsqueeze(1)).squeeze(1)
		with torch.no_grad():
			q_next = self.forward(nextStates).masked_fill(invalidMask, -1e9).argmax(1).unsqueeze(1)
			q_next = self._target(nextStates).gather(1, q_next).squeeze(1)
			q_target = rewards + 0.99 * q_next * (1 - dones.float())

		loss = self._lossFn(q_predicted, q_target)
		self._optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
		self._optimizer.step()
	
	def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
		ret = super().load_state_dict(state_dict, strict, assign)
		self._target: nn.Sequential = deepcopy(self._model).requires_grad_(False)
		return ret
	
	def memorize(self, state: torch.Tensor, action: int, reward: float, nextState: torch.Tensor, done: bool) -> None:
		self._memory.append((state, action, reward, nextState, done))
	
	def editLastMemory(self, state: torch.Tensor|None = None, action: int|None = None, reward: float|None = None, nextState: torch.Tensor|None = None, done: bool|None = None) -> None:
		if len(self._memory) == 0:
			return
		prev: tuple = self._memory[-1]
		new: tuple = (state, action, reward, nextState, done)
		self._memory[-1] = tuple(v if new[i] is None else new[i] for i, v in enumerate(prev))