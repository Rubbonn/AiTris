from typing import Mapping, Any
import torch.nn as nn
import torch.optim as optim
import torch
import random
from collections import deque
from copy import deepcopy

class Agent(nn.Module):
	def __init__(self, epsilon: float = 0.1, batchSize: int = 64):
		super().__init__()
		self._model: nn.Sequential = nn.Sequential(
			nn.Linear(9, 128),
			nn.ReLU(),
			nn.Linear(128, 9),
		)
		self._epsilon: float = epsilon
		self._batchSize: int = batchSize
		self._memory: deque = deque([], 5_000)
		self._optimizer: optim.Adam = optim.Adam(self.parameters())
		self._lossFn: nn.SmoothL1Loss = nn.SmoothL1Loss()

		self._target: nn.Sequential = deepcopy(self._model).requires_grad_(False)
		self._learnStep: int = 0
		self._updateTargetFreq: int = 1000
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self._model(x)
	
	def chooseMove(self, table: torch.Tensor) -> int:
		if torch.rand(1).item() < self._epsilon:
			while True:
				output = int(torch.randint(0, 9, (1,)).item())
				if table[output] == 0:
					return output
		
		with torch.no_grad():
			output = self.forward(table)
			for i, v in enumerate(output):
				output[i] = -torch.inf if table[i] != 0 else v
			
			return int(torch.argmax(output).item())
	
	def learn(self) -> None:
		if len(self._memory) < self._batchSize:
			return
		
		miniBatch = random.sample(self._memory, self._batchSize)
		states = torch.stack([s for s, a, r, ns, d in miniBatch])
		actions = torch.tensor([a for s, a, r, ns, d in miniBatch])
		rewards = torch.tensor([r for s, a, r, ns, d in miniBatch])
		nextStates = torch.stack([ns for s, a, r, ns, d in miniBatch])
		dones = torch.tensor([d for s, a, r, ns, d in miniBatch])

		q_predicted = self.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1)
		with torch.no_grad():
			q_next = self._target(nextStates).max(1)[0]
			q_target = rewards + 0.9 * q_next * (1 - dones.float())
		
		loss = self._lossFn(q_predicted, q_target)
		self._optimizer.zero_grad()
		loss.backward()
		self._optimizer.step()

		self._learnStep += 1
		if self._learnStep % self._updateTargetFreq == 0:
			self._target.load_state_dict(self._model.state_dict())

	def setEpsilon(self, epsilon: float) -> None:
		self._epsilon = epsilon
	
	def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
		ret = super().load_state_dict(state_dict, strict, assign)
		self._target: nn.Sequential = deepcopy(self._model).requires_grad_(False)
		return ret