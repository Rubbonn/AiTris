import torch
import random

class Agent(torch.nn.Module):
	def __init__(self, epsilon: float = 0.1):
		super().__init__()
		self._model = torch.nn.Sequential(
			torch.nn.Linear(9, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 9),
		)
		self._epsilon = epsilon
		self._batchSize = 64
		self._memory = []
		self._optimizer = torch.optim.Adam(self.parameters())
		self._lossFn = torch.nn.MSELoss()
	
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
	
	def learn(self):
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
			q_next = self.forward(nextStates).max(1)[0]
			q_target = rewards + 0.99 * q_next * (1 - dones.float())
		
		loss = self._lossFn(q_predicted, q_target)
		self._optimizer.zero_grad()
		loss.backward()
		self._optimizer.step()