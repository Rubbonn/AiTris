from classes.Game import Game
from classes.Agent import Agent
from time import sleep
from pathlib import Path
from torch import save, load, Tensor, zeros_like, exp, tensor
import random

def agentMakeMove(state: Tensor, currentPlayer: int) -> int:
	while True:
		action: int = agent.chooseMove(state)
		if game.move(action, currentPlayer):
			break
	return action

def penalizeLastMove() -> None:
	if len(agent._memory) > 0:
		state, action, _, nextState, _ = agent._memory[-1]
		agent._memory[-1] = (state, action, -1, nextState, True)

def trainMatches(matches: int = 1_000) -> None:
	for _ in range(matches):
		game.reset()
		done: bool = False
		pending: list = []
		
		currentPlayer: int = random.randint(1, 2)
		while not done:
			game.reverseTable()
			state: Tensor = game.table.clone().detach()
			action: int = agentMakeMove(state, 1)

			if len(pending) > 0:
				game.reverseTable()
				nextState: Tensor = game.table.clone().detach()
				game.reverseTable()
				agent._memory.append((pending[0], pending[1], pending[2], nextState, pending[4]))
			
			reward: float = -0.005 # Penalty for each move
			win: int = game.checkWin()
			if win >= 0:
				done = True
				if win == 0:
					reward = 0
				elif win == currentPlayer:
					reward = 1
					penalizeLastMove()
				agent._memory.append((state, action, reward, zeros_like(state), done))
			
			pending = [state, action, reward, None, done]

			if debug:
				game.printTable()
				print('-' * 20)
				sleep(1)
			if done:
				if debug:
					print(f'Ha vinto il giocatore {win}')
					print('-' * 20)
				break

			currentPlayer = 3 - currentPlayer
		agent.learn()

if __name__ == '__main__':
	debug: bool = False
	game: Game = Game()
	agent: Agent = Agent(epsilon=0.9)

	if Path('agent.pt').exists():
		print('Caricamento del modello...')
		agent.load_state_dict(load('agent.pt'))

	# Training loop
	print('Inizio dell\'allenamento...')
	epochs = 50
	matches = 1_000
	episodes = epochs * matches
	agent.train()
	for epoch in range(epochs):
		agent.epsilon = 0.01 + (0.9 - 0.01) * exp(tensor(-(epoch / 40))).item()
		trainMatches(matches)
		
	
	save(agent.state_dict(), 'agent.pt')
	print('Allenamento completato!')