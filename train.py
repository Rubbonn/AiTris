from classes.Game import Game
from classes.Agent import Agent
from time import sleep
from pathlib import Path
from torch import save, load, Tensor, set_default_device
import torch.cuda as cuda

if __name__ == '__main__':
	debug: bool = False
	set_default_device('cuda' if cuda.is_available() else 'cpu')
	game: Game = Game()
	agent: Agent = Agent(epsilon=0.99)

	if Path('agent.pt').exists():
		print('Caricamento del modello...')
		agent.load_state_dict(load('agent.pt'))

	# Training loop
	print('Inizio dell\'allenamento...')
	epochs = 20
	matches = 1_000
	episodes = epochs
	epsilonDecay = max((episodes - 2.5)/episodes, 0)
	agent.train()
	for epoch in range(epochs):
		#agent.reset()
		agent.epsilon = (agent.epsilon - 0.01) * epsilonDecay + 0.01
		for match in range(matches):
			game.reset()
			done: bool = False
			if debug:
				print(agent.epsilon)
			
			while not done:
				state: Tensor = game.table.clone().detach()
				while True:
					action: int = agent.chooseMove(state)
					if game.move(action):
						break
				nextState: Tensor = game.table.clone().detach()
				
				reward: float = -0.01 # Penalty for each move
				win: int = game.checkWin()
				if win >= 0:
					reward = 1 if win == 1 else 0
					done = True
				
				agent._memory.append((state, action, reward, nextState, done))
				if win == 1 and len(agent._memory) >= 2:
					state, action, _, nextState, _ = agent._memory[-2]
					agent._memory[-2] = (state, action, -1, nextState, True)

				if debug:
					game.printTable()
					print('-' * 20)
					sleep(1)
				if done:
					if debug:
						print(f'Ha vinto il giocatore {win}')
						print('-' * 20)
					break

				game.reverseTable()  # Reverse the table for the opponent's turn
				state: Tensor = game.table.clone().detach()
				while True:
					action = agent.chooseMove(state)
					if game.move(action):
						break
				nextState: Tensor = game.table.clone().detach()
				game.reverseTable()

				reward: float = -0.01 # Penalty for each move
				win: int = game.checkWin()
				if win >= 0:
					reward: float = 1 if win == 2 else 0
					done = True

				agent._memory.append((state, action, reward, nextState, done))
				if win == 2 and len(agent._memory) >= 2:
					state, action, _, nextState, _ = agent._memory[-2]
					agent._memory[-2] = (state, action, -1, nextState, True)

				if debug:
					game.printTable()
					print('-' * 20)
					sleep(1)
				if done:
					if debug:
						print(f'Ha vinto il giocatore {win}')
						print('-' * 20)
					break
			agent.learn()
	
	save(agent.state_dict(), 'agent.pt')
	print('Allenamento completato!')