from classes.Game import Game
from classes.Agent import Agent
from time import sleep
from pathlib import Path
from torch import save, load

if __name__ == '__main__':
	debug: bool = False
	game: Game = Game()
	agent: Agent = Agent(epsilon=0.1)

	if Path('agent.pt').exists():
		print('Caricamento del modello...')
		agent.load_state_dict(load('agent.pt'))

	# Training loop
	print('Inizio dell\'allenamento...')
	for epoch in range(10_000):
		game.reset()
		done: bool = False
		
		while not done:
			state = game.table.clone()
			while True:
				action = agent.chooseMove(state)
				if game.move(action):
					break
			
			reward = -0.01 # Penalty for each move
			win: int = game.checkWin()
			if win >= 0:
				reward = 1 if win == 1 else 0
				done = True
			
			nextState = game.table.clone()
			agent._memory.append((state, action, reward, nextState, done))
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
			while True:
				action = agent.chooseMove(state)
				if game.move(action):
					break
			game.reverseTable()

			win: int = game.checkWin()
			if win >= 0:
				state, action, reward, nextState, done = agent._memory[-1]
				reward = -1 if win == 2 else 0
				done = True
				agent._memory[-1] = (state, action, reward, nextState, done)

			if debug:
				game.printTable()
				print('-' * 20)
				sleep(1)
			if done and debug:
				print(f'Ha vinto il giocatore {win}')
				print('-' * 20)
		
		agent.learn()
	
	save(agent.state_dict(), 'agent.pt')
	print('Allenamento completato!')