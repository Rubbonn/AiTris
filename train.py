from classes.Game import Game
from classes.Agent import Agent
from time import sleep
from pathlib import Path
import torch
import random
import time

def agentMakeMove(state: torch.Tensor, currentPlayer: int) -> int:
	while True:
		action: int = agent.chooseMove(state)
		if game.move(action, currentPlayer):
			break
	return action

def penalizeLastMove() -> None:
	agent.editLastMemory(reward=-1, done=True)

def trainMatches(matches: int) -> tuple[int, int]:
	won: int = 0
	draws: int = 0
	for _ in range(matches):
		game.reset()
		done: bool = False
		pending: list = []
		
		currentPlayer: int = random.randint(1, 2)
		while not done:
			opposingPlayer = 3 - currentPlayer
			game.reverseTable()
			state: torch.Tensor = game.cloneTable()

			currentPlayerWinningMoves = game.getWinningMoves(currentPlayer)
			opposingPlayerWinningMoves = game.getWinningMoves(opposingPlayer)

			action: int = agentMakeMove(state, 1)

			if len(pending) > 0:
				game.reverseTable()
				nextState: torch.Tensor = game.cloneTable()
				game.reverseTable()
				agent.memorize(pending[0], pending[1], pending[2], nextState, pending[4])

			reward: float = -0.005 # Penalità per vincere in meno mosse possibili

			if len(currentPlayerWinningMoves) and action not in currentPlayerWinningMoves: # Poteva vincere ma non l'ha fatto
				reward -= 0.2

			elif len(opposingPlayerWinningMoves) and action not in opposingPlayerWinningMoves: # Poteva difendere ma non l'ha fatto
				reward -= 0.2

			elif len(opposingPlayerWinningMoves) and action in opposingPlayerWinningMoves: # Ha difeso
				reward += 0.2
			
			
			win: int = game.checkWin()
			if win >= 0:
				done = True
				if win == 0:
					reward = 0
					draws += 1
				elif win == currentPlayer:
					reward = 1
					penalizeLastMove()
					won += 1
				agent.memorize(state, action, reward, torch.zeros_like(state), done)
			
			pending = [state, action, reward, None, done]

			if debug:
				game.printTable()
				print('-' * 20)
				sleep(1)
			if done:
				if debug:
					print(f'Ha vinto il giocatore {win}')
					print('-' * 20)

			currentPlayer = opposingPlayer
		agent.learn()
	return (won, draws)

if __name__ == '__main__':
	debug: bool = False
	game: Game = Game()
	agent: Agent = Agent()

	if Path('agent.pt').exists():
		print('Caricamento del modello...')
		agent.load_state_dict(torch.load('agent.pt'))

	# Training loop
	print('Inizio dell\'allenamento...')
	epochs: int = 70
	matches: int = 1000
	decay_rate: float = .8
	agent.train()
	for epoch in range(epochs):
		startTime = time.perf_counter()
		agent.epsilon = 0.01 + (0.99 - 0.01) * (1 - (epoch / epochs) ** decay_rate)
		won, draws = trainMatches(matches)
		print(
			f"Epoca {epoch + 1}/{epochs} | "
			f'Epsilon: {agent.epsilon:.3f} | '
			f'Vinte: {won} | '
			f'Pareggiate: {draws} | '
			f'Durata: {time.perf_counter() - startTime:.2f} sec'
		)
		
	
	torch.save(agent.state_dict(), 'agent.pt')
	print('Allenamento completato!')