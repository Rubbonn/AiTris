from classes.Game import Game
from classes.Agent import Agent
from pathlib import Path
from sys import exit
from torch import load, no_grad, Tensor
import random

def playerMove(table: Tensor) -> int:
	while True:
		try:
			choice: int = int(input('Scegli la casella da segnare (1-9):'))
			if choice < 1 or choice > 9:
				print('Inserisci un numero tra 1 e 9 compresi.')
				continue
			break
		except:
			print('Inserisci un numero valido.')
			continue
	return choice - 1


def agentMove(table: Tensor) -> int:
	return agent.chooseMove(table)

if __name__ == '__main__':
	game: Game = Game()
	agent: Agent = Agent(epsilon=0.1)

	if not Path('agent.pt').exists():
		print('Nessun modello trovato, lancia prima train.py per allenarne uno nuovo.')
		exit(1)

	print('Caricamento del modello...')
	agent.load_state_dict(load('agent.pt'))

	print('Inizio del gioco!')
	newGame: bool = True
	agent.eval()
	with no_grad():
		while newGame:
			game.reset()
			done: bool = False
			startingPlayer: int = random.randint(1, 2)

			while not done:
				game.printTable()
				if startingPlayer == 2:
					game.reverseTable()
				while True:
					move: int = playerMove(game.table) if startingPlayer == 1 else agentMove(game.table)
					if game.move(move):
						break
					elif startingPlayer == 1:
						print('Mossa non valida, scegline un altra.')
				if startingPlayer == 2:
					game.reverseTable()

				win: int = game.checkWin()
				if win >= 0:
					game.printTable()
					print(f'Ha vinto il giocatore {win}!') if win != 0 else print('Pareggio!')
					break

				startingPlayer = 3 - startingPlayer
			
			while True:
				choice: str = input('Vuoi fare una nuova partita? (Y/n):').lower()
				if choice != 'y' and choice != 'n' and choice != '':
					print('Scegli tra y e n')
					continue
				newGame = choice == 'y' or choice == ''
				break
	exit(0)