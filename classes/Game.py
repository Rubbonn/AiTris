import torch

class Game:
	def __init__(self):
		self.table = torch.zeros(9)
	
	def reset(self) -> None:
		self.table = torch.zeros(9)

	def move(self, position: int, player: int) -> bool:
		if position < 0 or position > 8 or player not in [1, 2]:
			return False
		if self.table[position] != 0:
			return False
		self.table[position] = 1
		return True
	
	def reverseTable(self) -> None:
		self.table = 3 - self.table  # Inverti i valori della tabella (1 diventa 2 e viceversa)
		self.table[self.table == 3] = 0
	
	def checkWin(self) -> int:
		win_conditions = [
			(0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
			(0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
			(0, 4, 8), (2, 4, 6)              # Diagonals
		]
		
		for a, b, c in win_conditions:
			if self.table[a] == self.table[b] == self.table[c] != 0:
				return int(self.table[a])
		
		if torch.all(self.table != 0):
			return 0
		
		return -1
	
	def printTable(self) -> None:
		symbols: dict = {1: 'X', 2: 'O', 0: '.'}
		boardStr: str = ""
		numbers: str = ''
		for i, cell in enumerate(self.table):
			boardStr += symbols[int(cell)] + " "
			numbers += f'{i+1} '
			if (i + 1) % 3 == 0:
				boardStr += f'\t{numbers}\n'
				numbers = ''
		print(boardStr)