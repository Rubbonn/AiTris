from typing import Any
from flask import Flask, render_template, request, abort
import torch
from classes.Agent import Agent

app: Flask = Flask(__name__)
agent: Agent = Agent()
agent.load_state_dict(torch.load('agent.pt'))

@app.route('/')
def homepage():
	return render_template('tris.html')

@app.route('/play', methods=['POST'])
def play():
	table: Any = request.json
	if table is not None:
		move: int = agent.chooseMove(torch.tensor(table, dtype=torch.float32))
	else:
		abort(400)
	print(move)
	return {'move': move}


if __name__ == '__main__':
	app.run(debug=False, threaded=True)