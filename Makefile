all:
	python ./test/t1_priors.py 

env:
	python -m venv env
	source env/bin/activate.fish
	pip install -r requirements.txt
	pip install -e .

test:
	python ./test/t1_priors.py

