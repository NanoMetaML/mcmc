all:
	python ./test/t1_priors.py 

env:
	python -m venv env
	source env/bin/activate.fish
	pip install -r requirements.txt
	pip install -e .

test1:
	python ./test/t1_priors.py

test2:
	python ./test/t2_uniform_MCMC.py

test3:
	python ./test/t3_uniform_MCMC_process.py

test4:
	python ./test/t4_uniform_MCMC_spectral_gap.py

test5:
	python ./test/t5_uniform_MCMC_plot.py
