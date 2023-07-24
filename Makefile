AUTHOR=Alexander Nano
QUEUE=brk1080
PROJECT=QuEra

MODULE=test_module
REPO=modules

RUN_SCRIPT_PATH=./run.py
EXPERIMENT_PATH=./tests/experiments/
EXPERIMENT_NAME=load_data

module:
	@echo "Building module" ./$(REPO)"/"$(MODULE_NAME) 
	@mkdir -p ./$(REPO)/$(MODULE_NAME)
	@touch ./$(REPO)/$(MODULE_NAME)/__init__.py
	@echo "# Module $(MODULE)\n# Author: $(AUTHOR_NAME)\n# Date: \nfrom .module import load\n" >> ./$(REPO_NAME)/$(MODULE_NAME)/__init__.py
	@echo "# Module $(MODULE)\n# Author: $(AUTHOR_NAME)\n# Date: \ndef load(**kwargs):\n	return NotImplementedError('Load not implemented for module $(MODULE_NAME)')\n" >> ./$(REPO_NAME)/$(MODULE_NAME)/module.py

run_clearml: 
	clearml-task --script $(RUN_SCRIPT_PATH) --project QuEra --name $(EXPERIMENT_NAME)  --branch $(BRANCH) --requirements requirements.txt --args exp_path=$(EXP_PATH) exp_name=$(EXP_NAME) --queue $(QUEUE_NAME)

run_local:
	python $(RUN_SCRIPT_PATH) exp_path=$(EXPERIMENT_PATH) exp_name=$(EXP)