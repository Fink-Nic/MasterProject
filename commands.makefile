ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
ACTIVATE_PATH := madnis/.venv/bin/activate
SHELL := /bin/bash

venv:
	@echo "Enabling the momtrop+madnis virtual environment"
	source $(ROOT_DIR)$(ACTIVATE_PATH)

1dslice:
	@echo "Plotting a 1dslice of the latent space after training"
	python3 ./momtrop_madnis.py --plot 1dslice -ni 1000 -li 50 -fp outputs

channel_prog:
	@echo "Plotting the training progression of the discrete channel probabilities"
	python3 ./momtrop_madnis.py --plot channel_prog -ni 1000 -li 50 -fp outputs