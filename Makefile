# Create and configure environment for Ligo, build the Jupyterbook, and cleaning folders

.ONESHELL:
SHELL = /bin/bash

## create_environment : Create and configure environment
.PHONY : env
env: 
	source /srv/conda/etc/profile.d/conda.sh
	conda env create -f environment.yml
	conda activate ligo
	conda install ipykernel
	python -m ipykernel install --user --name makeLigo --display-name "IPython - ligo"

## html : Build the Jupyterbook
.PHONY : html
html:
	jupyter-book build .

## clean : Remove audio, figure, and _build folder contents 
.PHONY : clean
clean : 
	rm -f audio/*
	rm -f figures/*
	rm -r _build/*
