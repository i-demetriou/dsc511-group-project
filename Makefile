.PHONY: all init
SHELL=/bin/bash

all: init

init:
	virtualenv venv
	. venv/bin/activate && pip install -r requirements.txt
	git submodule update --recursive

execute: execute/notebook
execute/notebook:
	jupyter nbconvert --to notebook\
		--execute --ExecutePreprocessor.timeout=None --ExecutePreprocessor.allow_errors=False\
		--output ./out.ipynb ./dsc511.project.mtsilidou.anikodimou.idemetriou.ipynb

execute/python:
	jupyter nbconvert --to notebook\
		--execute --ExecutePreprocessor.timeout=None --ExecutePreprocessor.allow_errors=False\
		--output ./out.ipynb ./dsc511.project.mtsilidou.anikodimou.idemetriou.py

html:
	jupyter nbconvert --to html ./dsc511.project.mtsilidou.anikodimou.idemetriou.ipynb

nominatim/build:
	cd n7m && docker-compose build &&\
		docker-compose run feed download --wiki --grid\
		docker run -v ${PWD}/data:/tileset openmaptiles/openmaptiles-tools download-osm monaco
		docker-compose run feed setup

nominatim/run:
	cd n7m && docker-compose up -d
