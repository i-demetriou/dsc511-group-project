.PHONY: all init
SHELL=/bin/bash

all: init

init:
	virtualenv venv
	. venv/bin/activate && pip install -r requirements.txt
	git submodule update --recursive

nominatim/build:
	cd n7m && docker-compose build &&\
		docker-compose run feed download --wiki --grid\
		docker run -v ${PWD}/data:/tileset openmaptiles/openmaptiles-tools download-osm monaco
		docker-compose run feed setup

nominatim/run:
	cd n7m && docker-compose up -d
