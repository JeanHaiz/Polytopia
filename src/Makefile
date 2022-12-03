.PHONY: database
.PHONY: tests

up:
	docker-compose up --detach slash_bot

restart:
	docker-compose restart slash_bot map_patching map_analysis

build:
	docker-compose up --detach --build

database: 
	docker-compose down
	docker image rm polytopia_database
	docker-compose up --build --detach database

tests:
	python3.9 -m pytest -v
