.PHONY: database
.PHONY: tests

up:
	docker-compose up --detach helper

restart:
	docker-compose restart helper

build:
	docker-compose up --detach --build

database: 
	docker-compose down
	docker image rm polytopia_database
	docker-compose up --build --detach database

tests:
	python3.9 -m pytest -v
