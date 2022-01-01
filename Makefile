.PHONY: database

up:
	docker-compose up --detach python

restart:
	docker-compose restart python

build:
	docker-compose up --detach --build

database: 
	docker-compose down
	docker image rm polytopia_database
	docker-compose up --build --detach database