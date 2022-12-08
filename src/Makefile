include .env
export

.PHONY: database
.PHONY: tests
.PHONY: map_analysis
.PHONY: map_patching

up:
	docker-compose up --detach slash_bot

restart:
	docker-compose restart slash_bot map_patching map_analysis

bot:
	docker-compose up --detach --build slash_bot

map_patching:
	docker-compose up --build --detach map_patching

map_analysis:
	docker-compose up --build --detach map_analysis

rabbitmq:
	docker-compose up --build --detach rabbitmq

database:
	docker-compose up --build --detach database

tests:
	python3.10 -m pytest -v
