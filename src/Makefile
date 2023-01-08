include .env
include .env.dev
export

.PHONY: database
.PHONY: tests
.PHONY: map_analysis
.PHONY: map_patching
.PHONY: registry

up:
	docker-compose up --detach slash_bot

restart:
	docker-compose restart slash_bot map_patching map_analysis header_footer_recognition

bot: .docker-build-${BASE_VERSION} registry
	docker-compose up --detach --build slash_bot

turn_recognition: .docker-build-${BASE_VERSION} registry
	docker-compose up --detach --build header_footer_recognition

map_patching: .docker-build-${BASE_VERSION} registry
	docker-compose up --build --detach map_patching

map_analysis: .docker-build-${BASE_VERSION} registry
	docker-compose up --build --detach map_analysis

rabbitmq:
	docker-compose up --build --detach rabbitmq

database:
	docker-compose up --build --detach database

tests:
	python3.9 -m pytest -v

registry:
	docker-compose up -d registry

.docker-build-%: registry
	docker build -t polyhelper-base:$* ./base
	docker tag polyhelper-base:$* localhost:5000/polyhelper-base:$*
	docker push localhost:5000/polyhelper-base:$*
	docker pull localhost:5000/polyhelper-base:$*
	docker tag localhost:5000/polyhelper-base:$* localhost:5000/polyhelper-base:latest
	docker push localhost:5000/polyhelper-base:latest
	@echo "" > .docker-build-$*
