APP_NAME_API=sentiment-api
APP_NAME_MONITORING=sentiment-monitor
VOLUME_NAME=sentiment-logs
NETWORK_NAME=sentiment-net

.PHONY: build run clean

build:
	docker build -t $(APP_NAME_API) ./api
	docker build -t $(APP_NAME_MONITORING) ./monitoring

run:
	docker network create $(NETWORK_NAME) || true
	docker volume create $(VOLUME_NAME) || true

	docker run -d --rm --name api \
		--network $(NETWORK_NAME) \
		-v $(VOLUME_NAME):/logs \
		-p 8000:8000 $(APP_NAME_API)

	docker run -d --rm --name monitor \
		--network $(NETWORK_NAME) \
		-v $(VOLUME_NAME):/logs \
		-p 8501:8501 $(APP_NAME_MONITORING)

clean:
	docker stop api monitor || true
	docker network rm $(NETWORK_NAME) || true
	docker volume rm $(VOLUME_NAME) || true
	docker rmi $(APP_NAME_API) $(APP_NAME_MONITORING) || true