.PHONY: run train scrape docker lint

run:
	python -m uvicorn api.main:app --reload

train:
	python -m training.train --trials 30

train-sample:
	python -m training.train --data data/sample/customer_booking_sample.csv --trials 10

scrape:
	python scripts/run_scraper.py

docker:
	docker-compose up --build

lint:
	ruff check .
