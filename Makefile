.PHONY: instalar dados features treino relatorio pipeline testes api docker

instalar:
	pip install -e ".[treino,dev]"

dados:
	python -m atrasos.download
	python -m atrasos.limpeza

features:
	python -m atrasos.preparar

treino:
	python -m atrasos.treino

relatorio:
	python -m atrasos.relatorio

pipeline: dados features treino relatorio

testes:
	ruff check src tests
	pytest -q

api:
	uvicorn atrasos.api:app --reload

docker:
	docker build -t atrasos-api .
	docker run --rm -p 8000:8000 atrasos-api
