.PHONY: install synth smoke-train demo test

install:
	pip install -e ".[dev]"

synth:
	python -m pulmoscan.data.synthetic --out ./data/synthetic

smoke-train: synth
	python -m pulmoscan.classification.train --data-root ./data/synthetic --epochs 1 --batch-size 2 --device cpu --output-dir ./artifacts/checkpoints

demo:
	python -m pulmoscan.inference.demo_app --port 7865 --device cpu

test:
	pytest -q
