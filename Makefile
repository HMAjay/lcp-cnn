.PHONY: install synth smoke-train smoke-detect demo test

install:
	pip install -e ".[dev]"

synth:
	python -m pulmoscan.data.synthetic --out ./data/synthetic

smoke-train: synth
	python -m pulmoscan.classification.train --data-root ./data/synthetic --epochs 1 --batch-size 2 --device cpu --output-dir ./artifacts/checkpoints

smoke-detect: synth
	python -m pulmoscan.detection.train --data-root ./data/synthetic --epochs 1 --batch-size 1 --device cpu --output-dir ./artifacts/detection_smoke
	python -m pulmoscan.detection.infer --data-root ./data/synthetic --checkpoint ./artifacts/detection_smoke/best.pt --split test --device cpu --output-dir ./artifacts/detections_smoke

demo:
	python -m pulmoscan.inference.demo_app --port 7865 --device cpu

test:
	pytest -q
