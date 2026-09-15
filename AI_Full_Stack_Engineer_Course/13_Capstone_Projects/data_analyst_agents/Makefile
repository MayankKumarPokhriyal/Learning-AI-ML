# make data · make test · make lint · make eval-live · make eval-replay · make gate · make serve · make docker
PYTHON ?= python
export PYTHONPATH := src

.PHONY: data questions test lint eval-live eval-replay gate serve docker

data:  ## download the BIRD mini-dev subset (3 databases + descriptions + questions, ~37 MB) into data/bird_minidev
	$(PYTHON) -m data_analyst_agents.data

questions:  ## re-apply the selection rule and check it matches evals/question_set.json
	$(PYTHON) -m data_analyst_agents.questions

test:  ## unit + agent + API tests with the scripted FakeLLM (offline, no data needed)
	$(PYTHON) -m pytest -m "not live"

lint:
	uvx ruff@0.16.7 check .

eval-live:  ## real LLM: both systems, pass^k trials, writes evals/recordings/*.jsonl.gz
	$(PYTHON) -m data_analyst_agents.evaluate --mode live --record --out data/eval/live.json

eval-replay:  ## offline: the same evaluation replayed from evals/recordings (what CI runs)
	$(PYTHON) -m data_analyst_agents.evaluate --mode replay --out data/eval/replay.json

gate: eval-replay
	$(PYTHON) -m data_analyst_agents.gate --report data/eval/replay.json --baseline evals/baseline_summary.json

serve:
	$(PYTHON) -m uvicorn data_analyst_agents.api:app --port 8000

docker:
	docker build -t data-analyst-agents:1.0.0 .
