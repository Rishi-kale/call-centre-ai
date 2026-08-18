.PHONY: install demo batch run clean

# Install Python deps (assumes ffmpeg is already on PATH)
install:
	pip install -r requirements.txt

# Populate the DB with synthetic calls + playable audio — no GPU or API key needed.
# Fastest way to see the whole product working.
demo:
	python -m backend.seed_demo

# Transcribe + analyse every (audio, metadata) pair under data/.
# Put the real recordings in data/audio/ and data/metadata/ first.
#   make batch                 # all calls
#   make batch LIMIT=20        # first 20 (quick check)
#   make batch BACKEND=heuristic   # skip the LLM
LIMIT ?=
BACKEND ?= auto
batch:
	python -m backend.batch $(if $(LIMIT),--limit $(LIMIT),) --backend $(BACKEND)

# Start the API + dashboard at http://127.0.0.1:8000
run:
	uvicorn backend.api:app --host 0.0.0.0 --port 8000

clean:
	rm -f callradar.db
	rm -f data/audio/demo*.mp3
