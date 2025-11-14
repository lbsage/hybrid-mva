
# Franken-MVA Repo

- 15-puzzle core with PUCT MCTS
- Optional PyTorch HRM (A2C-style trainer)
- Replay buffer (prioritized optional)
- Live reflexion engine (rules + synthetic supervision)

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional: pip install torch
python examples/run_training.py
```
