"""Example training script for the hybrid MVA solver.

Usage::

    python -m examples.run_training
    python -m examples.run_training --sims 400 --episodes 20 --depth 25

Set LOG_LEVEL=DEBUG for detailed per-step output.
"""

import argparse
import logging
import os
import sys

# Allow running as a script from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from franken_mva.config import SolverConfig
from franken_mva.solver import Solver, random_scramble

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the hybrid MVA solver.")
    p.add_argument("--sims", type=int, default=200, help="MCTS simulations per step")
    p.add_argument("--c-puct", type=float, default=1.2, help="PUCT exploration constant")
    p.add_argument("--episodes", type=int, default=5, help="Number of training episodes")
    p.add_argument("--depth", type=int, default=20, help="Scramble depth")
    p.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = SolverConfig(sims=args.sims, c_puct=args.c_puct)
    solver = Solver(config=cfg)

    logger.info(
        "Starting training — episodes=%d  depth=%d  sims=%d  c_puct=%.2f",
        args.episodes,
        args.depth,
        args.sims,
        args.c_puct,
    )

    for ep in range(1, args.episodes + 1):
        s0 = random_scramble(args.depth)
        solved, steps, hs = solver.solve(s0, max_steps=args.max_steps)
        logger.info(
            "[EP %03d] solved=%-5s  steps=%3d  buffer=%6d  last_h=%s",
            ep,
            solved,
            steps,
            len(solver.rb.replay_buffer),
            hs[-1] if hs else None,
        )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
