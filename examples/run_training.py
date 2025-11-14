
from franken_mva.solver import Solver, random_scramble, GOAL_STATE, heuristic

def main():
    solver = Solver(sims=200, c_puct=1.2)
    episodes = 5
    depth = 20
    for ep in range(1, episodes+1):
        s0 = random_scramble(depth)
        solved, steps, hs = solver.solve(s0, max_steps=300)
        print(f"[EP {ep:03d}] solved={solved} steps={steps} buffer={len(solver.rb.replay_buffer)} last_h={hs[-1] if hs else None}")
    print("Done.")

if __name__ == "__main__":
    main()
