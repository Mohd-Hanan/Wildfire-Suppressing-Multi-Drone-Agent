import argparse
import json
import os
import torch
from collections import OrderedDict
from wildfire.environment.wildfire_env import WildfireEnv
from wildfire.rl.evaluate import evaluate_policy
from wildfire.agents.random_agent import RandomAgent
from wildfire.agents.heuristic import HeuristicAgent
from wildfire.agents.no_suppression import NoSuppressionAgent
from wildfire.agents.suppression_test_agent import ControlledSuppressionAgent

def main():
    parser = argparse.ArgumentParser(description="Test suppression mechanical effect")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=100, help="Starting random seed")
    parser.add_argument("--output", type=str, default="suppression_effect_results.json")
    args = parser.parse_args()
    
    seeds = [args.seed + i for i in range(args.episodes)]
    env = WildfireEnv()
    device = torch.device("cpu")
    
    agents = {
        "no_suppression": NoSuppressionAgent(),
        "random": RandomAgent(seed=args.seed),
        "heuristic": HeuristicAgent(),
        "controlled_suppression": ControlledSuppressionAgent()
    }
    
    results = {}
    
    print(f"Running Controlled Suppression Experiment over {args.episodes} seeds...")
    
    for name, agent in agents.items():
        print(f"Evaluating {name}...")
        res = evaluate_policy(env, seeds, agent, device, deterministic=True)
        results[name] = res
        
    print("\n--- BURNED CELLS COMPARISON ---")
    print(f"{'Seed':<6} | {'NoSupp':<8} | {'Random':<8} | {'Heuristic':<9} | {'ControlSupp':<11}")
    
    per_seed_results = []
    
    for i, s in enumerate(seeds):
        no_supp_b = results["no_suppression"]["episodes"][i]["burned_cells"]
        rand_b = results["random"]["episodes"][i]["burned_cells"]
        heur_b = results["heuristic"]["episodes"][i]["burned_cells"]
        cont_b = results["controlled_suppression"]["episodes"][i]["burned_cells"]
        
        no_supp_e = results["no_suppression"]["episodes"][i]["extinguished"]
        cont_e = results["controlled_suppression"]["episodes"][i]["extinguished"]
        
        print(f"{s:<6} | {no_supp_b:<8} | {rand_b:<8} | {heur_b:<9} | {cont_b:<11}")
        
        per_seed_results.append({
            "seed": s,
            "no_suppression": results["no_suppression"]["episodes"][i],
            "random": results["random"]["episodes"][i],
            "heuristic": results["heuristic"]["episodes"][i],
            "controlled_suppression": results["controlled_suppression"]["episodes"][i]
        })
        
    print("\n--- SUCCESSFUL WATER DROPS ---")
    print(f"{'NoSupp':<8} | {'Random':<8} | {'Heuristic':<9} | {'ControlSupp':<11}")
    print(f"{results['no_suppression']['diagnostics']['successful_water']['mean']:<8} | {results['random']['diagnostics']['successful_water']['mean']:<8} | {results['heuristic']['diagnostics']['successful_water']['mean']:<9} | {results['controlled_suppression']['diagnostics']['successful_water']['mean']:<11}")
    
    out_dict = {
        "experiment": "Controlled Suppression Effect",
        "seeds": seeds,
        "agents": list(agents.keys()),
        "per_seed": per_seed_results,
        "summary": {name: res for name, res in results.items() if "episodes" not in res} # Will fix this later in a cleaner way
    }
    
    # Strip episodes from summary correctly
    summary_clean = {}
    for name, res in results.items():
        summary_clean[name] = {k: v for k, v in res.items() if k != "episodes"}
    out_dict["summary"] = summary_clean
    
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out_dict, f, indent=4)
        print(f"\nSaved results to {args.output}")

if __name__ == "__main__":
    main()
