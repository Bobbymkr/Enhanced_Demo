"""
Multi-seed simulation runner
Runs basic_simulation.py with different random seeds
Collects results across runs
"""
import subprocess
import os
import sys

def run_with_seeds():
    seeds = [42, 123, 256, 789, 1001]
    results = {}
    
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Running simulation with SEED = {seed}")
        print(f"{'='*70}\n")
        
        env = os.environ.copy()
        env["SIM_SEED"] = str(seed)
        
        try:
            result = subprocess.run(
                [sys.executable, "tests/basic_simulation.py"],
                env=env,
                capture_output=False
            )
            if result.returncode != 0:
                print(f"⚠️  Seed {seed} may have had issues")
        except Exception as e:
            print(f"❌ Error running seed {seed}: {e}")
    
    print(f"\n{'='*70}")
    print("All seeds completed!")
    print(f"{'='*70}")

if __name__ == "__main__":
    run_with_seeds()