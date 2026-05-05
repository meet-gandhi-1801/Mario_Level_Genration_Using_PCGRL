"""
FULL PIPELINE MASTER SCRIPT
============================
1. Generate 5 levels with GAN (diverse latent sampling)
2. Repair each with GA (50 iterations)
3. Find the hardest level (lowest A* completion)
4. Apply difficulty modifier
5. Re-evaluate with Mario AI Framework agent
6. Print final results
"""

import subprocess
import os
import shutil
import glob
import random
import sys

# ── Config ────────────────────────────────────────────────────
MARIO_FRAMEWORK  = r"C:\Users\DELL\Mario-AI-Framework"
MARIO_REPAIRER   = r"C:\Users\DELL\MarioLevelRepairer"
GAN_LEVEL_DST    = os.path.join(MARIO_FRAMEWORK, "levels", "generated", "gan_level.txt")
NUM_LEVELS       = 5       # how many levels to generate and test
DIFFICULTY       = 20      # how hard to make the final level

# ── Helpers ───────────────────────────────────────────────────

def get_last_iteration():
    files = glob.glob(os.path.join(MARIO_REPAIRER, "GA", "result", "txt", "iteration*.txt"))
    if not files:
        return None
    files.sort(key=lambda x: int(x.split("iteration")[1].replace(".txt", "")))
    return files[-1]


def run_agent():
    result = subprocess.run(
        ["java", "-cp", "classes", "PlayGANLevel"],
        cwd=MARIO_FRAMEWORK,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    for line in result.stdout.split('\n'):
        if 'Percentage Completion' in line:
            try:
                pct = float(line.split('Percentage Completion: ')[1].split()[0])
                return pct
            except:
                pass
    return 1.0


def generate_and_repair():
    print("  Generating level with GAN...")
    subprocess.run(
        ["python", "LevelGenerator/GAN/generate_level.py"],
        cwd=MARIO_REPAIRER
    )
    print("  Clearing old GA results...")
    subprocess.run(["python", "GA/clear.py"], cwd=MARIO_REPAIRER)
    print("  Repairing with GA (50 iterations)...")
    subprocess.run(["python", "GA/run.py"], cwd=MARIO_REPAIRER)


def copy_to_framework(src):
    os.makedirs(os.path.dirname(GAN_LEVEL_DST), exist_ok=True)
    shutil.copy(src, GAN_LEVEL_DST)


def make_harder(input_path, output_path, difficulty=1):
    """Add gaps and enemies to increase difficulty."""
    with open(input_path) as f:
        lines = f.readlines()

    level = [list(line.rstrip('\n')) for line in lines]
    h = len(level)
    w = len(level[0]) if level else 0

    # Add gaps in ground
    num_gaps = difficulty * 3
    gaps_added = 0
    attempts = 0
    while gaps_added < num_gaps and attempts < 1000:
        attempts += 1
        gap_start = random.randint(10, max(11, w - 15))
        gap_size = random.randint(3, 5)
        valid = True
        for g in range(gap_size):
            j = gap_start + g
            if j >= w:
                valid = False
                break
            if level[h-1][j] != 'X':
                valid = False
                break
        if valid:
            for g in range(gap_size):
                j = gap_start + g
                for row in range(h - 3, h):
                    if level[row][j] in ['X', 'S', '#']:
                        level[row][j] = '-'
            gaps_added += 1

    # Add enemies
    num_enemies = difficulty * 4
    enemies_added = 0
    attempts = 0
    while enemies_added < num_enemies and attempts < 1000:
        attempts += 1
        j = random.randint(5, w - 5)
        for i in range(h - 2, 0, -1):
            if level[i][j] in ['X', 'S'] and level[i-1][j] == '-':
                level[i-1][j] = 'E'
                enemies_added += 1
                break

    with open(output_path, 'w') as f:
        for row in level:
            f.write(''.join(row) + '\n')

    print(f"  Added {gaps_added} gaps and {enemies_added} enemies (difficulty={difficulty})")


# ── PHASE 1: Find hardest level ───────────────────────────────
print("=" * 60)
print("  PHASE 1: Generating and testing levels")
print("=" * 60)

best_level_path = None
lowest_completion = 2.0
results = []

for i in range(NUM_LEVELS):
    print(f"\n--- Level {i+1}/{NUM_LEVELS} ---")
    generate_and_repair()

    last_iter = get_last_iteration()
    if not last_iter:
        print("  ERROR: No iteration files found, skipping.")
        continue

    saved_path = os.path.join(MARIO_REPAIRER, f"generated_level_{i+1}.txt")
    shutil.copy(last_iter, saved_path)
    copy_to_framework(last_iter)

    print(f"  Running Mario AI agent...")
    completion = run_agent()
    results.append((i+1, completion, saved_path))
    print(f"  Level {i+1} completion: {completion*100:.4f}%")

    if completion < lowest_completion:
        lowest_completion = completion
        best_level_path = saved_path
        shutil.copy(saved_path, os.path.join(MARIO_REPAIRER, "hardest_level.txt"))

# ── PHASE 1 Summary ───────────────────────────────────────────
print("\n" + "=" * 60)
print("  PHASE 1 RESULTS")
print("=" * 60)
for idx, pct, path in results:
    marker = " ← HARDEST" if pct == lowest_completion else ""
    print(f"  Level {idx}: {pct*100:.4f}% completion{marker}")

# ── PHASE 2: Apply difficulty ─────────────────────────────────
print("\n" + "=" * 60)
print(f"  PHASE 2: Applying difficulty={DIFFICULTY} to hardest level")
print("=" * 60)

hardest_path    = os.path.join(MARIO_REPAIRER, "hardest_level.txt")
modified_path   = os.path.join(MARIO_REPAIRER, "hardest_level_modified.txt")

make_harder(hardest_path, modified_path, difficulty=DIFFICULTY)
copy_to_framework(modified_path)

# ── PHASE 3: Final evaluation ─────────────────────────────────
print("\n" + "=" * 60)
print("  PHASE 3: Final evaluation with difficulty applied")
print("=" * 60)

print("\nRunning Mario AI agent on modified level...")
final_completion = run_agent()

# ── Final Summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"  Levels generated       : {NUM_LEVELS}")
print(f"  Hardest base level     : {lowest_completion*100:.4f}% completion")
print(f"  Difficulty applied     : {DIFFICULTY}")
print(f"  Final completion       : {final_completion*100:.4f}%")
print(f"  Difficulty increase    : -{(lowest_completion - final_completion)*100:.4f}%")
print()
if final_completion < 0.5:
    print("  Result: VERY HARD level — Mario struggles significantly!")
elif final_completion < 0.8:
    print("  Result: HARD level — Mario completes with difficulty")
elif final_completion < 0.99:
    print("  Result: MEDIUM level — Mario barely completes it")
else:
    print("  Result: Level still too easy — try higher difficulty")
print()
print(f"  Modified level saved to : hardest_level_modified.txt")
print(f"  Loaded in framework at  : {GAN_LEVEL_DST}")
print("=" * 60)