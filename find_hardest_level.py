import subprocess
import os
import shutil
import glob

MARIO_FRAMEWORK = r"C:\Users\DELL\Mario-AI-Framework"
MARIO_REPAIRER  = r"C:\Users\DELL\MarioLevelRepairer"
GAN_LEVEL_DST   = os.path.join(MARIO_FRAMEWORK, "levels", "generated", "gan_level.txt")

def get_last_iteration():
    """Get the last iteration txt file from GA results."""
    files = glob.glob(os.path.join(MARIO_REPAIRER, "GA", "result", "txt", "iteration*.txt"))
    if not files:
        return None
    # Sort by iteration number
    files.sort(key=lambda x: int(x.split("iteration")[1].replace(".txt", "")))
    return files[-1]

def run_agent():
    """Run the Mario AI agent and return completion percentage."""
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
    return 1.0  # default if parsing fails

def generate_and_repair():
    """Generate a new level and repair it."""
    print("  Generating level with GAN...")
    subprocess.run(
        ["python", "LevelGenerator/GAN/generate_level.py"],
        cwd=MARIO_REPAIRER
    )
    print("  Clearing old repair results...")
    subprocess.run(
        ["python", "GA/clear.py"],
        cwd=MARIO_REPAIRER
    )
    print("  Repairing with GA...")
    subprocess.run(
        ["python", "GA/run.py"],
        cwd=MARIO_REPAIRER
    )

def copy_level_to_framework(src):
    """Copy a level file to the Mario AI Framework."""
    os.makedirs(os.path.dirname(GAN_LEVEL_DST), exist_ok=True)
    shutil.copy(src, GAN_LEVEL_DST)
    print(f"  Copied {src} → {GAN_LEVEL_DST}")

# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    NUM_LEVELS = 5  # change to 10 if you have more time

    best_level_path = None
    lowest_completion = 2.0
    results = []

    print("=" * 60)
    print(f"  Searching for hardest level across {NUM_LEVELS} generated levels")
    print("=" * 60)

    for i in range(NUM_LEVELS):
        print(f"\n--- Level {i+1}/{NUM_LEVELS} ---")

        generate_and_repair()

        last_iter = get_last_iteration()
        if not last_iter:
            print("  ERROR: No iteration files found, skipping.")
            continue

        print(f"  Using: {last_iter}")

        # Save a copy of this level
        saved_path = os.path.join(MARIO_REPAIRER, f"generated_level_{i+1}.txt")
        shutil.copy(last_iter, saved_path)

        copy_level_to_framework(last_iter)

        completion = run_agent()
        results.append((i+1, completion, saved_path))
        print(f"  ✓ Level {i+1} completion: {completion*100:.2f}%")

        if completion < lowest_completion:
            lowest_completion = completion
            best_level_path = saved_path
            # Save as hardest so far
            shutil.copy(saved_path, os.path.join(MARIO_REPAIRER, "hardest_level.txt"))

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for idx, pct, path in results:
        marker = " ← HARDEST" if pct == lowest_completion else ""
        print(f"  Level {idx}: {pct*100:.4f}% completion{marker}")

    print(f"\n  Hardest level saved to: hardest_level.txt")
    print(f"  Completion: {lowest_completion*100:.4f}%")

    # Copy hardest level to framework for final demo
    if best_level_path:
        copy_level_to_framework(best_level_path)
        print(f"\n  Hardest level is now loaded in Mario AI Framework.")
        print(f"  Run: java -cp classes PlayGANLevel")
        print(f"  to see Mario struggle through it!")
    print("=" * 60)