import random
import sys

def make_harder(input_path, output_path, difficulty=1):
    with open(input_path) as f:
        lines = f.readlines()
    
    level = [list(line.rstrip('\n')) for line in lines]
    h = len(level)
    w = len(level[0]) if level else 0
    ground_row = h - 1

    # Add gaps in ground (most effective difficulty increase)
    num_gaps = difficulty * 3
    for _ in range(num_gaps):
        gap_start = random.randint(10, w - 15)
        gap_size = random.randint(3, 5)
        for g in range(gap_size):
            j = gap_start + g
            if j < w:
                # Remove ground tiles to create a pit
                for row in range(h - 3, h):
                    if level[row][j] == 'X':
                        level[row][j] = '-'

    # Add more enemies
    num_enemies = difficulty * 4
    for _ in range(num_enemies):
        j = random.randint(5, w - 5)
        # Place enemy on ground level
        for i in range(h - 2, 0, -1):
            if level[i][j] == 'X' and level[i-1][j] == '-':
                level[i-1][j] = 'E'
                break

    # Write output
    with open(output_path, 'w') as f:
        for row in level:
            f.write(''.join(row) + '\n')
    print(f"Harder level saved to {output_path}")

if __name__ == "__main__":
    difficulty = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    make_harder(
        "hardest_level.txt",
        "hardest_level_modified.txt",
        difficulty=difficulty
    )