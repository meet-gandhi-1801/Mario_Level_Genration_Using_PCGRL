import os

# Tile mapping from repo numbers to Mario AI Framework characters
TILE_MAP = {
    0:  '-',   # air
    1:  'X',   # ground
    2:  'S',   # brick
    3:  '?',   # question block
    4:  'Q',   # used block
    5:  '#',   # platform
    6:  '<',   # pipe top left
    7:  '>',   # pipe top right
    8:  '[',   # pipe body left
    9:  ']',   # pipe body right
    10: 'E',   # enemy
    11: 'X',   # border → treat as ground
}

def convert_level(input_path, output_path):
    with open(input_path) as f:
        lines = f.readlines()
    
    converted = []
    for line in lines:
        row = ''
        for ch in line.strip().split():
            tile = int(ch)
            row += TILE_MAP.get(tile, '-')
        if row:
            converted.append(row)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(converted))
    print(f"Converted: {output_path}")

# Convert the repaired level
convert_level(
    'GA/result/txt/iteration169.txt',
    '../Mario-AI-Framework/levels/generated/gan_level.txt'
)