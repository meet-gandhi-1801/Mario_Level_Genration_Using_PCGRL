# Mario Level Generation and Repair Using PCGRL

A comprehensive procedural content generation and repair system for Super Mario Bros levels using Convolutional Neural Networks (CNet) and Genetic Algorithms.

## 📋 Overview

This project implements a novel approach for repairing defective game levels using machine learning and evolutionary algorithms. It learns tile probability distributions from real levels and autonomously repairs invalid patterns in generated levels without explicit constraint programming.

**Research Paper:**
- Shu, T., Wang, Z., Liu, J., & Yao, X. (2020). "A Novel CNet-assisted Evolutionary Level Repairer and Its Applications to Super Mario Bros," 2020 IEEE Congress on Evolutionary Computation (CEC), Glasgow, United Kingdom, pp. 1-10. [[ArXiv](https://arxiv.org/abs/2005.06148)]

## 🎮 Key Features

- **CNet Model**: Learns conditional probability distributions of tiles based on surrounding context
- **Genetic Algorithm Repairer**: Optimizes tile replacements to repair illegal patterns
- **GAN Integration**: Supports GAN-generated levels for repair and analysis
- **Random Destruction**: Generate deliberately broken levels for testing
- **Visualization Tools**: Render repair progress and analyze results
- **Reinforcement Learning**: RL-based level generation alternatives using stable-baselines3
- **Multi-scale Analysis**: F1-score variants (F1, F2, F3) for different repair metrics

## 📁 Project Structure

```
├── CNet/                          # Convolutional Neural Network for tile prediction
│   ├── model.py                   # CNet architecture and training
│   ├── test.py                    # Model evaluation
│   ├── rule_fake.json             # Fake rules for testing
│   └── data/                      # Training data and rules
│       ├── generate.py            # Data generation script
│       ├── legal_rule.json        # Valid tile patterns
│       ├── illegal_rule.json      # Invalid tile patterns
│       └── *_F*.json              # F-score variants
├── GA/                            # Genetic Algorithm repairer
│   ├── run.py                     # Main repair execution
│   ├── repair.py                  # GA repair logic
│   ├── evaluate.py                # Performance evaluation
│   ├── render.py                  # Visualization renderer
│   ├── draw_graph.py              # Results graphing
│   ├── clear.py                   # Result cleanup
│   └── result/                    # Output and results
├── LevelGenerator/                # Level generation systems
│   ├── GAN/
│   │   ├── dcgan.py               # GAN architecture
│   │   ├── generate_level.py      # Generate levels with GAN
│   │   └── generator.pth          # Pre-trained GAN model
│   └── RandomDestroyed/           # Random destruction
│       ├── generate.py            # Create destroyed levels
│       └── lv*.txt                # Example destroyed levels
├── LevelText/                     # Original game levels
│   ├── pipes.txt                  # Pipe reference
│   └── MarioBrother2/             # SMB2 levels
├── RL/                            # Reinforcement Learning approach
│   ├── mario_rl_gen.py            # PPO-based generation
│   ├── train_agent.py             # Agent training
│   └── evaluate_agent.py           # Agent evaluation
├── utils/                         # Utility functions
│   ├── level_process.py           # Level I/O and processing
│   └── visualization.py           # Visualization helpers
├── Assets/Tiles/                  # Tile graphics for rendering
├── root.py                        # Root path utility
└── requirements.txt               # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- GPU (CUDA) recommended for CNet and GAN training

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Mario_Level_Genration_Using_PCGRL.git
cd Mario_Level_Genration_Using_PCGRL

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Repair Without Retraining

With pre-trained models and generated data included, repair a defective level:

```bash
# Clean previous results
python GA/clear.py

# Run repair
python GA/run.py

# Visualize results
python GA/draw_graph.py
python GA/evaluate.py
python GA/render.py
```

## 🔧 Complete Workflow (From Scratch)

### 1. Generate Training Data

```bash
python CNet/data/generate.py
```

### 2. Train CNet Model

```bash
python CNet/model.py
python CNet/test.py
```

### 3. Generate Levels to Repair

```bash
# Random destruction
python LevelGenerator/RandomDestroyed/generate.py

# Using pre-trained GAN
python LevelGenerator/GAN/generate_level.py
```

### 4. Repair Levels

```bash
python GA/run.py
```

### 5. Analyze Results

```bash
python GA/evaluate.py
python GA/draw_graph.py
python GA/render.py
```

### 6. Train RL Agent (Alternative)

```bash
python RL/train_agent.py
python RL/evaluate_agent.py
python RL/mario_rl_gen.py
```

## 📊 Understanding Tile Encoding

| ID | Tile | Description |
|---|---|---|
| 0 | Air | Empty space |
| 1 | Ground | Solid block |
| 2 | Brick | Breakable block |
| 3 | Question | Question mark block |
| 4 | Used | Used/empty block |
| 5 | Platform | Floating block |
| 6 | Pipe TL | Pipe top-left |
| 7 | Pipe TR | Pipe top-right |
| 8 | Pipe BL | Pipe body-left |
| 9 | Pipe BR | Pipe body-right |
| 10 | Enemy | Goomba/Koopa |

## 📝 Configuration

### Level Dimensions
- **Height**: 16 rows (rows)
- **Width**: Variable (typically 200+ columns)

### CNet Architecture
- **Input**: 12×8 neighbor tiles + 1 center tile = 97 features
- **Hidden Layers**: 200 → 100 neurons
- **Output**: 12 tile probabilities

### GA Parameters
Configured in `GA/repair.py` - adjust:
- Population size
- Mutation rate
- Crossover rate
- Number of generations

## 📈 Results

Results are saved in `GA/result/`:
- `txt/`: Numerical repair data per iteration
- `figure/`: Visual repair progress
- `json/`: Detailed metrics

Run `GA/evaluate.py` to compute:
- Tiles changed from illegal to legal
- Tiles changed from legal to illegal
- Repair success rate

## 🔬 RL Integration

The `RL/` folder contains PPO-based level generation using `stable-baselines3`:
- Generates levels tile-by-tile (left to right, top to bottom)
- Rewards valid and playable structures
- Can be combined with CNet for constraint satisfaction

## 🎯 Use Cases

- Game level design assistance
- Automated content validation
- Procedurally generated content repair
- AI-assisted creative tools
- Level design pattern analysis

## 🛠️ Extensibility

This system is designed for other grid-based games:
1. Adapt tile definitions for your game
2. Prepare training data from valid levels
3. Train new CNet model
4. Run GA repairer with new rules

## 📚 Requirements

- **numpy** ≥ 1.21.0 - Array operations
- **torch** ≥ 1.13.1 - Deep learning framework
- **pygame** ≥ 2.0.1 - Graphics rendering
- **matplotlib** ≥ 3.5.2 - Visualization and plotting
- **gymnasium** ≥ 0.28.1 - RL environment API
- **stable-baselines3** ≥ 2.0.0 - RL algorithms (PPO)
- **Pillow** ≥ 9.2.0 - Image processing

## ⚙️ System Requirements

- **OS**: Windows, macOS, Linux
- **Python**: 3.7 or higher
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

## 🐛 Troubleshooting

**GPU not detected:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Module not found:**
```bash
pip install --upgrade -r requirements.txt
```

**Level file errors:**
- Ensure level files are in correct text format
- Check `LevelText/` for format examples

## 📄 License

Please cite the original research paper if you use this code:

```bibtex
@INPROCEEDINGS{shu2020novel,
  title={A Novel CNet-assisted Evolutionary Level Repairer and Its Applications to Super Mario Bros},
  author={Shu, Tianye and Wang, Ziqi and Liu, Jialin and Yao, Xin},
  booktitle={2020 IEEE Congress on Evolutionary Computation (CEC)},
  doi={10.1109/CEC48606.2020.9185538},
  pages={1-10},
  year={2020}
}
```

## 📧 Contact & Attribution

- **Original Paper Authors**: Tianye Shu, Ziqi Wang, Jialin Liu, Xin Yao
- **Paper**: IEEE Congress on Evolutionary Computation (CEC) 2020
- **ArXiv**: https://arxiv.org/abs/2005.06148

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Original research team at University of Science and Technology of China / Southern University of Science and Technology
- Super Mario Bros game assets and design inspiration
- PyTorch, Gymnasium, and Stable-Baselines3 communities
