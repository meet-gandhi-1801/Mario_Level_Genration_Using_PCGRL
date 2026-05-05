import sys, os
sys.path.append(sys.path[0] + '//..' + '//..')
os.chdir(sys.path[0])
import torch
from torch.autograd import Variable
from LevelGenerator.GAN.dcgan import Generator
from utils.level_process import *
from utils.visualization import *
from root import rootpath
import numpy as np


def get_level(noise, to_string, name, size):
    model_to_load = name
    batch_size = 1
    image_size = 32 * size
    ngf = 64
    nz = 32
    z_dims = 10
    generator = Generator(nz, ngf, image_size, z_dims)
    generator.load_state_dict(torch.load(model_to_load, map_location=lambda storage, loc: storage))
    latent_vector = torch.FloatTensor(noise).view(batch_size, nz, 1, 1)
    with torch.no_grad():
        levels = generator(Variable(latent_vector))
    im = levels.data.cpu().numpy()
    im = np.argmax(im, axis=1)
    im = little_level(im[0], size)
    if to_string:
        return arr_to_str(im[0:14, 0:28])
    else:
        return im[0:14, 0:28]


def get_level_with_noise(noise):
    """Generate a full long level using a specific noise vector."""
    lvs = []
    for i in range(int(120 / 28)):
        # Perturb noise slightly for each segment to add variety
        segment_noise = noise + np.random.randn(1, 32) * 0.3
        lvs.append(get_level(segment_noise, False, './generator.pth', 1))
    lv = np.concatenate(lvs, axis=-1)
    lv = addLine(lv)
    return lv


def sample_noise(strategy):
    """
    Sample latent vectors using different strategies to explore
    different regions of the GAN's latent space.
    """
    if strategy == 0:
        # Standard normal — baseline
        return np.random.randn(1, 32)
    elif strategy == 1:
        # Wide normal — more extreme features
        return np.random.randn(1, 32) * 2.5
    elif strategy == 2:
        # Narrow normal — conservative, simpler levels
        return np.random.randn(1, 32) * 0.4
    elif strategy == 3:
        # Uniform — completely different distribution
        return np.random.uniform(-3, 3, (1, 32))
    elif strategy == 4:
        # Positive biased — explores one side of latent space
        return np.abs(np.random.randn(1, 32)) * 1.5
    elif strategy == 5:
        # Negative biased — explores other side
        return -np.abs(np.random.randn(1, 32)) * 1.5
    elif strategy == 6:
        # Sparse — most dims near zero, few extreme
        noise = np.random.randn(1, 32) * 0.1
        extreme_dims = np.random.choice(32, 8, replace=False)
        noise[0, extreme_dims] = np.random.randn(8) * 3.0
        return noise
    else:
        # Mixed — random combination
        return np.random.randn(1, 32) * np.random.uniform(0.5, 2.5)


if __name__ == '__main__':
    lvs = []
    total = 200      # generate more candidates
    select = 5       # keep best 5

    print(f"Generating {total} candidate levels with diverse sampling...")

    for i in range(total):
        print(f'\rgenerate {i+1}/{total}', end='')

        # Cycle through sampling strategies for maximum diversity
        strategy = i % 8
        noise = sample_noise(strategy)

        lv = get_level_with_noise(noise)
        cnt = calculate_broken_pipes(lv)

        # Store noise with level so we can reproduce it
        lvs.append((cnt, lv, strategy, noise))

    lvs.sort(key=lambda s: s[0], reverse=True)

    print()
    print(f"\nTop {select} most structurally complex levels:")
    cnt_sum = 0
    for i in range(select):
        cnt, lv, strategy, noise = lvs[i]
        saveLevelAsImage(lv, 'Destroyed//lv' + str(i))
        with open('Destroyed//lv' + str(i) + '.txt', 'w') as f:
            f.write(arr_to_str(lv))
        print(f'lv{i}: broken_pipes={cnt}, sampling_strategy={strategy}')
        cnt_sum += cnt

    print(f'avg_broken_pipe_combinations= {cnt_sum / total:.2f}')
    print('Levels saved in LevelGenerator//GAN//Destroyed folder.')