import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2
from scipy.ndimage import label
from scipy.stats import entropy
import json
import os

class PatternGenerator:
    def __init__(self, size=16):
        self.size = size
        self.dim = size * size

    def pattern_entropy(self, pattern):
        binary = ((pattern + 1) / 2).astype(int)  # convert -1/1 to 0/1
        p0 = np.mean(binary == 0)
        p1 = np.mean(binary == 1)
        probs = np.array([p0, p1])
        return entropy(probs, base=2)

    def pattern_complexity_structural(self, pattern):
        img = pattern.reshape(self.size, self.size)

        # Symmetry Score
        v_sym = np.mean(img == np.fliplr(img))
        h_sym = np.mean(img == np.flipud(img))
        d_sym = np.mean(img == img.T)
        symmetry = (v_sym + h_sym + d_sym) / 3

        # Connected Components Score
        binary_img = (img == 1).astype(int)
        structure = np.ones((3, 3))
        labeled, n_components = label(binary_img, structure=structure)
        connectedness = 1 / (n_components + 1e-6)

        return symmetry, connectedness

    def normalized_complexity(self, pattern, stats=None):
        entropy_val = self.pattern_entropy(pattern)
        symmetry, connectedness = self.pattern_complexity_structural(pattern)

        if stats is None:
            return entropy_val  # fallback

        # Normalize entropy
        entropy_norm = (entropy_val - stats['entropy_min']) / (stats['entropy_max'] - stats['entropy_min'] + 1e-6)

        # Final score: lower is more meaningful (more structure)
        # Intuitive human-like weighting: entropy (50%), symmetry (30%), connectedness (20%)
        structural_score = 0.2 * (1 - symmetry) + 0.3 * (1 - connectedness)
        return 0.5 * entropy_norm + structural_score

    def generate_random_pattern(self):
        return np.random.choice([1, -1], size=self.dim)

    def generate_structured_pattern(self):
        img = np.random.choice([1, -1], size=(self.size, self.size))

        if np.random.rand() < 0.5:
            img = (img + np.fliplr(img)) / 2
        if np.random.rand() < 0.5:
            img = (img + np.flipud(img)) / 2
        if np.random.rand() < 0.3:
            img = (img + img.T) / 2

        return np.where(img >= 0, 1, -1).flatten()

    def mutate_pattern(self, base_pattern, mutation_rate=0.05):
        pattern = base_pattern.copy()
        indices = np.random.choice(len(pattern), int(len(pattern) * mutation_rate), replace=False)
        pattern[indices] *= -1
        return pattern

    def generate_stat_baseline(self, samples=1000):
        entropy_vals = []
        for _ in range(samples):
            p = self.generate_structured_pattern()
            entropy_vals.append(self.pattern_entropy(p))
        return {
            'entropy_min': np.min(entropy_vals), 'entropy_max': np.max(entropy_vals)
        }

    def generate_similar_complexity_patterns(self, target_pattern, count=100, tolerance=0.05):
        #stats = self.generate_stat_baseline()
        target_score = self.normalized_complexity(target_pattern)

        valid_patterns = []
        attempts = 0
        while len(valid_patterns) < count and attempts < 200000:
            p = self.generate_structured_pattern()
            score = self.normalized_complexity(p)
            if abs(score - target_score) <= tolerance:
                valid_patterns.append((p, score))
            attempts += 1
        return valid_patterns

    def save_patterns(self, patterns, folder="saved_patterns"):
        os.makedirs(folder, exist_ok=True)
        metadata = {}

        for idx, (pattern, score) in enumerate(patterns):
            name_npy = f"pattern_{idx:03d}.npy"
            name_png = f"pattern_{idx:03d}.png"

            np.save(os.path.join(folder, name_npy), pattern)
            self.save_pattern_image(pattern, os.path.join(folder, name_png))

            metadata[name_npy] = {
                "complexity_score": score,
                "image_file": name_png
            }

        with open(os.path.join(folder, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    def save_pattern_image(self, pattern, filepath):
        img = pattern.reshape(self.size, self.size)
        plt.imsave(filepath, img, cmap='gray')

    def load_patterns(self, folder="saved_patterns", n=10):
        with open(os.path.join(folder, "metadata.json"), "r") as f:
            metadata = json.load(f)

        selected = list(metadata.items())[:n]
        patterns = []
        for filename, info in selected:
            pattern = np.load(os.path.join(folder, filename))
            patterns.append((pattern, info["complexity_score"]))
        return patterns

    def show_patterns(self, patterns, n=10, save_path=None):
        cols = 10
        rows = (n + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))
        axs = axs.flatten()

        for i in range(cols * rows):
            axs[i].axis('off')
            if i < n:
                axs[i].imshow(patterns[i][0].reshape(self.size, self.size), cmap='gray')
                axs[i].set_title(f"{patterns[i][1]:.3f}", fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

if __name__ == '__main__':
    generator = PatternGenerator()

    # Example: Use "十字" pattern as reference
    cross_pattern = np.array([1 if (i % 16 == 8 or i // 16 == 8) else -1 for i in range(256)])
    #test = np.array([1 if i//16 == i%16 else -1 for i in range(256)])
    #print(generator.normalized_complexity(test))

    similar_patterns = generator.generate_similar_complexity_patterns(cross_pattern, count=100, tolerance=0.1)
    print(len(similar_patterns))
    generator.save_patterns(similar_patterns)
    generator.show_patterns(similar_patterns, n=100, save_path="saved_patterns/overview.png")

    # Load and show saved patterns
    #loaded_patterns = generator.load_patterns(folder="saved_patterns", n=100)
    #generator.show_patterns(loaded_patterns, n=100, save_path="saved_patterns/overview_loaded.png")
