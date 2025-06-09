import numpy as np

class HopfieldNetwork: #主要神經網路模型
    def __init__(self, N=256, death_rate=0.0, residual_strength=0.1, plaque_intensity=0.0, plaque_count=3):
        self.N = N #神經元數量
        self.death_rate = death_rate #突觸受傷數量
        self.residual_strength = residual_strength #突觸受傷程度(1代表死亡)
        self.plaque_intensity = plaque_intensity #神經元受傷程度(1代表死亡)
        self.plaque_count = plaque_count #神經元受傷數量
        self.W = np.zeros((N, N)) #神經網路權重(類比突觸)
        self.plaque_mask = np.ones((N, N)) #神經元受傷遮罩
        self.death_mask = np.ones((N, N)) #突觸受傷遮罩

    def train(self, patterns): #主要訓練模型
        P = len(patterns)
        self.W = np.zeros((self.N, self.N))
        for p in patterns:
            self.W += np.outer(p, p)
        self.W /= self.N
        np.fill_diagonal(self.W, 0)

        ### 突觸稀疏
        self.death_mask = np.ones((self.N, self.N))
        total_connections = [(i, j) for i in range(self.N) for j in range(i)]
        dead_connections = int(len(total_connections) * self.death_rate)
        death_indices = np.random.choice(len(total_connections), dead_connections, replace=False)
        for idx in death_indices:
            i, j = total_connections[idx]
            self.death_mask[i, j] = 1 - self.residual_strength
            self.death_mask[j, i] = 1 - self.residual_strength

        ### 空間擴散式神經元受傷
        self.plaque_mask = np.ones((self.N, self.N))
        side_len = int(np.sqrt(self.N))
        grid = np.arange(self.N).reshape((side_len, side_len))
        for _ in range(self.plaque_count):
            cx, cy = np.random.randint(0, side_len), np.random.randint(0, side_len)
            max_radius = np.random.randint(2, 6)
            for dx in range(-max_radius, max_radius + 1):
                for dy in range(-max_radius, max_radius + 1):
                    x, y = cx + dx, cy + dy
                    if 0 <= x < side_len and 0 <= y < side_len:
                        idx = grid[x, y]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist <= max_radius:
                            # 神經元受傷程度與距離成反比
                            local_intensity = (1 - dist / max_radius) * self.plaque_intensity
                            factor = max(0, 1 - local_intensity)
                            self.plaque_mask[idx, :] *= factor
                            self.plaque_mask[:, idx] *= factor

        self.W = self.W * self.death_mask * self.plaque_mask

    """
    def recall(self, pattern, steps=10): #回憶受到噪點影響的圖形
        s = pattern.copy()
        for _ in range(steps):
            s = np.sign(self.W @ s)
            s[s == 0] = 1
        return s
    """

    def recall(self, pattern, steps=10): #回憶受到噪點影響的圖形
        s = pattern.copy()
        for _ in range(steps):
            for i in np.random.permutation(self.N):
                h_i = np.dot(self.W[i, :], s)
                s[i] = 1 if h_i >= 0 else -1
        return s

    def add_noise(self, pattern, noise_level=0.1): #增加輸入圖片噪點
        noisy = pattern.copy()
        n_flip = int(self.N * noise_level)
        flip_indices = np.random.choice(self.N, n_flip, replace=False)
        noisy[flip_indices] *= -1
        return noisy