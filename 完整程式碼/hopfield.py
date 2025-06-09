import numpy as np

class HopfieldNetwork: #主要神經網路模型
    def __init__(self, N=256, death_rate=0.0, residual_strength=0.1, plaque_intensity=0.0, plaque_rate=0.3):
        self.N = N #神經元數量
        self.death_rate = death_rate #突觸受傷數量
        self.residual_strength = residual_strength #突觸受傷程度(1代表死亡)
        self.plaque_intensity = plaque_intensity #神經元受傷程度(1代表死亡)
        self.plaque_count = int(plaque_rate*self.N) #神經元受傷數量
        self.W = np.zeros((N, N)) #神經網路權重(類比突觸)
        self.plaque_mask = np.ones((N, N)) #神經元受傷遮罩
        self.death_mask = np.ones((N, N)) #突觸受傷遮罩

    def train(self, patterns): #主要訓練模型
        P = len(patterns)
        self.W = np.zeros((self.N, self.N))
        k = 0
        while k < P:
            new_p = [j for j in patterns[k]]
            if len(new_p) == len(self.W):
                self.W += np.outer(new_p, new_p)
            elif len(new_p) < len(self.W):
                while len(new_p) < len(self.W):
                    n = int((len(new_p))**(0.5))
                    new_p = np.reshape(new_p, (n, n))
                    newer_p = np.reshape([0 for _ in range(4*n*n)], (2*n, 2*n))
                    for i in range(n):
                        for j in range(n):
                            newer_p[2*i][2*j] = new_p[i][j]
                            newer_p[2*i+1][2*j] = new_p[i][j]
                            newer_p[2*i][2*j+1] = new_p[i][j]
                            newer_p[2*i+1][2*j+1] = new_p[i][j]
                    new_p = []
                    for i in newer_p:
                        new_p += list(i)
                self.W += np.outer(new_p, new_p)
            else:
                while len(new_p) > len(self.W):
                    n = int((len(new_p))**(0.5))
                    new_p = np.reshape(new_p, (n, n))
                    newer_p = np.reshape([0 for _ in range(n*n//4)], (n//2, n//2))
                    for i in range(n//2):
                        for j in range(n//2):
                            newer_p[i][j] = (new_p[2*i][2*j] + new_p[2*i+1][2*j] + new_p[2*i][2*j+1] + new_p[2*i+1][2*j+1])/4
                    new_p = []
                    for i in newer_p:
                        new_p += list(i)
                self.W += np.outer(new_p, new_p)
            k += 1
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

    def recall(self, pattern): #回憶受到噪點影響的圖形
        s = pattern.copy() #1d
        n = self.N
        while len(s) < len(self.W):
            n = int((len(s))**(0.5))
            s = np.reshape(s, (n, n))
            new_s = np.reshape([0 for _ in range(4*n*n)], (2*n, 2*n))
            for i in range(n):
                for j in range(n):
                    new_s[2*i][2*j] = s[i][j]
                    new_s[2*i+1][2*j] = s[i][j]
                    new_s[2*i][2*j+1] = s[i][j]
                    new_s[2*i+1][2*j+1] = s[i][j]
            s = []
            for i in new_s:
                s += list(i)
        for i in range(len(s)):
            sum = np.dot(self.W[i], s)
            s[i] = 1 if sum > 0 else -1
        return np.array(s)

    def add_noise(self, pattern, noise_level=0.1): #增加輸入圖片噪點
        noisy = pattern.copy()
        size = len(pattern)
        n_flip = int(size * noise_level)
        flip_indices = np.random.choice(size, n_flip, replace=False)
        noisy[flip_indices] *= -1
        return noisy

if __name__ == "__main__":
    # Define training patterns (bipolar: -1 and 1)
    temp = [1, -1]
    patterns = np.array([
        np.random.choice(temp, 64) for _ in range(20)
    ])

    # Initialize and train the network
    hopfield_net = HopfieldNetwork(N=64)
    hopfield_net.train(patterns)

    # Test pattern recall
    test_pattern = patterns[0]
    recalled_pattern = hopfield_net.recall(test_pattern)

    print("Input Pattern:   ", test_pattern)
    print("Recalled Pattern:", recalled_pattern)
    print("success rate", sum([abs(test_pattern[i]+recalled_pattern[i]) for i in range(len(test_pattern))]) / (2*len(test_pattern)))
