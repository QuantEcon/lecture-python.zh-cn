from numba import float64
from numba.experimental import jitclass

opt_growth_data = [
    ('α', float64),          # 生产参数
    ('β', float64),          # 折现因子
    ('μ', float64),          # 冲击的均值参数
    ('s', float64),          # 冲击的尺度参数
    ('grid', float64[:]),    # 网格（数组）
    ('shocks', float64[:])   # 冲击样本（数组）
]

@jitclass(opt_growth_data)
class OptimalGrowthModel:

    def __init__(self,
                α=0.4,
                β=0.96,
                μ=0,
                s=0.1,
                grid_max=4,
                grid_size=120,
                shock_size=250,
                seed=1234):

        self.α, self.β, self.μ, self.s = α, β, μ, s

         # 设置网格
        self.grid = np.linspace(1e-5, grid_max, grid_size)

        # 存储冲击（设置随机种子以确保结果可重复）
        np.random.seed(seed)
        self.shocks = np.exp(μ + s * np.random.randn(shock_size))


    def f(self, k):
       "生产函数"
        return k**self.α


    def u(self, c):
        "效用函数"
        return np.log(c)

    def f_prime(self, k):
        "生产函数的一阶导数"
        return self.α * (k**(self.α - 1))


    def u_prime(self, c):
        "效用函数的一阶导数"
        return 1/c

    def u_prime_inv(self, c):
        "效用函数一阶导数的反函数"
        return 1/c
