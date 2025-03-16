import numpy as np
import scipy as sp
import scipy.linalg as la
import quantecon as qe
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm



class AMF_LSS_VAR:
    """
    此类将加性（乘性）泛函转换为QuantEcon线性状态空间系统。
    """

    def __init__(self, A, B, D, F=None, nu=None, x_0=None):
        # 解包必需的元素
        self.nx, self.nk = B.shape
        self.A, self.B = A, B

        # 检查D的维度（从标量情况扩展）
        if len(D.shape) > 1 and D.shape[0] != 1:
            self.nm = D.shape[0]
            self.D = D
        elif len(D.shape) > 1 and D.shape[0] == 1:
            self.nm = 1
            self.D = D
        else:
            self.nm = 1
            self.D = np.expand_dims(D, 0)

        # 为加性分解创建空间
        self.add_decomp = None
        self.mult_decomp = None

        # 设置F
        if not np.any(F):
            self.F = np.zeros((self.nk, 1))
        else:
            self.F = F

        # 设置nu
        if not np.any(nu):
            self.nu = np.zeros((self.nm, 1))
        elif type(nu) == float:
            self.nu = np.asarray([[nu]])
        elif len(nu.shape) == 1:
            self.nu = np.expand_dims(nu, 1)
        else:
            self.nu = nu

        if self.nu.shape[0] != self.D.shape[0]:
            raise ValueError("nu的维度与D不一致！")

        # 初始化模拟器
        self.x_0 = x_0

        # 构建大型状态空间表示
        self.lss = self.construct_ss()

    def construct_ss(self):
        """
        这将创建可以传递到quantecon LSS类的状态空间表示。
        """

        # 提取有用信息
        nx, nk, nm = self.nx, self.nk, self.nm
        A, B, D, F, nu = self.A, self.B, self.D, self.F, self.nu

        if self.add_decomp:
            nu, H, g = self.add_decomp
        else:
            nu, H, g = self.additive_decomp()

        # 用0和1填充lss矩阵的辅助块
        nx0c = np.zeros((nx, 1))
        nx0r = np.zeros(nx)
        nx1 = np.ones(nx)
        nk0 = np.zeros(nk)
        ny0c = np.zeros((nm, 1))
        ny0r = np.zeros(nm)
        ny1m = np.eye(nm)
        ny0m = np.zeros((nm, nm))
        nyx0m = np.zeros_like(D)

        # 为LSS构建A矩阵
        # 状态顺序为：[1, t, xt, yt, mt]
        A1 = np.hstack([1, 0, nx0r, ny0r, ny0r])            # 1的转移
        A2 = np.hstack([1, 1, nx0r, ny0r, ny0r])            # t的转移
        A3 = np.hstack([nx0c, nx0c, A, nyx0m.T, nyx0m.T])   # x_{t+1}的转移
        A4 = np.hstack([nu, ny0c, D, ny1m, ny0m])           # y_{t+1}的转移
        A5 = np.hstack([ny0c, ny0c, nyx0m, ny0m, ny1m])     # m_{t+1}的转移
        Abar = np.vstack([A1, A2, A3, A4, A5])

        # 为LSS构建B矩阵
        Bbar = np.vstack([nk0, nk0, B, F, H])

        # 为LSS构建G矩阵
        # 观测顺序为：[xt, yt, mt, st, tt]
        G1 = np.hstack([nx0c, nx0c, np.eye(nx), nyx0m.T, nyx0m.T])    # x_{t}的选择器
        G2 = np.hstack([ny0c, ny0c, nyx0m, ny1m, ny0m])               # y_{t}的选择器
        G3 = np.hstack([ny0c, ny0c, nyx0m, ny0m, ny1m])               # 鞅的选择器
        G4 = np.hstack([ny0c, ny0c, -g, ny0m, ny0m])                  # 平稳部分的选择器
        G5 = np.hstack([ny0c, nu, nyx0m, ny0m, ny0m])                 # 趋势的选择器
        Gbar = np.vstack([G1, G2, G3, G4, G5])

        # 为LSS构建H矩阵
        Hbar = np.zeros((Gbar.shape[0], nk))

        # 构建LSS类型
        if not np.any(self.x_0):
            x0 = np.hstack([1, 0, nx0r, ny0r, ny0r])
        else:
            x0 = self.x_0

        S0 = np.zeros((len(x0), len(x0)))
        lss = qe.lss.LinearStateSpace(Abar, Bbar, Gbar, Hbar, mu_0=x0, Sigma_0=S0)

        return lss

    def additive_decomp(self):
        """
        返回鞅分解的值
            - nu        : Y的无条件均值差异
            - H         : （线性）鞅分量的系数 (kappa_a)
            - g         : 平稳分量g(x)的系数
            - Y_0       : 应该是X_0的函数（目前设置为0.0）
        """
        I = np.eye(self.nx)
        A_res = la.solve(I - self.A, I)
        g = self.D @ A_res
        H = self.F + self.D @ A_res @ self.B

        return self.nu, H, g

    def multiplicative_decomp(self):
        """
        返回乘性分解的值（示例5.4.4）
            - nu_tilde  : 特征值
            - H         : Jensen项的向量
        """
        nu, H, g = self.additive_decomp()
        nu_tilde = nu + (.5)*np.expand_dims(np.diag(H @ H.T), 1)

        return nu_tilde, H, g





def future_moments(amf_future, N=25):
    """
    计算未来时刻的矩
    """
    nx, nk, nm = amf_future.nx, amf_future.nk, amf_future.nm
    nu_tilde, H, g = amf_future.multiplicative_decomp()
    
    # 分配空间（nm是加性泛函的数量）
    mbounds = np.zeros((3, N))
    sbounds = np.zeros((3, N))
    ybounds = np.zeros((3, N))
    #mbounds_mult = np.zeros((3, N))
    #sbounds_mult = np.zeros((3, N))

    # 模拟所需的时长
    moment_generator = amf_future.lss.moment_sequence()
    tmoms = next(moment_generator)

    # 提取总体矩
    for t in range (N-1):
        tmoms = next(moment_generator)
        ymeans = tmoms[1]
        yvar = tmoms[3]

        # 每个加性泛函的上下界
        yadd_dist = norm(ymeans[nx], np.sqrt(yvar[nx, nx]))
        ybounds[:2, t+1] = yadd_dist.ppf([0.1, .9])
        ybounds[2, t+1] = yadd_dist.mean()

        madd_dist = norm(ymeans[nx+nm], np.sqrt(yvar[nx+nm, nx+nm]))
        mbounds[:2, t+1] = madd_dist.ppf([0.1, .9])
        mbounds[2, t+1] = madd_dist.mean()

        sadd_dist = norm(ymeans[nx+2*nm], np.sqrt(yvar[nx+2*nm, nx+2*nm]))
        sbounds[:2, t+1] = sadd_dist.ppf([0.1, .9])
        sbounds[2, t+1] = sadd_dist.mean()


        #Mdist = lognorm(np.asscalar(np.sqrt(yvar[nx+nm, nx+nm])), scale=np.asscalar(np.exp(ymeans[nx+nm]- \
        #                                              t*(.5)*np.expand_dims(np.diag(H @ H.T), 1))))
        #Sdist = lognorm(np.asscalar(np.sqrt(yvar[nx+2*nm, nx+2*nm])),
        #                scale = np.asscalar(np.exp(-ymeans[nx+2*nm])))
        #mbounds_mult[:2, t+1] = Mdist.ppf([.01, .99])
        #mbounds_mult[2, t+1] = Mdist.mean()

        #sbounds_mult[:2, t+1] = Sdist.ppf([.01, .99])
        #sbounds_mult[2, t+1] = Sdist.mean()

    ybounds[:, 0] = amf_future.x_0[2+nx]
    mbounds[:, 0] = mbounds[-1, 1]
    sbounds[:, 0] = -g @ amf_future.x_0[2:2+nx]

    #mbounds_mult[:, 0] = mbounds_mult[-1, 1]
    #sbounds_mult[:, 0] = np.exp(-g @ amf_future.x_0[2:2+nx])

    return mbounds, sbounds, ybounds #, mbounds_mult, sbounds_mult
