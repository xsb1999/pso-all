
import numpy as np
from sko.tools import func_transformer
from base import SkoBase
import matplotlib.pyplot as plt


class PSO_0(SkoBase):
    def __init__(self, func, dim, pop=40, max_iter=150, lb=None, ub=None, w=0.8, c1=0.5, c2=0.5):
#         待优化函数func
        self.func = func_transformer(func)
#         惯性系数w
        self.w = w  # inertia
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.pop = pop  # 粒子数
        self.dim = dim  # 维数
        self.max_iter = max_iter  # 最大迭代次数

        self.has_constraints = not (lb is None and ub is None)
        self.lb = -np.ones(self.dim) if lb is None else np.array(lb)
        self.ub = np.ones(self.dim) if ub is None else np.array(ub)
#         dim的维数要和lb、ub相同（判断输入是否正确）
        assert self.dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'
#         初始化粒子的位置
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim))
        v_high = self.ub - self.lb
#         初始化粒子的速度
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.dim))  # speed of particles
#         计算每个粒子的适应度值
        self.Y = self.cal_y()  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = self.Y.copy()  # best image of every particle in history
        
        self.gbest_x = np.zeros((1, self.dim))  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()

        # record values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}

    def update_V(self):
        r1 = np.random.rand(self.pop, self.dim)
        r2 = np.random.rand(self.pop, self.dim)
        self.V = self.w * self.V + self.cp * r1 * (self.pbest_x - self.X) + self.cg * r2 * (self.gbest_x - self.X)

    def update_X(self):
        self.X = self.X + self.V

        if self.has_constraints:
            self.X = np.clip(self.X, self.lb, self.ub)

    def cal_y(self):
        # calculate y for every x in X
        self.Y = self.func(self.X).reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.pbest_x = np.where(self.pbest_y > self.Y, self.X, self.pbest_x)
        self.pbest_y = np.where(self.pbest_y > self.Y, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        if self.gbest_y > self.Y.min():
            self.gbest_x = self.X[self.Y.argmin(), :].copy()
            self.gbest_y = self.Y.min()

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(self.max_iter):
            self.update_V()
            self.recorder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()

            self.gbest_y_hist.append(self.gbest_y)
        return self

    fit = run


class PSO_1(SkoBase):
    def __init__(self, func, dim, pop=40, max_iter=150, lb=None, ub=None, w_max=0.8, w_min=0.4, c1=0.5, c2=0.5):
        self.func = func_transformer(func)
        self.w_max = w_max  # inertia max
        self.w_min = w_min  # inertia min
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.pop = pop  # number of particles
        self.dim = dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter

        self.has_constraints = not (lb is None and ub is None)
        self.lb = -np.ones(self.dim) if lb is None else np.array(lb)
        self.ub = np.ones(self.dim) if ub is None else np.array(ub)
        assert self.dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'
        
#         np.random.seed(25)
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim))
        v_high = self.ub - self.lb
#         np.random.seed(20)
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.dim))  # speed of particles
        self.Y = self.cal_y()  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = self.Y.copy()  # best image of every particle in history
        self.gbest_x = np.zeros((1, self.dim))  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}

    def update_V(self):
#         np.random.seed(1)
        r1 = np.random.rand(self.dim)
#         np.random.seed(2)
        r2 = np.random.rand(self.dim)
        
        f_all = self.func(self.X)
        f_min = min(f_all)
        f_avg = sum(f_all) / len(f_all)
          
        V_list = []
        
        for i in range(len(f_all)):
            
            if f_all[i] > f_avg:
                w = self.w_max
            else:
                w = self.w_min + ((self.w_max - self.w_min) * (f_all[i] - f_min) / (f_avg - f_min))
            
            if w < self.w_min:
                w = self.w_min
                
            v = w * self.V[i] + self.cp * r1 * (self.pbest_x[i] - self.X[i]) + self.cg * r2 * (self.gbest_x - self.X[i])
            
            
            V_list.append(v)
        
        self.V = np.array(V_list)
        
            
    def update_X(self):
        self.X = self.X + self.V

        if self.has_constraints:
            self.X = np.clip(self.X, self.lb, self.ub)

    def cal_y(self):
        # calculate y for every x in X
        self.Y = self.func(self.X).reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.pbest_x = np.where(self.pbest_y > self.Y, self.X, self.pbest_x)
        self.pbest_y = np.where(self.pbest_y > self.Y, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        if self.gbest_y > self.Y.min():
            self.gbest_x = self.X[self.Y.argmin(), :].copy()
            self.gbest_y = self.Y.min()

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(self.max_iter):
            self.update_V()
            self.recorder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()

            self.gbest_y_hist.append(self.gbest_y)
        return self

    fit = run


class PSO_2(SkoBase):
    def __init__(self, func, dim, pop=40, max_iter=150, lb=None, ub=None, w_max=0.8, w_min=0.4, c1=0.5, c2=0.5, mul=500):
        self.mul = mul
        self.func = func_transformer(func)
        self.w_max = w_max  # inertia max
        self.w_min = w_min  # inertia min
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.pop = pop  # number of particles
        self.dim = dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter

        self.has_constraints = not (lb is None and ub is None)
        self.lb = -np.ones(self.dim) if lb is None else np.array(lb)
        self.ub = np.ones(self.dim) if ub is None else np.array(ub)
        assert self.dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'
        
#         np.random.seed(25)
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim))
        v_high = self.ub - self.lb
#         np.random.seed(20)
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.dim))  # speed of particles
        self.Y = self.cal_y()  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = self.Y.copy()  # best image of every particle in history
        self.gbest_x = np.zeros((1, self.dim))  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}

    def update_V(self):
#         np.random.seed(1)
        r1 = np.random.rand(self.dim)
#         np.random.seed(2)
        r2 = np.random.rand(self.dim)
        
        f_all = self.func(self.X)
        f_min = min(f_all)
        f_avg = sum(f_all) / len(f_all)
        
        V_list = []
        
#         len(f_all) = 粒子数
        for i in range(len(f_all)):
            
            print(f_avg - f_all[i])
            print("ok")
            
            if f_all[i] > f_avg:
                w = self.w_max
            else:
                w = self.w_max - (self.mul*(f_avg - f_all[i])*(self.w_max - self.w_min) * (f_all[i] - f_min) / (f_avg - f_min))
        
            if w < self.w_min:
                w = self.w_min
                
            v = w * self.V[i] + self.cp * r1 * (self.pbest_x[i] - self.X[i]) + self.cg * r2 * (self.gbest_x - self.X[i])
                      
            V_list.append(v)
                     
        self.V = np.array(V_list)
             
        
    def update_X(self):
        self.X = self.X + self.V

        if self.has_constraints:
            self.X = np.clip(self.X, self.lb, self.ub)

    def cal_y(self):
        # calculate y for every x in X
        self.Y = self.func(self.X).reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.pbest_x = np.where(self.pbest_y > self.Y, self.X, self.pbest_x)
        self.pbest_y = np.where(self.pbest_y > self.Y, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        if self.gbest_y > self.Y.min():
            self.gbest_x = self.X[self.Y.argmin(), :].copy()
            self.gbest_y = self.Y.min()

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(self.max_iter):
            self.update_V()
            self.recorder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()

            self.gbest_y_hist.append(self.gbest_y)
        return self

    fit = run
    

class PSO_3(SkoBase):
    def __init__(self, func, dim, pop=40, max_iter=150, lb=None, ub=None, w_max=0.8, w_min=0.4, c1=0.5, c2=0.5, mul=500):
        self.mul = mul
        self.func = func_transformer(func)
        self.w_max = w_max  # inertia max
        self.w_min = w_min  # inertia min
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.pop = pop  # number of particles
        self.dim = dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter

        self.has_constraints = not (lb is None and ub is None)
        self.lb = -np.ones(self.dim) if lb is None else np.array(lb)
        self.ub = np.ones(self.dim) if ub is None else np.array(ub)
        assert self.dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'
        
#         np.random.seed(25)
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim))
        v_high = self.ub - self.lb
#         np.random.seed(20)
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.dim))  # speed of particles
        self.Y = self.cal_y()  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = self.Y.copy()  # best image of every particle in history
        self.gbest_x = np.zeros((1, self.dim))  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}

    def update_V(self):
#         np.random.seed(1)
        r1 = np.random.rand(self.dim)
#         np.random.seed(2)
        r2 = np.random.rand(self.dim)
        
        f_all = self.func(self.X)
        f_min = min(f_all)
        f_avg = sum(f_all) / len(f_all)
        
        V_list = []
        
#         len(f_all) = 粒子数
        for i in range(len(f_all)):
            
            print(f_avg - f_all[i])
            print("ok")
            
            if f_all[i] > f_avg:
                w = self.w_max
            else:
                w = self.w_max - (self.mul * (f_avg - f_all[i]) * (self.w_max - self.w_min) * (f_avg - f_min))
        
            if w < self.w_min:
                w = self.w_min
                
            v = w * self.V[i] + self.cp * r1 * (self.pbest_x[i] - self.X[i]) + self.cg * r2 * (self.gbest_x - self.X[i])
                      
            V_list.append(v)
                     
        self.V = np.array(V_list)
             
        
    def update_X(self):
        self.X = self.X + self.V

        if self.has_constraints:
            self.X = np.clip(self.X, self.lb, self.ub)

    def cal_y(self):
        # calculate y for every x in X
        self.Y = self.func(self.X).reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.pbest_x = np.where(self.pbest_y > self.Y, self.X, self.pbest_x)
        self.pbest_y = np.where(self.pbest_y > self.Y, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        if self.gbest_y > self.Y.min():
            self.gbest_x = self.X[self.Y.argmin(), :].copy()
            self.gbest_y = self.Y.min()

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(self.max_iter):
            self.update_V()
            self.recorder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()

            self.gbest_y_hist.append(self.gbest_y)
        return self

    fit = run



