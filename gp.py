import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.stats as st

class GPR(object):
    def __init__(self):
        self.is_fit = False
        self.trian_x,self.trian_y = None,None
        self.params = {"l":0.5,"sigma_f":0.2}
    
    def fit(self,trian_x,trian_y):
        self.trian_x,self.trian_y = trian_x,trian_y
        self.is_fit=True
    
    def predict(self,test_x):
        assert self.is_fit,"please fit first"
        test_x = np.array(test_x)
        kfy = self.gaussian_kernel(self.trian_x,test_x)
        kyy = self.gaussian_kernel(test_x,test_x)
        kff = self.gaussian_kernel(self.trian_x,self.trian_x)

        kff_inv = np.linalg.inv(kff)

        mu = kfy.T.dot(kff_inv).dot(self.trian_y)
        cov = kyy-kfy.T.dot(kff_inv).dot(kfy)
        return mu,cov


    def gaussian_kernel(self,x1,x2):
        d = np.sum(x1**2,1).reshape(-1,1) + np.sum(x2**2,1) - 2*np.dot(x1,x2.T)
        return self.params['sigma_f']**2 * np.exp(-0.5/self.params['l']**2*d)

def Branin(x1,x2):
    PI = 3.14159265359
    a = 1
    b = 5.1/(4*pow(PI, 2))
    c = 5/PI
    r = 6
    s = 10
    t = 1/(8*PI)
    return a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s
 


def EI(tau,mu,sigma):
    return (tau-mu)*st.norm.cdf((tau-mu)/sigma)+sigma*st.norm.pdf((tau-mu)/sigma)

def PI(tau,mu,sigma):
    return st.norm.cdf((tau-mu)/sigma)

def UCB(tau,mu,sigma):
    return -mu+sigma


def regret(f,f_):
    return f-f_

train_x = np.random.uniform(size=(3,2))
train_x[:,0],train_x[:,1] = train_x[:,0]*15-5,train_x[:,1]*15
train_y = Branin(train_x[:,0],train_x[:,1])

# 绘图

fig = plt.figure()
ax = plt.subplot(1,1,1,projection='3d')
x1 = np.arange(-5,10,1)
x2 = np.arange(0,15,1)
x1,x2 = np.meshgrid(x1,x2)
y = Branin(x1,x2)
print(Branin(9.42478,2.475))
ax.plot_surface(x1,x2,y,alpha=0.5)
ax.scatter3D(train_x[:,0],train_x[:,1],train_y,color='red')


# BO
def BO(gpr,n_sample,n_iter,X,Y,ac_func,f_opt):
    '''
    gpr: 高斯过程回归模型
    n_sample:每次采样量
    n_iter:最大迭代次数
    X,Y: 已有数据
    ac_func:收集函数
    f_opt:最优值
    '''
    train_x = copy.deepcopy(X)
    train_y = copy.deepcopy(Y)
    reg = np.full((n_iter+1,),0.)
    reg[0] = np.min(train_y)-f_opt
    for t in range(n_iter):
        gpr.fit(train_x,train_y)
        x1 = np.random.uniform(-5.,10.,n_sample)
        x2 = np.random.uniform(0.,15.,n_sample)
        sample_x = np.asarray(list(map(list,zip(x1,x2))))
        mu,cov = gpr.predict(sample_x)
        sigma = np.sqrt(np.ravel(np.diag(cov)))+1e-8
        tau = np.min(train_y)
        ac = ac_func(tau,mu,sigma)
        idx = np.argmax(ac)
        next_x = np.asarray([x1[idx],x2[idx]])
        next_y = Branin(x1[idx],x2[idx])
        train_x = np.insert(train_x,0,next_x,axis=0)
        train_y = np.insert(train_y,0,next_y)
        f_min = np.min(train_y)
        reg[t+1] = f_min-f_opt
    return reg, f_min


gpr = GPR()
n_sample = 20
n_iter = 200
f_opt = Branin(9.42478,2.475) #最优值

regret_ei,f_min_ei = BO(gpr,n_sample,n_iter,train_x,train_y,EI,f_opt)
regret_pi,f_min_pi = BO(gpr,n_sample,n_iter,train_x,train_y,PI,f_opt)
regret_ucb,f_min_ucb = BO(gpr,n_sample,n_iter,train_x,train_y,UCB,f_opt)
print("EI:",f_min_ei)
print("PI:",f_min_pi)
print("UCB:",f_min_ucb)
plt.figure()
plt.plot(regret_ei,label='EI')
plt.plot(regret_pi,label='PI')
plt.plot(regret_ucb,label='ucb')
plt.legend()
plt.show()
