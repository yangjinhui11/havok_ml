import numpy as np
import matplotlib.pylab as plt
import scipy.linalg as la
from scipy.io import loadmat
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from havok_py import HAVOK

from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor

matdata = loadmat("./DATA/lorenz.mat")
xdat = matdata['xdat']
dt = matdata['dt'][0][0]
tspan = matdata['tspan'][0]
######################### interpolate to dt=0.001
tspan = tspan-dt
Tend = tspan[-1]
ndt = 0.1*dt
new_tspan = np.arange(0,Tend+ndt,ndt)

fint = interp1d(tspan,xdat[:,0],kind='cubic')
new_data = fint(new_tspan).reshape(-1,1)

xdat = new_data
dt = ndt
tspan = new_tspan
############################
train_len = int(len(xdat)*0.5)+6
train_span = range(3000,train_len)
x_train = xdat[train_span,:]

plt.title("Lorenz system training data",fontsize=12)
plt.plot(tspan[train_span],x_train[:,0])
plt.xlabel('t',fontsize=12)
plt.ylabel('x',fontsize=12)
plt.show()
print("train time",tspan[train_span[0]],tspan[train_span[-1]])
n_timesteps = len(tspan)
print("lorenz_simulation",n_timesteps,xdat.shape,dt)
stackmax = 40
lamb = 1e-5
rmax = 11
dIndx = 5


## test for HAVOK
reg =  ExtraTreesRegressor(n_estimators=10,random_state=20)
#reg =  RandomForestRegressor(n_estimators=20,random_state=20)
#reg = LLNModel(maxM = 30,alpha = 1e-3)
#reg = MLPRegressor()

havok = HAVOK(rmax,stackmax,predInd=dIndx,lamb = lamb,regressor=reg)
havok.fit(x_train[:,0:1],dt)
havok.test(tspan[train_span])
iters = 1000
test_len = 600#len(xdat)-train_len
errors = np.zeros((iters,test_len*2))
for ite in range(iters):
#for ite in range(500,500+1):
    start_ind = train_len+ite*10
    if ite % 10==0: print("ite:",ite,start_ind)
    x_test = xdat[start_ind:,:]
    t_test = tspan[start_ind:]
    y_predic,vout = havok.predict(xdat[start_ind-stackmax:start_ind,0:1],test_len)
    errors[ite,:test_len] = x_test[:test_len,0]
    errors[ite,test_len:] = y_predic[:test_len,0]
    #print("mean error",np.mean(np.abs(errors[ite,:500,:]),axis=0))
np.savetxt("./result/lorenz/error_x.txt",errors)


plt.figure(3)
plt.subplot(2,1,1)
plt.plot(t_test[:test_len],y_predic[:test_len,0],label="predict x")
plt.plot(t_test[:test_len],x_test[:test_len,0],label="x")
plt.legend()
plt.subplot(2,1,2)
plt.plot(t_test[:test_len],vout[:],label="predict v")
plt.legend()
havok.test()
