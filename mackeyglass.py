import numpy as np
from scipy.io import loadmat
import matplotlib.pylab as plt
from havok_py import HAVOK,LLNModel
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor

metdata = loadmat("./DATA/mackeyglass.mat")
xdat = metdata['xdat']
dt = metdata["dt"][0][0]
tspan = metdata['tspan'][0]
train0 = 3000 #int(100 /dt)
train1 = int(len(xdat)*0.7)
train_span = np.arange(train0,train1)
x_train = xdat[train_span]
print("train times: ",tspan[train0],tspan[train1],dt,xdat.shape)
stackmax = 5
lamb = 0#0.0025
rmax = 5
dIndx = 1
plt.figure(5)
plt.title("Mackey-Glass training data",fontsize=12)
plt.plot(tspan[train_span],x_train[:,0])
plt.xlabel('time',fontsize=12)
plt.ylabel('x',fontsize=12)

############################
reg = LLNModel(maxM = 30,alpha = 1e-6)
#reg =  ExtraTreesRegressor(n_estimators=10,random_state=20)
############################
havok = HAVOK(rmax,stackmax,predInd=dIndx,lamb = lamb,regressor=reg)
havok.fit(x_train[:,:],dt)
havok.test(tspan[train_span])

# predict
train1 = train1
test_len = 100
iters = 4000
errors = np.zeros((iters,test_len*2))
for ite in range(iters):
    test1 = train1+ite
    if ite%100==0: print(test1,ite,len(xdat))
    x_test =   xdat[test1:,:]
    t_test = tspan[test1:]
    y_predic,vout  = havok.predict(xdat[test1-stackmax:test1,:],test_len)
    errors[ite,:test_len] = x_test[:test_len,0]
    errors[ite,test_len:] = y_predic[:test_len,0]

np.savetxt("./result/mackeyglass/error_x.txt",errors)
plt.figure(3)
plt.subplot(1,2,1)
plt.plot(y_predic[:test_len,0],label="predict x")
plt.plot(x_test[:test_len,0],label="x")
plt.legend()
plt.subplot(1,2,2)
plt.plot(t_test[:test_len],vout[:],label="predict v")
plt.legend()
