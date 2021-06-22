import numpy as np
from havok_py import HAVOK,LLNModel
import matplotlib.pylab as plt
from scipy.interpolate import interp1d


xdat = np.loadtxt('./DATA/SN_ms_tot_V2.0.csv',dtype='float',delimiter=';')
print(xdat.shape)
N = len(xdat)
L = range(1030,N-200)
month_label = xdat[L,0:2]
N = len(L)
print(N,month_label[0],month_label[-1])
xdata = xdat[L,3]
xmax = np.max(xdata)
xmin = np.min(xdata)
xdata = (2*xdata-(xmax+xmin))/(xmax-xmin) # normalize
months = np.arange(N)
dt = 1

fint = interp1d(months,xdata,kind='cubic')
dt = 0.02
new_month = np.arange(0,N-1+dt,dt)
new_data = fint(new_month).reshape(-1,1)
tspan = new_month
N = len(new_data)

train_start = 0
train_len = 50000 #int(N*0.5)
train_span = np.arange(train_start,train_len)
print("train data",train_len,(train_len-train_start)*dt, month_label[int(train_start*dt)],month_label[int(train_len*dt)])
x_train = new_data[train_span]
plt.figure(3)
plt.title("Sunspot  training data",fontsize=12)
plt.plot(tspan[train_span],x_train[:,0])
plt.xlabel('time',fontsize=12)
plt.ylabel('x',fontsize=12)

stackmax = 140
lamb = 0.0025
rmax = 7
dIndx = 20
############################
reg = LLNModel(maxM=3,alpha = 1e-6)
#reg =  ExtraTreesRegressor(n_estimators=10,random_state=20)
############################
havok = HAVOK(rmax,stackmax,predInd=dIndx,lamb = lamb,regressor=reg)

havok.fit(x_train[:],dt)

havok.test( tspan[train_span])

print("data length",N,train_len)
iters = 1000
test_len = 12*int(1./dt)#len(xdat)-train_len
#train_len = int(N*0.6)
errors = np.zeros((iters,test_len*2))
for ite in range(iters):
    start_ind = train_len+int(1./(dt)*ite)
    if ite % 100==0:
        
        print("ite:",ite,start_ind,N)
        print("start month",month_label[int(start_ind*dt)])
    x_test = new_data[start_ind:,:]
    t_test = tspan[start_ind:]
    y_predic,vout = havok.predict(new_data[start_ind-stackmax:start_ind,0:1],test_len)
    errors[ite,:test_len] = x_test[:test_len,0]
    errors[ite,test_len:] = y_predic[:test_len,0]
    #print("mean error",np.mean(np.abs(errors[ite,:500,:]),axis=0))
#rmse = np.sqrt(np.mean((errors[:,:,1]-errors[:,:,0])**2,axis=0))
#nmse = np.sum((errors[:,:,1]-errors[:,:,0])**2,axis=0)/np.sum( (errors[:,:,1]-np.mean(errors[:,:,1],axis=0))**2 )
np.savetxt("./result/sunspot/error_x.txt",errors)

plt.figure(3)
plt.subplot(1,2,1)
plt.plot(t_test[:test_len],y_predic[:test_len,0],label="predict x")
plt.plot(t_test[:test_len],x_test[:test_len,0],label="x")
plt.legend()
plt.subplot(1,2,2)
plt.plot(t_test[:test_len],vout[:],label="predict v")
plt.legend()
