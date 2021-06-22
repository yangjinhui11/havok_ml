import numpy as np
import matplotlib.pylab as plt

dt = 0.02
idt = int(1./dt)

def getErrors(fname):
    errors = np.loadtxt(fname)
    print(errors.shape)
    test_len = int(errors.shape[1]/2)
    ite = errors.shape[0]
    L= np.arange(test_len)[idt-1::idt]
    print(test_len,L)
    x_real = errors[:,:test_len]
    x_pred = errors[:,test_len:]

    rmse = np.sqrt(np.mean((x_real-x_pred)**2,axis=0))
    nmse = np.zeros(test_len)
    for i in range(test_len):
        part1 = np.sum((x_real[:,i]-x_pred[:,i])**2)
        part2 = np.sum((x_real[:,i]-np.mean(x_real[:,i]))**2)
        nmse[i] = part1/part2
    error = x_real-x_pred
    std = np.std(error,axis=0)
    return x_pred,x_real,rmse,nmse,std,L
x_pred,x_real,rmse,nmse,std,L = getErrors('./result/sunspot/error_x.txt')

print("nmse",nmse[L])
print("rmse",rmse[L])
print("std",std[L])
plt.figure(0)
plt.subplot(2,2,1)
plt.plot(x_real[:,L[0]],label="Actural x")
plt.plot(x_pred[:,L[0]],label="1 step prediction")
plt.xlabel('N',fontsize=12)
plt.legend()
plt.subplot(2,2,2)
plt.plot(x_real[:,L[3]],label="Actural x")
plt.plot(x_pred[:,L[3]],label="4 step prediction")
plt.xlabel('N',fontsize=12)
plt.legend()
plt.subplot(2,2,3)
plt.plot(x_real[:,L[7]],label="Actural x")
plt.plot(x_pred[:,L[7]],label="8 step prediction")
plt.xlabel('N',fontsize=12)
plt.legend()
plt.subplot(2,2,4)
plt.plot(x_real[:,L[11]],label="Actural x")
plt.plot(x_pred[:,L[11]],label="12 step prediction")
plt.xlabel('N',fontsize=12)
plt.legend()
plt.figure(1)
print(L[:])

#_,_,rmse_vr0,_,_,_ = getErrors('./result/sunspot/error_x_5_5_vr=0.txt')
_,_,rmse_140_7,_,_,_ = getErrors('./result/sunspot/error_x_140_7.txt')
#_,_,rmse_75,_,_,_ = getErrors('./result/sunspot/error_x_7_5.txt')

#plt.plot(L[:]+1,rmse[L],label="rmse lolimot")
#plt.plot(L[:]+1,rmse_vr0[L[:40]],label="rmse r=5 q=5 vr=0")
plt.plot(rmse_140_7[L],label="rmse r=7 q=140")
#plt.plot(L[:40]+1,rmse_75[L[:40]],label="rmse r=7 q=5")
#plt.xticks([0,2,4,6,8,10])
plt.xlabel('N',fontsize=12)
plt.legend()
plt.figure(2)
plt.subplot(2,1,1)
plt.title("MultiStep prediction of Lorenz system")
plt.plot(x_real[1,L],label="real x")
plt.plot(x_pred[1,L],label="prediction x")
plt.legend()
plt.subplot(2,1,2)
plt.title("Error")
plt.plot(x_real[1,L]-x_pred[1,L])
plt.show()
