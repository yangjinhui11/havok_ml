import numpy as np
import matplotlib.pylab as plt

#errors = np.loadtxt("./result/lorenz/error_x.txt")
errors = np.loadtxt("./result/lorenz/error_x_20_9.txt")
test_len = int(errors.shape[1]/2)
ite = errors.shape[0]
L= np.arange(9,test_len,10)

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
plt.plot(x_real[:,L[9]],label="Actural x")
plt.plot(x_pred[:,L[9]],label="10 step prediction")
plt.xlabel('N',fontsize=12)
plt.legend()
plt.subplot(2,2,3)
plt.plot(x_real[:,L[19]],label="Actural x")
plt.plot(x_pred[:,L[19]],label="20 step prediction")
plt.xlabel('N',fontsize=12)
plt.legend()
plt.subplot(2,2,4)
plt.plot(x_real[:,L[49]],label="Actural x")
plt.plot(x_pred[:,L[49]],label="50 step prediction")
plt.xlabel('N',fontsize=12)
plt.legend()
plt.figure(1)
print(L[:])
plt.plot([1,2,3,4,5,6,7,8,9,10],rmse[L[:10]],label="lorenz rmse_x")
plt.xticks([0,2,4,6,8,10])
plt.xlabel('N',fontsize=12)
plt.legend()
plt.figure(2)
plt.subplot(2,1,1)
plt.title("MultiStep prediction of Lorenz system")
plt.plot(x_real[500,L],label="real x")
plt.plot(x_pred[500,L],label="prediction x")
plt.legend()
plt.subplot(2,1,2)
plt.title("Error")
plt.plot(x_real[500,L]-x_pred[500,L])
plt.show()
