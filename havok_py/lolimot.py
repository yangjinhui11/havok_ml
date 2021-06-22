import numpy as np
from scipy.io import loadmat
import matplotlib.pylab as plt
def genMackey():
    metdata = loadmat("./DATA/mackeyglass.mat")
    xdat = metdata['xdat']
    dt = metdata["dt"][0][0]
    tspan = metdata['tspan'][0]
    dataInput = []
    dataTarget = []
    print(xdat.shape)
    for t in range(1000,len(xdat)):
        dataInput.append([xdat[t-24,0],xdat[t-18,0],xdat[t-12,0],xdat[t-6,0]])
        dataTarget.append(xdat[t])
    xdata =  np.array(dataInput)
    ydata = np.array(dataTarget)
    train_len = int(len(xdata)*0.7)
    x_train = xdata[:train_len,:]
    y_train = ydata[:train_len,:]
    x_test = xdata[train_len:,:]
    y_test = ydata[train_len:,:]
    return x_train, y_train,x_test,y_test
def genSunspot():
    metdata = np.loadtxt('./DATA/SN_ms_tot_V2.0.csv',dtype='float',delimiter=';')
    print(metdata.shape)
    xdat = metdata[:,3].reshape(-1,1)
    N = len(xdat)
    xmax = np.max(xdat)
    xmin = np.min(xdat)
    xdat = (2*xdat-(xmax+xmin))/(xmax-xmin) # normalize
    months = np.arange(N)
    dataInput = []
    dataTarget = []
    print(xdat.shape)
    L = range(1030,N-238)
    print("months",N,metdata[L[0],0:2],metdata[L[-1],0:2],len(L))
    for t in range(1000,len(xdat)):
        dataInput.append([xdat[t-5,0],xdat[t-4,0],xdat[t-3,0],xdat[t-2,0],xdat[t-1,0]])
        dataTarget.append(xdat[t])
    xdata =  np.array(dataInput)
    ydata = np.array(dataTarget)
    train_len = int(len(xdata)*0.5)
    x_train = xdata[:train_len,:]
    y_train = ydata[:train_len,:]
    x_test = xdata[train_len:,:]
    y_test = ydata[train_len:,:]
    return x_train, y_train,x_test,y_test
def getNino():
    metdata = np.loadtxt('./DATA/soda_train_nino_s.txt')
    print(metdata.shape)
    xdat = metdata[:,3].reshape(-1,1)
    N = len(xdat)
    xmax = np.max(xdat)
    xmin = np.min(xdat)
    xdat = (2*xdat-(xmax+xmin))/(xmax-xmin) # normalize
    months = np.arange(N)
    dataInput = []
    dataTarget = []
    print(xdat.shape)
    L = range(N)
    for t in range(100,len(xdat)):
        dataInput.append([xdat[t-10,0],
                          xdat[t-8,0],
                          xdat[t-6,0],
                          xdat[t-4,0],
                          xdat[t-2,0],
                        ])
        dataTarget.append(xdat[t])
    xdata =  np.array(dataInput)
    ydata = np.array(dataTarget)
    train_len = int(len(xdata)*0.5)
    x_train = xdata[:train_len,:]
    y_train = ydata[:train_len,:]
    x_test = xdata[train_len:,:]
    y_test = ydata[train_len:,:]
    return x_train, y_train,x_test,y_test
# locally linear neurofuzzy model
class LLNModel():
    def __init__(self,maxM = 20,alpha =1e-10):
        self.zarib = 0.33 #standard deviation per extention of the hyperrectangle
        self.LLM_NUM = maxM
        self.alpha = alpha
        self.eps = alpha
        print("LLNModel, M max")
        pass

    def LLM(self,xin,target,X,c,sigma):
        # X=N*q X=N*(q+1),C=M*q,sigma=M*q
        N,q = xin.shape
        M,q = c.shape
        # c M*q 
        #sigma M*q
        #print("lolimot",xin.shape,c.shape,sigma.shape,M)
        input_expand = np.zeros((q,N,M))
        c_expand = np.zeros((q,N,M))
        sigma_expand = np.zeros((q,N,M))
        for i in range(M):
            input_expand[:,:,i] = xin.T
          
        for i in range(N):
            c_expand[:,i,:] = c.T
            sigma_expand[:,i,:] = sigma.T
        power = ((input_expand-c_expand)/sigma_expand)**2 # q*N*M
        
        powered = np.sum(power,axis = 0) # N*M
        MSF_temp = np.exp(-1./2*powered.T)  # M*N
        phi = MSF_temp/np.sum(MSF_temp,axis=0) # M*N
        
        w = np.empty((q+1,M))
        y = np.empty((N,M))
        e = np.empty((N,M))
        I = np.empty((M))
        out = np.empty(N)
        
        for i in range(M):
            #Q = np.diag(phi[i,:])
            #XTQ = np.dot(X.T,Q)
            XTQ = X.T*phi[i,:]
            XTQX = XTQ.dot(X)
            U,S,Vh = np.linalg.svd(XTQX)
            #print("lolimot",S)
            r = len(S[np.where(S>=self.eps)])
            U = U[:,:r]
            V = Vh.conj().T[:,:r]*(1./S[:r])
            inv_XTQX = V.dot(U.conj().T)
            #print("cond number",i,np.linalg.cond(XTQX),r)
            #w[:,i:i+1] = np.linalg.inv(XTQX+np.eye(q+1)*self.alpha).dot(XTQ).dot(target)
            w[:,i:i+1] = inv_XTQX.dot(XTQ).dot(target)
            y[:,i] = X.dot(w[:,i])
            e[:,i] = target[:,0]-y[:,i]
            I[i] = np.sum(phi[i,:]*(e[:,i]**2))
        out = np.sum(phi*y.T,axis=0)
        return w,I,out
    def lolimot(self,xin,target,X):
        pass
    def fit(self,xin,target):
        N,q = xin.shape
        M = 1
        target = target.reshape(-1,1)
        print("N,q",N,q)
        a = np.min(xin,axis = 0)
        b = np.max(xin,axis = 0)
        b = b.reshape(M,-1)
        a = a.reshape(M,-1)
        c = (a+b)/2.
        sigma = (b-a)*self.zarib
        #c = c.reshape(M,-1)
        #sigma = sigma.reshape(M,-1)
        X = np.hstack((np.ones((N,1)),xin)) # input vector for network----X=[]N*(p+1)
        w,I,out = self.LLM(xin,target,X,c,sigma)
        s = np.argmax(I)
        r = I[s]
        IError2 = np.zeros(q)
        for i in range(1,self.LLM_NUM):
            M+=1
            print("Level of LLM = ",M)
            
            #找到误差最小的a,b,c
            #
            emin = 1e10
            b_temp = b.copy() #max
            a_temp = a.copy() #min
            b_temp = np.vstack((b_temp,b[s:s+1,:]))
            a_temp = np.vstack((a_temp,a[s:s+1,:]))
            for j in range(q):
                b_temp[s,j] = (a[s,j]+b[s,j])/2.
                a_temp[-1,j] = (a[s,j]+b[s,j])/2.
                # The end of updating intervals
                c_temp = (a_temp+b_temp)/2.
                sigma_temp = (b_temp-a_temp)*self.zarib
                w,I,out = self.LLM(xin,target,X,c_temp,sigma_temp)
                error = out[:]-target[:,0]
                IError2[j] = np.sqrt(np.average(error**2))
                if IError2[j] < emin:
                    a_min = a_temp.copy()
                    b_min = b_temp.copy()
                    c_min = c_temp.copy()
                    sigma_min = sigma_temp.copy()
                    I_min = I.copy()
                    emin = IError2[j]
                    out_min = out.copy()
                    w_min = w.copy()
            print("min error",emin)
            a = a_min
            b = b_min
            c = c_min
            sigma =sigma_min
            s = np.argmax(I)
            r = I[s]
            self.w = w_min
            self.c = c
            self.sigma = sigma
        #plt.plot(out_min)
        #plt.plot(target)
        #plt.show()
    def one_step_predict(self,xin):
        if len(xin.shape)==1:
            xin = xin.reshape(1,-1)
        N,q = xin.shape
        M,q = self.c.shape
        out = np.zeros((N,1))
        X = np.hstack((np.ones((N,1)),xin))
        y = X.dot(self.w)
        
        mu = np.empty((N,M))
        phi = np.empty((N,M))
        for i in range(M):
            power = ((xin-self.c[i,:])/self.sigma[i,:])**2*(-0.5)
            mu[:,i] = np.exp(np.sum(power,axis=1))
        sum_mu = np.sum(mu,axis=1)
        for j  in range(M):
            phi[:,j] = mu[:,j]/sum_mu
        out = np.sum(phi*y,axis=1)
        return out
    def predict(self,xin,steps = 1):
        if len(xin.shape)==1:
            xin = xin.reshape(1,-1)
        N,q = xin.shape
        output = np.empty((N,steps))
        for step in range(steps):
            output[:,step] = self.one_step_predict(xin)
            xin = np.hstack((xin[:,1:],output[:,step:step+1]))
        return output
    def forward(self,x):
        pass
if __name__=='__main__':
    #dinput,dtarget,xtest,ytest = genMackey()
    dinput,dtarget,xtest,ytest = genSunspot()
    #dinput,dtarget,xtest,ytest = getNino()
    net = LLNModel(maxM = 40)
    net.fit(dinput,dtarget)
    out = net.predict(xtest)
    rmse = np.sqrt(np.mean((out-ytest)**2))
    print("test_error",rmse)
    
    plt.plot(ytest)
    plt.plot(out)
    plt.show()
