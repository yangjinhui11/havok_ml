import numpy as np
from .utils import SINDy
from control.matlab import ss,lsim,isctime
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pylab as plt
from .predict_vr import predict_vr
import scipy as sp
import scipy.linalg as la
class HAVOK:
    
    def __init__(self,rmax,q,predInd=2,isDiscrete=False,lamb = 0.025,
        regressor=None, predictor_vr=None):
        # The max r truncation
        # q The columns of H matrix 
        self.rmax = rmax
        self.q = q
        self.isDiscrete = isDiscrete
        self.predInd = predInd # < q
        self.lamb = lamb
        self.regressor=regressor
        self.predictor=predictor_vr
    @classmethod
    def getPredictIndx(cls,M,q,predInd):
        return np.arange(0,M*q,predInd)
    def fit(self,xdata,dt):
        N = len(xdata)
        M = 1 if len(xdata.shape)==1 else xdata.shape[1]
        q = self.q
        xdat = xdata.reshape(N,M)
        H = np.zeros((q*M,N-q+1))
        for k in range(q):
            for j in range(M):
                if(k==q-1):
                    H[k+j*q,:] = xdat[k:,j]
                else:
                    H[k+j*q,:] = xdat[k:-q+k+1,j]
        
        #for j in M:
        #    H[]
        
        r = int(self.rmax)
        U,S,V = np.linalg.svd(H, full_matrices=False)
        #fft filter for vr
        #self.fft_filter(V[r-1,:],0.9) 

        print("S diag",S[:r-1],S[r-1])
        
        if self.isDiscrete:
            differentiation_method='discrete'
        else:
            differentiation_method='derivative'

        sindy = SINDy(differentiation_method = differentiation_method)
        sindy.fit(V[:r,:],1,t=dt,coefficient_threshold =self.lamb)
        x = sindy.RHS[1:,]
        dx = sindy.LHS
        Xi = sindy.Xi[1:,:-1].T
        error = sindy.error
        print("sindy error",error[:r-1],error[r-1])
        A = Xi[:,:-1]
        B = Xi[:,-1].reshape(r-1,1)
        print(V.shape)
        #print("A",A)
        print("B",B)
        print("train Vr using sklearn")
        #predictor = predict_vr(xdat[start:end:2,:],V[r-1,::2]) # for lorenz
        print()
        #fft filter
        self.L = HAVOK.getPredictIndx(M,q,self.predInd) # predict index
        print("self.L",self.L)
        if self.predictor == None:
            self.predictor = predict_vr(H[self.L,:-1].T,V[r-1,1:],regressor=self.regressor) 
        else:
            self.predictor.fit(H[-1,:-1],V[r-1,1:])
            
        self.U,self.S,self.V,self.A,self.B,self.x,self.xdat,self.M,self.N= U,S,V,A,B,x,xdat,M,N
        self.dt = dt
        self.H = H
    def fft_filter(self,x_in,coef):
        plt.figure(4)
        plt.subplot(1,2,1)
        plt.plot(x_in,label="before fft filter")
        #fft filter
        fft_vr = np.fft.rfft(x_in)
        plt.subplot(1,2,2)
        plt.plot(fft_vr,label="fft vr{}".format(int(coef*len(fft_vr))))
        plt.legend()
        fft_vr[int(len(fft_vr)*coef):] = 0 
        x_back = np.fft.irfft(fft_vr)
        x_in[:] = 0
        x_in[:len(x_back)] = x_back
        #
        plt.subplot(1,2,1)
        plt.plot(x_in,label="After filter")
        
        plt.legend()
    def construct_lsim(self):
        A,B,dt= self.A,self.B,self.dt
        
        n_states = A.shape[0]
        n_inputs = B.shape[1]
        M = np.block([[A * dt, B * dt, np.zeros((n_states, n_inputs))],
                         [np.zeros((n_inputs, n_states + n_inputs)),
                          np.identity(n_inputs)],
                         [np.zeros((n_inputs, n_states + 2 * n_inputs))]])
        expM = sp.linalg.expm(M)
        Ad = expM[:n_states, :n_states]
        Bd1 = expM[:n_states, n_states+n_inputs:]
        Bd0 = expM[:n_states, n_states:n_states + n_inputs] - Bd1
        self.Bd1 = Bd1
        self.Bd0 = Bd0
        self.Ad = Ad
        #print("construct",Bd1.shape,Bd0.shape,Ad.shape)
        #print("expM[:n_states, n_states:n_states + n_inputs]",expM[:n_states, n_states:n_states + n_inputs])
        #print("Bd1",Bd1)
        #print("Bd0",Bd0)
        #print("Ad",Ad)
        #print("A",A)
    def lsim_one_sep(self,v0,U0,U1):
        if self.isDiscrete: # 
            part1 = np.dot(self.A,v0)
            part2 = np.dot(self.B,U1)
            return part1+part2
        else:
            Ad,Bd0,Bd1 = self.Ad,self.Bd0,self.Bd1
            return np.dot(Ad, v0) + Bd0*U0 +  Bd1*U1
    def multi_predict(self,xin,steps):
        if len(xin.shape)==1:
            xin = xin.reshape(1,-1)
        N,q = xin.shape
        output = np.empty((N,steps))
        for n in range(N):
            print("havok predict,",n)
            output[n,:] = self.predict(xin[n,:],tstep)
        return output
    def predict(self,xin,tstep,Vr_in = []):
        q = self.q
        xlen = len(xin)
        #print(xin.shape)
        M = 1 if len(xin.shape)==1 else xin.shape[1]
        #print("predict xin",xlen,M)
        #assert M ==1
        assert xlen >= q and M ==self.M
        if len(xin.shape)==1:
            xin=xin.reshape(-1,1)
        # continue system
        # dv/dt = Av+Bv_r
        if not self.isDiscrete:
            self.construct_lsim()
        else:
            # discrete system
            # V^1 = Av+Bv_r
            pass
        H = np.zeros(M*q)
        for i in range(M):
            H[i*q:(i+1)*q] = xin[-q:,i]
        A,B,r = self.A,self.B,self.rmax
        US = np.dot(self.U,np.diag(self.S))
        US_inv = np.linalg.inv(US)
        US_r = US[:,:r-1]
        #US_r = US[:,:r]
        xout=[]
        for i in range(q):
            xout.append(xin[-q+i,:].tolist())
        V = US_inv.dot(H)
        vr0 = V[r-1]
        #print("V",(V[:r-1],self.V[:r-1,-1]))
        xy_pred = H
        
        #vr0 = V[r]
        V = V[:r-1].reshape(-1,1)
        vout = [vr0]
        for k in range(tstep): # tstep
            #vr = self.predictor.predict(np.array(xout[self.predInd+k:self.predInd+k+2]))
            xy_pred = xy_pred.reshape(1,-1)
            if len(Vr_in)>0:
                vr1 = Vr_in[k]
            else:
                vr1 = self.predictor.predict(xy_pred[:,self.L])[0]
                #vr1 = 0#self.predictor.predict(xy_pred[:,self.L])[0]
            V = self.lsim_one_sep(V,vr0,vr1).reshape(-1,1)
            xy_out = US_r.dot(V)+US[:,r-1:r]*vr1
                
            xy_out[:-1,0] = xy_pred[0,1:]
            xy_pred = xy_out
            xout.append(xy_pred[q-1::q,0].tolist())
            vout.append(vr1)
            vr0 = vr1
        return np.array(xout)[q:],vout[1:]
    def test(self,tspan=[]):
        U,S,V,A,B,x,xdat,M,N,q= self.U,self.S,self.V,self.A,self.B,self.x,self.xdat,self.M,self.N,self.q
        H =self.H
        N = int(len(x[0,:]))
        L = np.arange(N)
        r = self.rmax
        US = np.dot(U,np.diag(S))
        if self.isDiscrete:
            sys = ss(A,B,np.eye(r-1),0*B,dt=self.dt)
        else:
            sys = ss(A,B,np.eye(r-1),0*B)
        
        yout,ts,xout = lsim(sys,x[-1,L],L*self.dt,x[:-1,0])
        if len(tspan)==0:
            tspan=ts
        ##################################################
        print("havok test",N,len(tspan))
        self.predictor.test(tspan)
        #print("lenth of xout",xout.shape,len(L),yout.shape)
        #print("x[:-1,0]",x[:-1,0])
        #print("lsim predict xout",xout[:2,:])
        #print("lsim predict yout",yout[:2,:])
        yout = yout[1:,:]
        xout = xout[1:,:]
        L = L[:-1]
        N = N-1
        ######################################################
        xin = H[:,0].reshape(-1,M)
        xout,vout = self.predict(xin,N,x[-1,0:])
        US_inv = np.linalg.inv(US)
        #print("H[:,0]",US_inv.dot(H[:,1])[:])
        #print("recover",V[:,1])
        #print("x0",x[:,0])
        #print("vout",vout[:10])
        #print("vr",x[-1,:10])
        ############################
        #print(U.shape,S.shape,US.shape)
        US_r = US[:,:r]
        yout = np.hstack((yout,x[-1:,L].T))
        xy_pred = np.dot(US_r,yout.T)
        print(xout.shape)
        for i in range(M):
            score_x = r2_score(xy_pred[(i+1)*q-1,L],xdat[L+q-1,i])
            score_x2 = r2_score(xout[:,i],xdat[L+q-1,i])
            rmse_x = mean_squared_error(xy_pred[i*q,L],xdat[L,i])
            print("Predict {} variable score".format(i),score_x,score_x2,rmse_x)
        
        #L = L[tspan[L]>=50 ]
        
        L2 = L
        print(tspan[0],tspan[-1],tspan[L2[0]],tspan[L2[-1]])
        plt.figure(2)
        grid = plt.GridSpec(2,4,wspace=0.5,hspace=0.2)
        ax=plt.subplot(grid[0,0])
        plt.title("Matrix A")
        plt.imshow(A)
        A_x = np.arange(len(A))
        plt.xticks(A_x)
        #### offset box ###################
        #from matplotlib.offsetbox import TextArea,AnnotationBbox
        #offsetbox = TextArea(r"$\times v_{r-1}$", minimumdescent=False)
        #xy = (A_x[-1],A_x[int(len(A)/2)])        
        #ab = AnnotationBbox(offsetbox, (1.5,0.5),
        #                xybox=(20, 0),
        #                xycoords='axes fraction',
        #                boxcoords="offset points",
        #                arrowprops=dict(arrowstyle="->"))
        #ax.add_artist(ab)
        ####################

        plt.colorbar(orientation='horizontal')
        plt.subplot(grid[0,1])
        plt.title("Vecotr B")
        plt.imshow(B)
        plt.xticks([])
        #plt.colorbar()
        plt.colorbar(orientation='horizontal')
        plt.subplot(grid[0,2:])
        plt.plot(tspan[L],V[0,L],linewidth=1.5,label=r"Actural $V_1$")
        plt.plot(tspan[L2],yout[L2,0],'-.',linewidth=1.5,label=r"Predicted $V_1$")
        plt.ylabel(r'$v_1$',fontsize=12)
        plt.xlabel('time',fontsize=12)
        plt.legend(loc=1)
        plt.subplot(grid[1,2:])
        plt.plot(tspan[L],x[-1,L],'r',linewidth=1.5)
        #plt.ylim(-0.025,0.024)
        plt.xlabel("time",fontsize=12)
        plt.ylabel(r"$v_r$",fontsize=12)
        plt.subplot(grid[1,:2])
        plt.plot(tspan[L+q-1],xdat[L+q-1,0],linewidth=1.5,label=r"Actural $X(t)$")
        plt.plot(tspan[L+q-1],xy_pred[-1,L],'-.',linewidth=1.5,label=r"Predicted $X(t)$")
        #plt.plot(xout[L,-1],linewidth=2.5,label="predict X2")
        plt.ylabel(r'$x(t)$',fontsize=12)
        plt.xlabel('time',fontsize=12)
        plt.legend(loc=1)
        if M > 1:
            plt.subplot(grid[2,:2])
            plt.plot(tspan[L],xdat[L+q-1,1],linewidth=2.5,label="real Y")
            plt.plot(tspan[L],xy_pred[q,L],linewidth=2.5,label="real Y")
            plt.legend()
        if M > 2:
            plt.subplot(grid[2,2:])
            plt.plot(tspan[L],xdat[L+q-1,2],linewidth=2.5,label="real Z")
            plt.plot(tspan[L],xy_pred[2*q,L],linewidth=2.5,label="real Z")
            plt.legend()
        plt.show()
            
