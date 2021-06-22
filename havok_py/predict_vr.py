import numpy as np
#from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

import matplotlib.pylab as plt
#from axes_zoom_effect import zoom_effect02 ,zoom_effect01
class predict_vr:
    def __init__(self,xdata,vr,regressor=None):
        
        print(xdata.shape,vr.shape)
        data_len = xdata.shape[0]
        #分割测试数据，80%训练，20%测试
        train_coef = 0.8
        ss = StandardScaler()
        vr = ss.fit_transform(vr.reshape(-1,1)).reshape(-1)
        self.ss = ss

        train_len = int(train_coef*data_len)
        print("predict vr",train_len,data_len)
        x_train= xdata[:train_len,:]
        y_train= vr[:train_len]
        x_test = xdata[train_len:,:]
        y_test = vr[train_len:]
        
            #rfr = LLNModel(maxM = 10,alpha = 1e-4)
        if regressor==None:
            rfr =  ExtraTreesRegressor(n_estimators=200,random_state=20)
        else:
            rfr = regressor
        print("The regressor {0} predict vr ".format(type(rfr).__name__))
            #rfr =  RandomForestRegressor(n_estimators=300,random_state=20)
            #rfr = MLPRegressor(hidden_layer_sizes=(50,100,10),activation='relu') # for mackeyglass
            
        # train
        rfr.fit(x_train,y_train)
        self.predictor = rfr

        self.x_test = x_test
        self.y_test = y_test
        self.train_len = train_len
        #
        #print("feature importances",rfr.feature_importances_)
        # test
    def test(self,tspan=[]):
        
        x_test = self.x_test
        y_test = self.y_test
        train_len = self.train_len
        N = len(y_test)
        if len(tspan)==0:
            tspan = np.arange(0,N)
        else:
            tspan = tspan[train_len:train_len+N]
        #print("predict vr test",len(tspan[train_len]),N)
        rfr_y_predict = self.predictor.predict(x_test)
        print("rfr_y_predict",rfr_y_predict.shape,x_test.shape)
        plt.figure(1)
        ax1 = plt.subplot(2,1,2)
        ax2 = plt.subplot(2,1,1)
        plt.title("{} to predict Vr".format(type(self.predictor).__name__))
        ax2.plot(tspan,y_test,label=r"Actual $V_r$")
        ax2.plot(tspan,rfr_y_predict,label=r"Predicted $V_r$")
        ax2.set_ylabel(r"$V_r$",fontsize=12)
        
        #for lorenz system
        #N1 = 91
        #N2 = 92
        #for Mackey-Glass
        N1 = 3200
        N2 = 3300
        #
        #for sunspot
        N1 =850
        N2 =900
        ax1.set_xlim(N1,N2)
        
        ax1.plot(tspan,y_test,label=r"Actual $V_r$")
        ax1.plot(tspan,rfr_y_predict,label=r"Predicted $V_r$")
        ax1.set_xlabel("time",fontsize=12)
        ax1.set_ylabel(r"$V_r$",fontsize=12)
        #zoom_effect01(ax1,ax2,N1,N2)
        ax1.legend()
        ax2.legend()
        print('rfr test  score',r2_score(rfr_y_predict,y_test))
        
    def predict(self,x):
        out =  self.predictor.predict(x)
        result = self.ss.inverse_transform(out)
        return result
if __name__=='__main__':
    boston= load_boston()
    xdata = boston.data
    y = boston.target
    print(xdata.shape,y.shape)

