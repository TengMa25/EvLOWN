import random
from scipy.fftpack import fft
from scipy.signal import hilbert, find_peaks
from scipy.integrate import odeint
import numpy as np
from sympy import symbols, sin, cos, pi, integrate, lambdify,trigsimp
import matplotlib.pyplot as plt

def predict_values(X, w):
    # X will be  samples(N) * features(D)
    # w will be  features(D) * 1
    predictions = np.matmul(X,w) # samples(N) * 1
    return predictions

def rho_compute(dot,library,Xi,j):
    # dot will be samples(N) * 2(A,beta) 
    # library will be samples (N) * 2(A,beta)* features(D) 
    # Xi will be features(D)
    # j will be the idx of feature
    rho = 0
    for i in range(2):
        Library_k = np.delete(library[:,i,:], j, 1)
        Xi_k = np.delete(Xi,j)
        predict_k = predict_values(Library_k,Xi_k)
        rho += np.sum(library[:,i,j]*(dot[:,i]-predict_k))
    return rho

def z_compute(library):
    # library will be samples (N) * 2(A,beta)* features(D) 
    feature_length = np.shape(library)[2]
    z_vector = np.zeros(feature_length) # features(D) * 1
    for i in range(2):
        z_vector += np.sum(library[:,i,:]*library[:,i,:],axis = 0)
    return z_vector


def coordinate_descent(dot,library,alpha=1e-5,tolerance=1e-12):
#     import random.choice
    feature_length = np.shape(library)[2]
    N = np.shape(library)[0]
    # Initialise weight vector
    Xi = np.zeros(feature_length)
    
    z = z_compute(library)

    invalididx = np.where(np.abs(z)<1e-8)
    Xi[invalididx] = 0
    effidx = np.where(z>=1e-8)[0]
    max_step = 100.
    iteration = 0
    while(max_step > tolerance):
        iteration += 1
        #print("Iteration (start) : ",iteration)
        old_weights = np.copy(Xi)
#         print("\nOld Weights\n",old_weights)
        j = random.choice(effidx)
        rho_j = rho_compute(dot,library,Xi,j)
        # print(rho_j)
        if rho_j < -alpha*N:
            Xi[j] = (rho_j + (alpha*N))/z[j]
        elif rho_j >= -alpha*N and rho_j <= alpha*N:
            Xi[j] = 0
        elif rho_j > alpha*N:
#             delta = Xi[j] - 
            Xi[j] = (rho_j - (alpha*N))/z[j]
#         print("\nNew Weights\n",Xi)
        
        step_sizes = abs(old_weights - Xi)
        #print("\nStep sizes\n",step_sizes)
        max_step = step_sizes.max()
        #print("\nMax step:",max_step) 
    return Xi, iteration, max_step

def Xi(y,library, S):
    # 
    feature_length = np.shape(library)[2]
    N = np.shape(library)[0]
    X = np.copy(library)
    scale = np.ones(feature_length)
    # coef = np.zeros(feature_length)
    for i in range(feature_length):
        if i not in S:
            X[:,0,i] = np.zeros(N)
            X[:,1,i] = np.zeros(N)
        else:
            scale[i] = max(np.max(np.abs(X[:,0,i])),np.max(np.abs(X[:,1,i])))
            X[:,0,i] = X[:,0,i]/scale[i]
            X[:,1,i] = X[:,1,i]/scale[i]
            X[:,0,i] = X[:,0,i]
            X[:,1,i] = X[:,1,i]
            

    coef = np.dot(np.linalg.pinv(np.dot(X[:,0,:].T,X[:,0,:]) + np.dot(X[:,1,:].T,X[:,1,:])), (np.dot(X[:,0,:].T, y[:,0]) +np.dot(X[:,1,:].T, y[:,1])))
    result = np.zeros(feature_length)
    coef = coef/scale
    for i in range(feature_length):
        if i in S:
            result[i] = coef[i]
    
    return result

def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n

def corr(a,b):
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    cov = np.sum((a)*(b))
    a_norm = np.linalg.norm(a, ord = 2)
    b_norm = np.linalg.norm(b, ord = 2)
    return np.abs(cov/(a_norm*b_norm))


def OMP(y, library, sparse_threshold = 1e-3 ,stop_tolerance = 1e-4, step_tolerance = 1e-5, sparse_max = 6,smooth_window = 1,w_A2b = 2):
    feature_length = np.shape(library)[2]
    N = np.shape(library)[0]
    coef = np.zeros(feature_length)
    y_hat = np.copy(y)
    norm_A2b = np.max(np.abs(y[:,0]))/np.max(np.abs(y[:,1]))
    norm = np.array([norm_A2b,1])
    S = []
    r0 = np.sum((y_hat[:,0])**2 + (y_hat[:,1])**2)/N
    r = r0
    loss = [1]
    for n in range(sparse_max):
        
        Cdots = []
        for i in range(feature_length):
            if i in S:
                Cdots.append(0)
            else:
                new_dot = 0
                for j in range(2):
                    library_norm = np.max(np.abs(library[:,j,i]))
                    # print(library_norm)
                    if library_norm < 1e-5:
                        new_dot += 0
                    else:
                        if j == 0:
                            new_dot += corr(y_hat[:,j],library[:,j,i])*w_A2b
                        else:
                            new_dot += corr(y_hat[:,j],library[:,j,i])
                Cdots.append(new_dot)
        
        # print(Cdots)
        if (np.array(Cdots) < 1e-10).all():
            break
        Idx = np.argmax(Cdots)
        S.append(Idx)

        coef_new = Xi(y,library,S)
        r_new = np.sum(((y[:,0] - np.dot(library[:,0,:], coef_new)))**2 + ((y[:,1] - np.dot(library[:,1,:], coef_new)))**2)/N
        loss.append((r_new)/r0)
        plt.figure()
        plt.plot(y[:,0])
        plt.plot(np.dot(library[:,0,:], coef_new))
        plt.ylim(-1e-3,1e-3)
        coef = coef_new
        print((r_new)/r0)
        print((r-r_new)/r0)
        if (r_new)/r0<stop_tolerance or (r-r_new)/r0 < step_tolerance:  
            break
        r = r_new

        y_hat[:,0] = y[:,0] - np.dot(library[:,0,:], coef)
        y_hat[:,1] = y[:,1] - np.dot(library[:,1,:], coef)
    print(S)
    contributions = np.zeros((feature_length,2))
    for i in range(feature_length):
        A_i = coef[i] * library[:,0,i]
        beta_i = coef[i] * library[:,1,i]
        contribution_i = (np.sum(A_i**2/(y[:,0])**2)/N, np.sum(beta_i**2/(y[:,1])**2)/N)
        contributions[i,0] = contribution_i[0]
        contributions[i,1] = contribution_i[1]

    contributions[:,0] = contributions[:,0]/np.max(contributions[:,0])
    contributions[:,1] = contributions[:,1]/np.max(contributions[:,1])
    contributions = contributions[:,0] + contributions[:,1]
    print(contributions)
    for i in range(feature_length):
        if (contributions[i] < sparse_threshold):
            if i in S:
                S.remove(i)
    
    coef = Xi(y,library,S)

        
        
    y_predict = np.zeros([N,2])
    y_predict[:,0] = np.dot(library[:,0,:], coef)
    
    y_predict[:,1] = np.dot(library[:,1,:], coef)


    return coef, y_predict


class WeakNO:
    def __init__(self, x_dims,library,library_name):
        self.dims = x_dims # Degree of Freedom of weakly nonlinear oscillator
        self.evolutions = None
        self.t_evolutions = None
        self.frequencys = np.zeros(x_dims)
        self.library = library
        self.library_name = library_name
        self.data = None
        self.t = None
        self.length = None
        self.Phi = None
        self.Xi = np.zeros([self.dims,len(self.library)])
        self.predict = None

    def __str__(self):
        s = ""
        varble = []
        for d in range(self.dims):
            varble.append("x%d"%d)
            varble.append("x%d'"%d)
        for d in range(self.dims):
            s_sub = "x%d'' + "%d
            for i in range(len(self.Xi[d])):
                if i == 0:
                    xi = self.frequencys[d]**2+self.Xi[d][i]
                    s_sub += "%e%s + "%(xi,self.library_name[i](varble))
                elif np.abs(self.Xi[d][i])>1e-12:
                    s_sub += "%e%s + "%(self.Xi[d][i],self.library_name[i](varble))
            s_sub = s_sub[:-3]
            s_sub += ' = 0\n'
            s+=s_sub
        return s
            
                
                
                        
                

    def Get_frequency(self,X,t):
        # X: [length*dims]
        L = len(t)
        self.data = X
        self.t = t
        N = np.power(2, np.ceil(np.log2(L)))
        Fs = 1/np.mean(np.diff(t)) 
        for i in range(self.dims):
            fft_x = np.abs(fft(X[:,i],n = int(N)))[range(int(N/2))]
            Freq = np.arange(int(N/2))*Fs/N
            self.frequencys[i] = Freq[np.argmax(fft_x)]*2*np.pi
        

    def Get_Evolution(self,smooth_window,height = 0):
        dt = self.t[1]-self.t[0]
        T = 2*np.pi/self.frequencys[0]
        _index = np.linspace(0,len(self.data[:,0]),int(len(self.data[:,0])/(T/dt)),dtype = int,endpoint = False)
        self.evolutions = np.zeros([2,self.dims,len(_index)-2])
        self.t_evolutions = self.t[_index[1:-1]]
        omega_0 = self.frequencys[0]
        analytic_x0 = hilbert(self.data[:,0])
        origin_x0 = np.cos(omega_0*self.t)

        for i in range(self.dims):
            omega_i = self.frequencys[i]
            analytic_xi = hilbert(self.data[:,i])
            analytic_origin_xi = hilbert(origin_x0)
            amplitude_xi = np.abs(analytic_xi)
            amplitude_origin_xi = np.abs(analytic_origin_xi)
            instantaneous_phase_xi = np.angle(analytic_xi)
            instantaneous_phase_origin_xi = np.angle(analytic_origin_xi)
            for j in range(len(_index)-2):
                self.evolutions[0][i][j] = np.mean(amplitude_xi[_index[j+1]:_index[j+2]])
                sub_phase = np.unwrap(instantaneous_phase_xi[_index[j+1]:_index[j+2]]) - np.unwrap(instantaneous_phase_origin_xi[_index[j+1]:_index[j+2]])
                self.evolutions[1][i][j] = np.mean(sub_phase)
        
            self.evolutions[1][i] = np.unwrap(self.evolutions[1][i]) 
            

    def Library_rebuild(self):
        
        self.Phi = np.zeros([len(self.t_evolutions), 2, len(self.library), self.dims])
        t = symbols('t')
        As = []
        Betas = []
        states = []
        Variables = []
        # [x1, x1', x2, x2',...]
        for i in range(self.dims):
            As.append(symbols('A%d'%i))
            Betas.append(symbols('b%d'%i))
            omega_i = self.frequencys[i]
            states.append(As[i]*sin(omega_i*t + Betas[i]))
            states.append(As[i]*omega_i*cos(omega_i*t + Betas[i]))
        # print(states)
        Variables = [As,Betas] # [2, dims]
        for j in range(len(self.library)):
            Phi_j = self.library[j](states)
            for i in range(self.dims):
                omega_i = self.frequencys[i]
                A_i = As[i]
                Beta_i = Betas[i]
                phi_i = omega_i*t+Beta_i
                # print(Phi_j*cos(phi_i))
                Phi_iAj = -1/(2*pi)*integrate(Phi_j*cos(phi_i),(t,0,2*pi/omega_i))
                print(Phi_iAj)
                # Phi_iAj = trigsimp(Phi_iAj, method = 'matching')
                Phi_ibj = 1/(2*pi*A_i)*integrate(Phi_j*sin(phi_i),(t,0,2*pi/omega_i))
                # Phi_ibj = trigsimp(Phi_ibj, method = 'matching')
                func_Phi_iAj = lambdify([Variables],Phi_iAj)
                func_Phi_ibj = lambdify([Variables],Phi_ibj)
                self.Phi[:,0,j,i] = func_Phi_iAj(self.evolutions)
                
                self.Phi[:,1,j,i] = func_Phi_ibj(self.evolutions) 
                
                
            

    def optimize(self,sparse_threshold = 1e-3,stop_tolerance = 1e-4,step_tolerance = 1e-5,sparse_max = 6,smooth_window = 1,w_A2b = 2):
        self.predict = np.zeros([2,self.dims,len(self.t_evolutions)])
        # Xi = np.zeros(self.dims,len(self.library))
        dt = self.t_evolutions[1] - self.t_evolutions[0]
        X_librarys = []
        if smooth_window>2:
            dot_evolutions = np.zeros([2, self.dims,len(self.t_evolutions) - smooth_window+1])
            for i in range(self.dims):
                
                dot_evolutions[0,i,:] = moving_average(np.gradient(self.evolutions[0,i,:],dt,edge_order=2),smooth_window)
                dot_evolutions[1,i,:] = moving_average(np.gradient(self.evolutions[1,i,:],dt,edge_order=2),smooth_window)
                X_library = np.zeros([len(dot_evolutions[0,i,:]),2,len(self.library)])
                for j in range(2):
                    for n in range(len(self.library)):
                        X_library[:,j,n] = moving_average(self.Phi[:,j,n,i],smooth_window)
                # X_library = 

                X_librarys.append(X_library)
                # plt.subplots()
                # plt.plot(dot_evolutions[0][i])
        else:
            dot_evolutions = np.zeros([2, self.dims,len(self.t_evolutions)])
            for i in range(self.dims):
                dot_evolutions[0,i,:] = np.gradient(self.evolutions[0,i,:],dt,edge_order=2)
                dot_evolutions[1,i,:] = np.gradient(self.evolutions[1,i,:],dt,edge_order=2)
                X_library = self.Phi[:,:,:,i]
                X_librarys.append(X_library)
        

        for i in range(self.dims):
            Xi_i = OMP(dot_evolutions[:,i,:].T,X_librarys[i],sparse_threshold = sparse_threshold ,stop_tolerance = 1e-4, step_tolerance = step_tolerance, sparse_max = sparse_max,smooth_window =smooth_window,w_A2b = w_A2b)
            self.Xi[i,:] = Xi_i[0]