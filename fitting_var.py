import numpy             as np 
import matplotlib.pyplot as plt
import scipy.linalg

def ARmodel(N=5000, Fs = 200, cov = None):
	
	T = N / Fs

	time = np.linspace(0, T, N)

	X1 = np.random.random(N)
	X2 = np.random.random(N)

	E = np.random.multivariate_normal(np.zeros(cov.shape[0]), cov, size=(N,))
	for t in range(2, N):
		X1[t] = 0.55*X1[t-1] - 0.2*X2[t-1] + 0.0*X1[t-2] - 0.5*X2[t-2] + E[t,0]
		X2[t] = 0.70*X1[t-1] - 0.3*X2[t-1] - 0.8*X1[t-2] - 0.0*X2[t-2] + E[t,1]

	Z = np.zeros([2, N])

	Z[0] = X1
	Z[1] = X2

	return Z

N     = 50000
Nvars = 2
cov   = np.array([ [1.00, 0.00],[0.00, 1.00] ])
X     = ARmodel(N = N, Fs = 200, cov = cov)

########################################################################################
# Estimating coefficients via Least-Squares
########################################################################################
m   = 2

b = X.T[m:]
A = np.zeros([N-m,Nvars*m])

count = 0
for i in np.arange(0,m):
	for j in range(0,Nvars):
		A[:,count] = X.T[m-i-1:N-i-1,j]
		count      += 1

phi = np.matmul(np.matmul(scipy.linalg.inv(np.matmul(A.T,A)),A.T),b).T

########################################################################################
# Estimating coefficients via YW equations
########################################################################################
def xcorr(x,y,maxlags):
	N = x.shape[0]
	lags = np.arange(0,maxlags)
	Rxx = np.zeros(lags.shape[0])
	for k in lags:
		Rxx[k] = np.matmul(x[0:N-k],y[k:][:,None]) / N
	return lags, Rxx