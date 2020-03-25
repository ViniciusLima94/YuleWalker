##########################################################################################
# Compute GC of AR process by solving the Yule-Walker equations
##########################################################################################
import numpy             as np 
import matplotlib.pyplot as plt
import scipy.linalg

########################################################################################
# Functions for spectral analysis and GC
########################################################################################
def compute_freq(N, Fs):
	# Simulated time
	T = N / Fs
	# Frequency array
	f = np.linspace(1/T,Fs/2-1/T,int(N/2+1))

	return f

def cxy(X, Y=[], f=None, Fs=1):
	# Number of data points
	N = X.shape[0]

	if len(Y) > 0:
		Xfft = np.fft.fft(X)[1:len(f)+1]
		Yfft = np.fft.fft(Y)[1:len(f)+1]
		Pxy  = Xfft*np.conj(Yfft) / (N)
		return Pxy
	else:
		Xfft = np.fft.fft(X)[1:len(f)+1]
		Pxx  = Xfft*np.conj(Xfft) / (N)
		return Pxx

def compute_transfer_function(AR, f):

	m     = AR.shape[0]
	Nvars = AR.shape[1]

	H = np.zeros([Nvars,Nvars,f.shape[0]]) * (1 + 1j)
	S = np.zeros([Nvars,Nvars,f.shape[0]]) * (1 + 1j)

	for i in range(0,m+1):
		comp = np.exp(-1j * f * 2 * np.pi * i/Fs)
		if i == 0:
			for j in range(comp.shape[0]):
				H[:,:,j] += np.eye(Nvars) * comp[j]
		else:
			for j in range(comp.shape[0]):
				H[:,:,j] += -AR[i-1].T * comp[j]

	for i in range(f.shape[0]):
		H[:,:,i] = np.linalg.inv(H[:,:,i])

	for i in range(f.shape[0]):
		S[:,:,i] = np.matmul( np.matmul(H[:,:,i], Sigma), np.conj(H[:,:,i]).T )

	return H, S

def granger_causality(H, Z):

	N = H.shape[2]

	Hxx = H[0,0,:]
	Hxy = H[0,1,:]
	Hyx = H[1,0,:]
	Hyy = H[1,1,:]

	Hxx_tilda = Hxx + (Z[0,1]/Z[0,0]) * Hxy
	Hyx_tilda = Hyx + (Z[0,1]/Z[0,0]) * Hxx
	Hyy_circf = Hyy + (Z[1,0]/Z[1,1]) * Hyx

	Syy = Hyy_circf*Z[1,1]*np.conj(Hyy_circf) + Hyx*(Z[0,0]-Z[1,0]*Z[1,0]/Z[1,1]) * np.conj(Hyx)
	Sxx = Hxx_tilda*Z[0,0]*np.conj(Hxx_tilda) + Hxy*(Z[1,1]-Z[0,1]*Z[0,1]/Z[0,0]) * np.conj(Hxy)

	Ix2y = np.log( Syy/(Hyy_circf*Z[1,1]*np.conj(Hyy_circf)) )
	Iy2x = np.log( Sxx/(Hxx_tilda*Z[0,0]*np.conj(Hxx_tilda)) )

	Ixy  = np.zeros(N)

	for i in range(N):
		Ixy[i]  = np.log( (Hxx_tilda[i]*Z[0,0]*np.conj(Hxx_tilda[i]))*(Hyy_circf[i]*Z[1,1]*np.conj(Hyy_circf[i])/np.linalg.det(S[:,:,i])) ).real
	
	return Ix2y.real, Iy2x.real, Ixy.real


########################################################################################
# Akaike information criteria and cross-correlations
########################################################################################
def AIC(sig, m, N):
	Nvars = sig.shape[0]
	return 2*np.log(np.linalg.det(sig))+2*(Nvars**2)*m/N

def xcorr(x,y,maxlags):
	N = x.shape[1]
	lags = np.arange(0,maxlags)
	Rxx = np.zeros([lags.shape[0],Nvars,Nvars])
	for k in lags:
		Rxx[k,:,:] = np.matmul(x[:,0:N-k],y[:,k:].T)/N
	return lags, Rxx

########################################################################################
# Generate auto-regressive model
########################################################################################
def ARmodel(N=5000, Fs = 200, cov = None):
	
	T = N / Fs

	time = np.linspace(0, T, N)

	X1 = np.random.random(N)
	X2 = np.random.random(N)

	E = np.random.multivariate_normal(np.zeros(cov.shape[0]), cov, size=(N,))
	for t in range(2, N):
		#X1[t] = 0.55*X1[t-1] + 0.20*X2[t-1] - 0.80*X1[t-2] + 0.00*X2[t-2] + E[t,0]
		#X2[t] = 0.00*X1[t-1] + 0.55*X2[t-1] + 0.00*X1[t-2] - 0.80*X2[t-2] + E[t,1]
		X1[t] = 0.90*X1[t-1] + 0.00*X2[t-1] - 0.50*X1[t-2] + 0.00*X2[t-2] + E[t,0]
		X2[t] = 0.16*X1[t-1] + 0.80*X2[t-1] - 0.20*X1[t-2] - 0.50*X2[t-2] + E[t,1]

	Z = np.zeros([2, N])

	Z[0] = X1
	Z[1] = X2

	return Z

########################################################################################
# Estimate AR parameters via Yule-Walker equations
########################################################################################
def YuleWalker(X, m, maxlags=100):

	Nvars = X.shape[0]
	N     = X.shape[1] 
	#Maximum number of lags for cross-correlation
	#maxlags = 100
	# Compute cross-correlations matrices for each lag
	lag, Rxx = xcorr(X,X,maxlags)

	#  Reorganizing data to compute crosscorrelation matrix
	b = X.T[m:]
	A = np.zeros([N-m,Nvars*m])

	count = 0
	for i in np.arange(0,m):
		for j in range(0,Nvars):
			A[:,count] = X.T[m-i-1:N-i-1,j]
			count      += 1

	r = np.reshape( Rxx[1:m+1], (2*m,Nvars) )
	R = np.matmul(A.T, A)/N

	AR_yw  = np.matmul(scipy.linalg.inv(R).T,r).T
	AR_yw  = AR_yw.T.reshape((m,Nvars,Nvars))

	eps_yw = Rxx[0] 
	for i in range(m):
		eps_yw += np.matmul(-AR_yw[i].T,Rxx[i+1])

	return AR_yw, eps_yw

N     = 5000	
Fs    = 200
Tsim  = N/Fs
Nvars = 2
cov   = np.array([ [1.00, 0.40],[0.40, 0.70] ])
X     = ARmodel(N = N, Fs = Fs, cov = cov)

########################################################################################
# Computing coefficients and covariance matrix
########################################################################################
#m_values  = np.arange(2,52,1, dtype=int)
#aic       = []
#for m in m_values:
#	AR, Sigma = YuleWalker(X, m, maxlags=100)
#	aic.append( AIC(Sigma, m, N) )
m = 2
AR = np.zeros([m,Nvars,Nvars])
Sigma = np.zeros([Nvars,Nvars])

for i in range(500):
	aux1, aux2 = YuleWalker(X, m, maxlags=100)
	AR+=aux1 / 500
	Sigma+=aux2/500

########################################################################################
# Computing spectral matrix and granger causality
########################################################################################
# Frequency axis
f    = compute_freq(N, Fs) 
# Transfer function and spectral matrix
H, S = compute_transfer_function(AR, f)
# Granger causalities
Ix2y, Iy2x, Ixy = granger_causality(H, Sigma)