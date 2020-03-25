'''
	Simulate and fits an AR coefficient
'''
import numpy             as np 
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.signal

########################################################################################
# Functions for spectral analysis
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

def simulate_AR(N, Fs, c, seed = 1):

	np.random.seed(seed)

	# Time array
	t  = np.arange(0, N/Fs, 1/Fs)
	# AR(4) array initialized with random numbers
	X  = np.random.random(t.shape[0])
	# Noise vector
	xi = np.random.normal(0,1, size=N)

	for n in range(len(c), t.shape[0]-1):
		X[n+1] = np.dot(c[::-1], X[n-len(c)+1:n+1]) + xi[n]
		#c[0]*X[n] + c[1]*X[n-1] + c[2]*X[n-2] + c[3]*X[n-3] + np.random.normal(0,1)

	return X

# Number of trials
Trials = 100
# Sampling frequency
Fs     = 200
# Number of data points
N      = 1000
# Data matrix
X      = np.zeros([N, Trials])
# Simulation time
tsim   = N/Fs

# Coefficients of the ar model
c = [0.7, 0.2, -0.1, -0.3]
#c = [2.2137, -2.9403, 2.1697, -0.9606]
#c = [-0.9, -0.6, -0.5]

for T in range(Trials):
	X[:,T] = scipy.signal.lfilter([1], -np.array([-1]+c), np.random.randn(N))#simulate_AR(N, Fs, c, seed = T*1000)

########################################################################################
# Estimating spectrum via Fourier transform
########################################################################################
f = compute_freq(N, Fs)
S = np.zeros(int(N/2+1))

for T in range(Trials):
	S += cxy(X[:,T], Y=X[:,T], f=f, Fs=Fs).real

########################################################################################
# Estimating spectrum via AR coefficients
########################################################################################
s = 0
for i in range(1,len(c)+1): 
	s += c[i-1]*np.exp(-1j*np.pi*2*f*i/Fs)
S_ar = 1/(np.abs(1-s)**2)

plt.semilogy(f, S/np.trapz(S,f),  label='Estimated via Fourier Transform')
plt.semilogy(f, S_ar/np.trapz(S_ar,f), '--', label='Estimated with AR coefficients')
plt.ylabel(r'$S_{xx}$')
plt.xlabel('Frequency [Hz]')
plt.legend()
#plt.show()

########################################################################################
# Estimating coefficients via Least-Squares
########################################################################################
def ARcoef(x, m):
	N = x.shape[0]
	A = np.zeros([N-m,m])
	b = x[m:]

	for i in np.arange(0,m):
		A[:,i] = x[m-i-1:N-i-1]

	return b, A, np.matmul(np.matmul(scipy.linalg.inv(np.matmul(A.T,A)),A.T),b)

m  = 4
ARc = np.zeros([Trials, 4])
for T in range(Trials):
	b, A, ARc[T,:] = ARcoef(X[:,T], m)

phi = ARc.mean(axis=0)

eps = b[:,None] - np.matmul(A,phi[:,None])

########################################################################################
# Computing correlation
########################################################################################
def xcorr(x,y, maxlags):
	N = x.shape[0]
	lags = np.arange(0,maxlags)
	Rxx = np.zeros(lags.shape[0])
	for k in lags:
		Rxx[k] = np.matmul(x[0:N-k],y[k:][:,None])/N
	#lag = np.array(list(-lags[::-1])+list(lags))
	#Rxx = np.array((list(Rxx[::-1])+list(Rxx)))
	return lags, Rxx #/ Rxx.max()

maxlags = 1000
Rxx     = np.zeros([maxlags, Trials])
for T in range(Trials):
	print('Trial = ' + str(T))
	lag, Rxx[:, T] = xcorr(X[:,T],X[:,T], maxlags)

Rxxm = Rxx.mean(axis=1)
#Rxx = Rxx / Rxx.max()

def YuleWalker(Rxx, m):

	r = Rxx[1:m+1]
	R = np.zeros([m, m])
	for i in range(m):
		j      = np.abs( np.arange(0,m, dtype=int)-i )
		R[i,:] = Rxx[j]

	AR_yw  = np.matmul(scipy.linalg.inv(R),r)

	#if order == 2:
	#	eps_yw = b[:,None] - A*AR_yw[:,None]
	#else:
	eps_yw = Rxx[0] + np.sum(-AR_yw*Rxx[1:m+1])#b[:,None] - np.matmul(A, AR_yw[:,None])

	return AR_yw, eps_yw

m_order = np.arange(2,30,1,dtype=int) # Model order
akaike=[]
sigv  = []
for m in m_order:
	ARyw, Sig = YuleWalker(Rxxm, m)
	akaike.append( N*np.log(Sig) + 2*m )
	sigv.append(Sig)
'''
sigv=[]
for T in range(Trials):
	ARyw, sig = YuleWalker(Rxx[:,T], 3)
	sigv.append(sig)
'''