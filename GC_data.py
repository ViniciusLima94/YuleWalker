########################################################################################
# COMPUTE GC FROM DATA
########################################################################################
import scipy.io
import scipy.linalg
import numpy             as np 
import matplotlib.pyplot as plt

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

def compute_transfer_function(AR, sigma, f):

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
		S[:,:,i] = np.matmul( np.matmul(H[:,:,i], sigma), np.conj(H[:,:,i]).T )

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
def AIC(sig, m, N, Nvars, T):
	M = T*(N-m)
	k = m*Nvars*Nvars
	L = -(M/2)*np.log(np.linalg.det(sig))
	aic = -2*L + 2*k*(M/(M-k-1))
	return aic

def xcorr(x,y,maxlags):
	N = x.shape[1]
	lags = np.arange(0,maxlags)
	Rxx = np.zeros([lags.shape[0],Nvars,Nvars])
	for k in lags:
		Rxx[k,:,:] = np.matmul(x[:,0:N-k],y[:,k:].T)/N
	return lags, Rxx

def demean(X, norm = False):

	n,m,N = X.shape

	U     = np.ones([1, N*m])
	Y     = np.swapaxes(X, 1,2).reshape((n, N*m))

	Y = Y-np.matmul(Y.mean(axis=1)[:,None], U)
	if norm == True:
		Y = Y / np.matmul(Y.std(axis=1)[:,None], U)

	return np.swapaxes( Y.reshape([n,N,m]), 1 ,2)

########################################################################################
# Estimate AR parameters via Yule-Walker equations
########################################################################################
def YuleWalker(X, m, maxlags=100):

	Nvars = X.shape[0]
	N     = X.shape[1] 

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

	r = np.reshape( Rxx[1:m+1], (Nvars*m,Nvars) )
	R = np.matmul(A.T, A)/N

	AR_yw  = np.matmul(scipy.linalg.inv(R).T,r).T
	AR_yw  = AR_yw.T.reshape((m,Nvars,Nvars))

	eps_yw = Rxx[0] 
	for i in range(m):
		eps_yw += np.matmul(-AR_yw[i].T,Rxx[i+1])

	return AR_yw, eps_yw

########################################################################################
# Computing GC from data
########################################################################################
X = scipy.io.loadmat('fig6.mat')['X']
step = 20
X = X[:,::step,:]
X = demean(X)

Nvars  = X.shape[0]
N      = X.shape[1]
Trials = X.shape[2]

########################################################################################
# Computing best order to estimate AR coefficients
########################################################################################
#np.array([1, 2, 3, 4, 6, 8, 9, 13])
ROI = np.array([0,1,2,3,5,7,8,12])
area_names = ['V1', 'V2', 'V4', 'DP', 'MT', '8m', '5', '8l', '2', 'TEO', 'F1',
       'STPc', '7A', '46d', '10', '9/46v', '9/46d', 'F5', 'TEpd', 'Pbr',
       '7m', 'LIP', 'F2', '7B', 'ProM', 'STPi', 'F7', '8B', 'STPr', '24c']
Fs  = 1 / (step*0.2e-3)

count = 0
for i in range(ROI.shape[0]):
	for j in range(i+1, ROI.shape[0]):

		X2 = np.zeros([2, N, Trials])
		X2[0,:,:] = X[ROI[i],:,:].copy()
		X2[1,:,:] = X[ROI[j],:,:].copy()

		m = 7
		AR    = np.zeros([m, Nvars, Nvars])
		sigma = np.zeros([Nvars, Nvars])
		for T in range(Trials):
			aux1, aux2 = YuleWalker(X2[:,:,T], m, maxlags=100)
			AR    += aux1 / Trials
			sigma += aux2 / Trials

		f    = compute_freq(N, Fs) 
		# Transfer function and spectral matrix
		H, S = compute_transfer_function(AR, sigma, f)
		# Granger causalities
		Ix2y, Iy2x, Ixy = granger_causality(H, sigma)

		plt.figure()
		plt.plot(f, Ix2y, label=area_names[ROI[i]]+'->'+area_names[ROI[j]])
		plt.plot(f, Iy2x, label=area_names[ROI[j]]+'->'+area_names[ROI[i]])
		plt.legend()
		plt.xlim([0, 100])
		a = np.max(Ix2y); b = np.max(Iy2x); c = max(a, b)
		plt.ylim([0, max(c, 1e-3)])
		plt.savefig('figures/'+str(count)+'.png', dpi=300)
		plt.close()
		count += 1

'''
m_values = np.arange(1,31,1, dtype=int)

aic   = np.zeros([m_values.shape[0]])
for m in m_values:
	sigma = np.zeros([Nvars, Nvars])
	for T in range(Trials):
		_, aux = YuleWalker(X[:,:,T], m, maxlags=100)
		sigma += aux / Trials
	aic[m-1] = AIC(sigma, m, N, Nvars, Trials)
	print('Computing model order, m = ' + str(m) + ', AIC = ' + str(aic[m-1]))
'''
'''
m = 7#m_values[np.argmin(aic)]
AR    = np.zeros([m, Nvars, Nvars])
sigma = np.zeros([Nvars, Nvars])
for T in range(Trials):
	aux1, aux2 = YuleWalker(X[:,:,T], m, maxlags=100)
	AR    += aux1 / Trials
	sigma += aux2 / Trials

Fs  = 1 / (step*0.2e-3)

# Frequency axis
f    = compute_freq(N, Fs) 
# Transfer function and spectral matrix
H, S = compute_transfer_function(AR, sigma, f)
# Granger causalities
Ix2y, Iy2x, Ixy = granger_causality(H, sigma)

plt.plot(f, Ix2y, label='X->Y')
plt.plot(f, Iy2x, label='Y->X')
plt.legend()
plt.xlim([0, 140])
#plt.ylim([0,0.25])
plt.show()
'''