import numpy as np
import scipy.io as sio


# def generate_obj_fun(logtheta, x, alpha, offset, sign_objectives):
# 	_, n_obj = logtheta.shape
# 	logtheta_split = np.array(np.split(logtheta, n_obj, axis=1))
# 	x_split = np.array(np.split(x, n_obj, axis=1))
# 	alpha_split = np.array(np.split(alpha, n_obj, axis=1))
# 	offset_split = np.array(np.split(offset, n_obj, axis=1))

# 	@jit(nopython=True)
# 	def obj_fun(xstar):
		
# 		f = []
# 		for i in range(n_obj):
# 			Kstar = get_Kstar(logtheta_split[i], x_split[i], xstar)
# 			mu = np.dot(Kstar.T, alpha_split[i])
# 			mu = mu + offset_split[i]
# 			if len(mu) == 1:
# 				o = mu[0][0] # so that it's acceptable to jit
# 			f.append(o)

# 		f = np.array(f)
# 		f = np.multiply(f, sign_objectives) * -1
# 		return f

# 	return obj_fun

def evaluate(logtheta, x, alpha, xstar):
	Kstar = get_Kstar(logtheta, x, xstar)
	mu = np.dot(Kstar.T, alpha)
	return mu

def get_Kstar(logtheta, x, z):
	return squared_exponential(logtheta, x, z) #+ noise(logtheta, x, z)

def squared_exponential(logtheta, x, z):
	_, D = x.shape

	ell = np.ravel(np.exp(logtheta[:D])) # length scale. need to squeeze/ravel as we are using np.diag on it later. np.diag requires 1D array if we want a square diag matrix.
	if len(ell.shape) < 1:
		ell = ell.reshape(1,)
	sf2 = np.exp(2*logtheta[D]) # signal variance

	a = np.dot(np.atleast_2d(np.diag(1./ell)), x.T)
	b = np.dot(np.atleast_2d(np.diag(1./ell)), z.T)

	Kstar = sf2*np.exp(-sq_dist(a, b)/2) # Kstar should be of shape n_examples x n_tests
	return Kstar

def noise(logtheta, x, z):
	return 0 # zeros cross covariance by independence

def sq_dist(a, b):
	# a is of shape (dim, n_examples)
	# b is of shape (dim, n_tests)
	D, n = a.shape
	b = b.reshape(D, -1)

	_, m = b.shape
	
	C = np.zeros((n, m)) # in the same shape as Kstar
	for d in range(D):

		a1 = a[d, :].reshape(n, 1)
		b1 = b[d, :].reshape(1, m)

		x = np.square(b1 - a1)
		x = x.reshape(n, m)

		C = C + x
	return C


def get_sigma(logtheta, x, L, lik, z):
	D = np.size(x, 1)
	sf2 = np.exp(2*logtheta[D]) # signal variance
	s2 = np.exp(2*logtheta[-1])
	assert (logtheta[D+1] == logtheta[-1]) # check that the format of logtheta is as we assumed.

	Kstar = get_Kstar(logtheta, x, z)
	Kstarstar = sf2*np.ones((np.size(z, 0),1)) + s2

	# v = np.linalg.solve(L, Kstar)

	# sigma = Kstarstar - np.sum(np.multiply(v, v), 0).reshape((-1, 1))


	# FOR WHEN L IS NOT CHOLESKY DECOMPOSITION
	LKs = np.dot(L, Kstar)
	sigma = Kstarstar + np.sum(np.multiply(Kstar, LKs), 0).reshape((-1, 1))

	# print (LKs)
	# print (np.sum(LKs, 0).reshape((-1, 1)))
	# print (Kstar) is correct

	sn2 = np.exp(2*lik)
	sigma = sigma + sn2
	return sigma

if __name__ == '__main__':
	gpdata = sio.loadmat('sparsegpdata.mat')
	x = gpdata['xdata']
	logtheta = gpdata['logtheta']
	alpha = gpdata['alpha']
	xstar = gpdata['xs']
	L = gpdata['L']
	lik = np.asscalar(gpdata['likelihood'])
	# print (x.shape)
	# print (logtheta.shape)
	# print (alpha.shape)
	# print (xstar.shape)
	# print (L.shape)
	# print (lik)

	ymu = evaluate(logtheta, x, alpha, xstar)
	# print (ymu)

	ys2 = get_sigma(logtheta, x, L, lik, xstar)
	
	sio.savemat('out.mat', {'ymu_py': ymu, 'ys2_py': ys2})