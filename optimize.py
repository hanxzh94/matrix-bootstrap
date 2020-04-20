from trace import MatrixInfo, TraceOperator
import scipy.sparse as sparse
from scipy.sparse.linalg import lsqr
import numpy as np
import scipy.linalg
from solver import Solver
import cvxpy as cp
from utility import debug
import os


def operator_to_vector(sol, ops):
	"""
	Converts a TraceOperator into a vector such that the expectation valuee of the operator is the vector dot parameters.
	Arguments:
		sol (LinearSolution): solutions of operators in terms of the parameters
		ops (TraceOperator): the operator to convert (must be hermitian)
	Returns:
		vec (numpy array of shape (num_variables,)): <ops> = vec.dot(param)
	"""
	vec = np.zeros(sol.num_variables)
	for c, op in ops.components():
		vec += np.real(c) * sol.get_solution(op).toarray()[0]
	return vec

def minimal_eigval(tables, param):
	"""
	Computes the minimal eigenvalue of all the bootstrap matrices.
	Arguments:
		tables (list of tuples (int, sparse matrix)): the first integer gives the size of the bootstrap matrix and the sparse matrix then 
			has shape (size * size, num_variables), dot the matrix with the parameters and reshape into a list of real symmetric square 
			matrices. The constraint is that these matrices must be positive semidefinite.
		param (numpy array of shape (num_variables,)): the parameters
	Returns:
		eig (float): the minimal eigenvalue of all tables
	"""
	tables_val = [np.reshape(t.dot(param), (size, size)) for size, t in tables]
	return min([scipy.linalg.eigvalsh(t)[0] for t in tables_val])


def quad_constraint(cons, param, compute_grad=False):
	"""
	Returns the quadratic constraint vector and gradients from the QuadraticSolution at given parameters.
	Arguments:
		cons (QuadraticSolution): the quadratic constraints
		param (numpy array of shape (num_variables,)): the parameters
		compute_grad (bool): whether to compute the gradient matrix
	Returns:
		val (numpy array of shape (num_constraints,)): the constraint vector
		grad (when compute_grad=True, sparse matrix of shape (num_constraints, num_variables)): 
			gradients of the constraint vector with respect to the parameters
	"""
	# val = quad @ (param1 @ param * param2 @ param) + line @ param
	param1, param2 = cons.param1, cons.param2
	quad, line = cons.matrix_quad, cons.matrix_line
	p1, p2 = param1.dot(param), param2.dot(param)
	val = quad.dot(p1 * p2) + line.dot(param)
	if not compute_grad:
		return val
	# grad = quad @ (param1 * p2 + p1 * param2) + line
	p_grad = sparse.diags(p1) * param2 + sparse.diags(p2) * param1
	grad = line + quad.dot(p_grad)
	return val, grad

def sdp_init(tables, A, b, init, reg=1, maxiters=5_000, eps=1e-4, verbose=True):
	"""
	Finds the parameters such that
		1. All bootstrap tables are positive semidefinite;
		2. ||A.dot(param) - b||^2_2 + reg * ||param - init||^2_2 is minimized.
	Arguments:
		tables (list of tuples (int, sparse matrix)): the first integer gives the size of the bootstrap matrix and the sparse matrix then 
			has shape (size * size, num_variables), dot the matrix with the parameters and reshape into a list of real symmetric square 
			matrices. The constraint is that these matrices must be positive semidefinite.
		A (sparse matrix of shape (num_constraints, num_variables)), b (numpy array of shape (num_constraints,)): 
			linear constraints that A.dot(param) = b
		init (numpy array of shape (num_variables,)): the initial parameters
		reg (float): regularization
		maxiters (int), eps (float), verbose (bool): options for the convex solver
	Returns:
		param (numpy array of shape (num_variables,)): the optimal parameters found
	"""
	num_variables = init.size
	param = cp.Variable(num_variables)
	# the constraints 
	constraints  = [cp.reshape(t @ param, (size, size)) >> 0 for size, t in tables]
	# solve
	prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ param - b) + reg * cp.sum_squares(param - init)), constraints)
	prob.solve(verbose=verbose, max_iters=maxiters, eps=eps, solver=cp.SCS)
	if str(prob.status) != "optimal":
		debug("WARNING: sdp_init unexpected status: " + prob.status)
	return param.value

def sdp_relax(tables, A, b, init, radius, maxiters=10_000, eps=1e-4, verbose=True):
	"""
	Finds the parameters such that
		1. All bootstrap tables are positive semidefinite;
		2. ||param - init||_2 <= 0.8 * radius;
		3. The violation of linear constraints ||A.dot(param) - b||_2 is minimized.
	Arguments:
		tables (list of tuples (int, sparse matrix)): the first integer gives the size of the bootstrap matrix and the sparse matrix then 
			has shape (size * size, num_variables), dot the matrix with the parameters and reshape into a list of real symmetric square 
			matrices. The constraint is that these matrices must be positive semidefinite.
		A (sparse matrix of shape (num_constraints, num_variables)), b (numpy array of shape (num_constraints,)): 
			linear constraints that A.dot(param) = b
		init (numpy array of shape (num_variables,)): the initial parameters
		radius (float): radius of the trust region
		maxiters (int), eps (float), verbose (bool): options for the convex solver
	Returns:
		param (numpy array of shape (num_variables,)): the optimal parameters found
	"""
	num_variables = init.size
	param = cp.Variable(num_variables)
	# the constraints 
	constraints  = [cp.norm(param - init) <= 0.8 * radius]
	constraints += [cp.reshape(t @ param, (size, size)) >> 0 for size, t in tables]
	# solve
	prob = cp.Problem(cp.Minimize(cp.norm(A @ param - b)), constraints)
	prob.solve(verbose=verbose, max_iters=maxiters, eps=eps, solver=cp.SCS)
	if str(prob.status) != "optimal":
		debug("WARNING: sdp_relax unexpected status: " + prob.status)
	return param.value

def sdp_minimize(vec, tables, A, b, init, radius, reg=0, maxiters=10_000, eps=1e-4, verbose=True):
	"""
	Finds the parameters such that
		1. All bootstrap tables are positive semidefinite;
		2. ||param - init||_2 <= radius;
		3. A.dot(param) = b;
		4. vec.dot(param) + reg * np.linalg.norm(param) is minimized.
	Arguments:
		vec (numpy array of shape (num_variables,))
		tables (list of tuples (int, sparse matrix)): the first integer gives the size of the bootstrap matrix and the sparse matrix then 
			has shape (size * size, num_variables), dot the matrix with the parameters and reshape into a list of real symmetric square 
			matrices. The constraint is that these matrices must be positive semidefinite.
		A (sparse matrix of shape (num_constraints, num_variables)), b (numpy array of shape (num_constraints,)): 
			linear constraints that A.dot(param) = b
		init (numpy array of shape (num_variables,)): the initial parameters
		radius (float): radius of the trust region
		reg (float): regularization parameter
		maxiters (int), eps (float), verbose (bool): options for the convex solver
	Returns:
		param (numpy array of shape (num_variables,)): the optimal parameters found
	"""
	num_variables = init.size
	param = cp.Variable(num_variables)
	# the constraints 
	constraints  = [cp.norm(param - init) <= radius]
	constraints += [cp.reshape(t @ param, (size, size)) >> 0 for size, t in tables]
	constraints += [A @ param == b]
	# the loss to minimize
	loss = vec @ param + reg * cp.norm(param)
	# solve
	prob = cp.Problem(cp.Minimize(loss), constraints)
	prob.solve(verbose=verbose, max_iters=maxiters, eps=eps, solver=cp.SCS)
	if str(prob.status) != "optimal":
		debug("WARNING: sdp_minimize unexpected status: " + prob.status)
	return param.value

def minimize(op, tables, quad_cons, op_cons, init, maxiters=25, eps=5e-4, reg=5e-4, verbose=True, savefile=""):
	"""
	Minimizes the operator subject to the bootstrap positivity constraint, the quadratic cyclicity constraint, and the operator values
	constraints. The algorithm is a trust-region sequential semidefinite programming with regularization on l2 norm of the parameters.
	Arguments:
		op (TraceOperator): expectation value of this operator is to be minimized
		tables (list of tuples (int, sparse matrix)): the first integer gives the size of the bootstrap matrix and the sparse matrix then 
			has shape (size * size, num_variables), dot the matrix with the parameters and reshape into a list of real symmetric square 
			matrices. The constraint is that these matrices must be positive semidefinite.
		quad_cons (QuadraticSolution): the quadratic constraints from solving the cyclicity constraints
		op_cons (list of tuples (TraceOperator, float)): the extra constraints specifying the expectation values of the given operators
		init (numpy array of shape (num_variables,)): the initial value of the parameters
		eps (float): the accuracy goal
		reg (float): regulariation on parameters. The true objective to minimize will be expectation value of op + reg * l2-norm(param)
		verbose (bool): whether to print the optimization progress
		savefile (string): the filename to save the parameters
	Returns:
		param (numpy array of shape (num_variables,)): the optimal value of parameters found
	"""
	sol = quad_cons.sol
	# the loss function to minimize, i.e., the value of op
	vec = operator_to_vector(sol, op)
	loss = lambda param: vec.dot(param)
	# extra constraints in op_cons, i.e., <o> = v for o, v in op_cons
	A_op = sparse.csr_matrix((0, sol.num_variables))
	b_op = np.zeros(0)
	for o, v in op_cons:
		A_op = sparse.vstack([A_op, sparse.csr_matrix(operator_to_vector(sol, o))])
		b_op = np.append(b_op, v)
	# initialize parameters from file or from scratch
	last_param = None
	if savefile and os.path.isfile(savefile + ".npz"):
		npzfile = np.load(savefile + ".npz")
		radius, obs = npzfile["radius"], npzfile["obs"]
		param = lsqr(quad_cons.sol.matrix, obs)[0]
		# sanity checks
		debug("Error: {}".format(np.linalg.norm(quad_cons.sol.matrix.dot(param) - obs)))
		debug("minimal_eigval: {}".format(minimal_eigval(tables, param)))
		debug("Data read successfully")
	else:
		debug("Starting from scratch...")
		# find an initial parameter close to init that makes all bootstrap matrices positive
		param = sdp_init(tables, A_op, b_op, init, verbose=True)
		radius = np.linalg.norm(param) + 20
	# penalty parameter for violation of constraints
	mu = 1
	# optimization steps
	for step in range(maxiters):
		# combine the constraints from op_cons and linearized quadratic constraints, i.e., grad * (new_param - param) + val = 0
		val, grad = quad_constraint(quad_cons, param, compute_grad=True)
		A = sparse.vstack([A_op, grad])
		b = np.append(b_op, grad.dot(param) - val)
		# one step
		relaxed_param = sdp_relax(tables, A, b, param, radius, verbose=True)
		new_param = sdp_minimize(vec, tables, A, A.dot(relaxed_param), param, radius, reg=reg, verbose=True)
		if new_param is None:
			# wrongly infeasible
			radius *= 0.9
			continue
		# check progress
		cons_val = A.dot(param) - b
		maxcv = np.max(np.abs(cons_val)) # maximal constraint violation
		min_eig = minimal_eigval(tables, param)
		if verbose:
			print("Step {}: \tloss = {:.5f}, maxcv = {:.5f}, radius = {:.3f}, min_eig = {:+.5f}".format(step, loss(param), maxcv, radius, min_eig))
			print("\t\tnorm = {:.5f}, update = {:.5f}".format(np.linalg.norm(param), np.linalg.norm(new_param - param)))
		if last_param is not None and maxcv < eps and min_eig > -eps and loss(param) > loss(last_param) - 0.1 * eps and np.linalg.norm(param - last_param) < eps * radius:
			if verbose:
				print("Accuracy goal achieved.")
			return param
		# compute the changes in the merit function and decide whether the step should be accepted
		dloss = loss(new_param) - loss(param) + reg * (np.linalg.norm(new_param) - np.linalg.norm(param))
		dcons = np.linalg.norm(np.append(A_op.dot(new_param) - b_op, quad_constraint(quad_cons, new_param))) - np.linalg.norm(cons_val)
		if -dcons > eps:
			mu = max(mu, -2 * dloss / dcons)
		# the merit function is -loss(param) - reg * norm(param) - mu * norm(constraint vector)
		acred = -dcons # actual constraint reduction
		ared = -dloss + mu * acred # actual merit increase
		pcred = np.linalg.norm(cons_val) - np.linalg.norm(A.dot(new_param) - b)  # predicted constraint reduction
		pred = -dloss + mu * pcred # predicted merit increase
		rho = ared / pred
		# some sanity checks
		if verbose:
			print("\t\tmu = {:.3f}, margin = {:+.3f}".format(mu, -dloss - 0.5 * mu * dcons))
			print("\t\tacred = {:+.5f}, pcred = {:+.5f}, ared = {:+.5f}, pred = {:+.5f}, rho = {:+.3f}".format(acred, pcred, ared, pred, rho))
		if rho > 0.5:
			# accept
			if maxcv < eps and min_eig > -eps:
				last_param = param
				radius *= 1.2
			param = new_param
			if savefile:
				obs = quad_cons.sol.matrix.dot(param)
				np.savez(savefile, radius=radius, obs=obs)
				debug("Data saved successfully")
		else:
			# reject
			radius = 0.8 * np.linalg.norm(new_param - param)
	debug("WARNING: minimize did not converge to precision {:.5f} within {} steps.".format(eps, maxiters))
	return param


