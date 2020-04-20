from trace import MatrixInfo, TraceOperator
from solver import Solver
from optimize import minimize, operator_to_vector
from utility import debug
import numpy as np
import sys


def demo_one_matrix(g, maxlen, step=0.1, num_steps=10):
	"""
	Computes bootstrap ground state energy and bounds for <tr XX> in excited states, as a demonstration.
	Arguments:
		g (float): coupling in the Hamiltonian H = tr PP + tr XX + g / N tr XXXX
		maxlen (int): the maximal length of trial operators in the bootstrap matrix
		step (float), num_steps (int): will search for bounds of <tr XX> at energies E0 + i * step,
			where E0 is the ground state energy, i = 0, 1, ..., num_steps - 1
	Returns:
		E (numpy array of shape (num_steps,)): the energies E0 + i * step, 0 <= i < num_steps
		lb, ub (numpy array of shape (num_steps,)): the lower and upper bounds for <tr XX>
	"""
	# basic information about matrices and the hamiltonian
	mat_info = MatrixInfo("PX", np.array([[0, -1j], [1j, 0]]))
	mat_info.add_basis("AB", np.array([[1, -1j], [1, 1j]]) / 2)
	hamil = TraceOperator(mat_info, [(1, "PP"), (1, "XX"), (g, "XXXX")])
	debug("H = " + repr(hamil))
	gauge = [(1, "XP"), (-1, "PX"), (-1j, "")]
	# solving the model
	solver = Solver(hamil, gauge, "AB", lambda s: "".join([{"A": "B", "B": "A"}[c] for c in reversed(s)]))
	tables, cons = solver.table_with_constraints(maxlen, lambda s: len(s) % 2)
	for i, (size, _) in enumerate(tables):
		debug("Table %d: shape %s" % (i, (size, size)))
	# minimizing the energy
	hamil = hamil.rewrite("AB")
	unit = TraceOperator(mat_info, "")
	param = minimize(hamil, tables, cons, [(unit, 1)], np.zeros(solver.solution.num_variables))
	energy = lambda param: operator_to_vector(solver.solution, hamil).dot(param)
	e0 = energy(param)
	debug("E0 = {:.3f}".format(e0))
	# the observable tr XX
	obs = TraceOperator(mat_info, [(1, "XX")]).rewrite("AB")
	x2 = lambda param: operator_to_vector(solver.solution, obs).dot(param)
	debug("Scanning for lower bounds...")
	lb, es = [], []
	p = param
	for i in range(num_steps):
		es.append(e0 + i * step)
		p = minimize(obs, tables, cons, [(unit, 1), (hamil, e0 + i * step)], p)
		e, o = energy(p), x2(p)
		debug("E = {:.3f}, X2_low = {:.3f}".format(e, o))
		lb.append(o)
	debug("Scanning for upper bounds...")
	ub = []
	p = param
	for i in range(num_steps):
		p = minimize(-obs, tables, cons, [(unit, 1), (hamil, e0 + i * step)], p)
		e, o = energy(p), x2(p)
		debug("E = {:.3f}, X2_high = {:.3f}".format(e, o))
		ub.append(o)
	return np.array(es), np.array(lb), np.array(ub)

def demo_two_matrix(g, maxlen, init=None, save=False):
	"""
	Computes bootstrap ground state energy and observables <tr XX> and -<tr [X, Y]^2>, as a demonstration.
	Arguments:
		g (float): coupling in the Hamiltonian H = tr PP + tr QQ + tr XX + tr YY - g / N tr [X, Y]^2
		maxlen (int): the maximal length of trial operators in the bootstrap matrix
		init (None or numpy array of shape (num_variables,)): the initial parameters to use; if None, initialize with zeros
		save (bool): whether to save parameters to file
	Returns:
		param (numpy array of shape (num_variables,)): the optimal parameters found
		e0 (float): the minimal energy in the allowed region
		x2 (float): <tr XX> in the minimal energy state found
		c4 (float): -<tr [X, Y]^2> in the minimal energy state found
	"""
	# basic information about matrices and the hamiltonian
	mat_info = MatrixInfo("PXQY", np.array([[0, -1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]))
	mat_info.add_basis("ABCD", np.array([[1, -1j, -1j, -1], [1, 1j, 1j, -1], [1, -1j, 1j, 1], [1, 1j, -1j, 1]]) / 2)
	hamil = TraceOperator(mat_info, [(1, "PP"), (1, "XX"), (1, "QQ"), (1, "YY"), (-g, "XYXY"), (-g, "YXYX"), (g, "XYYX"), (g, "YXXY")])
	debug("H = " + repr(hamil))
	gauge = [(1, "XP"), (-1, "PX"), (1, "YQ"), (-1, "QY"), (-2j, "")]
	# solving the model
	solver = Solver(hamil, gauge, "ABCD", lambda s: "".join([{"A": "B", "B": "A", "C": "D", "D": "C"}[c] for c in reversed(s)]))
	tables, cons = solver.table_with_constraints(maxlen, lambda s: s.count("A") + s.count("D") - s.count("B") - s.count("C"))
	for i, (size, _) in enumerate(tables):
		debug("Table %d: shape %s" % (i, (size, size)))
	# minimizing the energy
	hamil = hamil.rewrite("ABCD")
	unit = TraceOperator(mat_info, "")
	filename = "param_{}_{:.2f}".format(maxlen, g)
	param = minimize(hamil, tables, cons, [(unit, 1)], init if init is not None else np.zeros(solver.solution.num_variables), savefile=filename if save else "")
	e0 = operator_to_vector(solver.solution, hamil).dot(param)
	debug("E0 = {:.3f}".format(e0))
	# the observable tr XX
	obs = TraceOperator(mat_info, [(0.5, "XX"), (0.5, "YY")]).rewrite("ABCD")
	x2 = operator_to_vector(solver.solution, obs).dot(param)
	debug("X2 = {:.3f}".format(x2))
	# the observable -tr [X, Y]^2
	obs = TraceOperator(mat_info, [(-1, "XYXY"), (-1, "YXYX"), (1, "XYYX"), (1, "YXXY")]).rewrite("ABCD")
	comm = operator_to_vector(solver.solution, obs).dot(param)
	debug("C4 = {:.3f}".format(comm))
	return param, e0, x2, comm


case = 3

if case == 0:
	# one matrix ground state energy and observable for different couplings
	ans_g = []
	ans_e, ans_lb, ans_ub = [], [], []
	for i in range(4, 21, 4):
		g = 0.2 * i
		ans_g.append(g)
		e, lb, ub = demo_one_matrix(g, 3, num_steps=1)
		ans_e = np.append(ans_e, e)
		ans_lb = np.append(ans_lb, lb)
		ans_ub = np.append(ans_ub, ub)
	with np.printoptions(precision=3, suppress=True):
		print("g =", repr(np.array(ans_g)))
		print("Energies:", repr(np.array(ans_e)))
		print("Lower bounds for <tr XX>:", repr(np.array(ans_lb)))
		print("Upper bounds for <tr XX>:", repr(np.array(ans_ub)))

if case == 1:
	# one matrix observable window for different energies at g = 1
	e, lb, ub = demo_one_matrix(1, 3, num_steps=20)
	with np.printoptions(precision=3, suppress=True):
		print("Energies:", repr(e))
		print("Lower bounds for <tr XX>:", repr(lb))
		print("Upper bounds for <tr XX>:", repr(ub))

if case == 2:
	# two matrix ground state energy and observable for different couplings
	ans_g = []
	ans_e, ans_x, ans_c = [], [], []
	p = None
	for i in range(4, 21, 4):
		g = 0.4 * i
		ans_g.append(g)
		p, e, x, c = demo_two_matrix(g, 3, p)
		ans_e = np.append(ans_e, e)
		ans_x = np.append(ans_x, x)
		ans_c = np.append(ans_c, c)
	with np.printoptions(precision=3, suppress=True):
		print("Energies:", repr(ans_e))
		print("<tr XX>:", repr(ans_x))
		print("-<tr [X, Y]^2>:", repr(ans_c))

if case == 3:
	# two matrix ground state energy and observable for one coupling
	assert len(sys.argv) == 3, "Usage: python demo.py L g"
	L, g = int(sys.argv[1]), float(sys.argv[2])
	p, e, x, c = demo_two_matrix(g, L, save=True)
	print("L = {}, g = {:.2f}".format(L, g))
	print("Energies: {:.3f}".format(e))
	print("<tr XX>: {:.3f}".format(x))
	print("-<tr [X, Y]^2>: {:.3f}".format(c))


