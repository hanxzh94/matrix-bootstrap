from utility import debug, debug_start, debug_end, expect, enable_debug, disable_debug, current_time
from trace import MatrixInfo, TraceOperator
from algebra import null_space, range_space
import scipy.sparse as sparse
import numpy as np
from itertools import product


def expand(A, shape): 
	# expand a sparse matrix to the given shape, padding zeros if necessary
	assert A.shape[0] <= shape[0] and A.shape[1] <= shape[1], "shape must be larger than original dimensions"
	return sparse.csr_matrix((A.data, A.indices, A.indptr), shape)

def expect_real(data):
	# the variables and constraints are expected to be real
	if data.dtype != np.complex:
		return data
	if data.imag.count_nonzero() == 0:
		return data.real
	if data.real.count_nonzero() == 0:
		return data.imag
	# throw a warning if not so
	debug("\tWARNING: complex entries")
	return data


class LinearSolution():
	"""
	To keep track of the solutions of the operators solved by the Solver class.
	Members:
		num_variables (int): the number of undetermined variables
		num_operators (int): the number of operators registered in the solution
		index (dict from string to int): the row index of the operators
		matrix (sparse matrix): solution data of shape (num_operators, num_variables)
		constraints (sparse matrix): linear relations between the variables, of shape (num_constraints, num_variables)
	"""

	def __init__(self):
		self.index = {}
		self.matrix = sparse.csr_matrix((0, 0))
		self.constraints = sparse.csr_matrix((0, 0))

	def __contains__(self, key):
		return key in self.index

	@property
	def num_variables(self):
		expect(self.matrix.shape[1] == self.constraints.shape[1])
		return self.matrix.shape[1]

	@property
	def num_operators(self):
		expect(self.matrix.shape[0] == len(self.index))
		return self.matrix.shape[0]

	def get_variable(self):
		"""
		Gets a new variable.
		Returns:
			res (sparse matrix): the new variable, of shape (1, num_variables)
		"""
		num = self.num_variables
		res = sparse.csr_matrix(([1], ([0], [num])), (1, num + 1))
		self.matrix = expand(self.matrix, (self.matrix.shape[0], num + 1))
		self.constraints = expand(self.constraints, (self.constraints.shape[0], num + 1))
		return res

	def add_constraints(self, constraints):
		"""
		Adds constraints of the variables.
		Arguments:
			constraints (sparse matrix): the constraints, of shape (num, num_variables)
		"""
		assert constraints.shape[-1] == self.num_variables, "dimension mismatch"
		self.constraints = sparse.vstack([self.constraints, expect_real(constraints)])

	def solve_constraints(self):
		"""
		Eliminates some variables by solving the constraints.
		"""
		if self.constraints.shape[0] == 0 or self.num_variables == 0:
			return
		sol = null_space(self.constraints)
		self.matrix = self.matrix.dot(sol)
		self.constraints = sparse.csr_matrix((0, sol.shape[-1]))

	def get_solution(self, key):
		"""
		Returns the solution of the operator specified by key, as a sparse matrix.
		Arguments:
			key (string): the operator
		Returns:
			sol (sparse matrix of size (1, num_variables)): solution for the operator
		"""
		return self.matrix[self.index[key]].copy()

	def add_solution(self, key, data):
		"""
		Adds a new entry to the solution.
		Arguments:
			key (string), data (sparse matrix of shape (1, num_variables)): the value for key is given by data
		"""
		assert isinstance(key, str) and sparse.issparse(data) and data.shape == (1, self.num_variables), "spurious insertion: key = %s, data = %s" % (key, data)
		assert key not in self.index, "duplicate data for key %s" % key
		self.index[key] = self.num_operators
		self.matrix = sparse.vstack([self.matrix, expect_real(data)])


class QuadraticSolution():
	"""
	To keep track of the quadratic equations from solving the cyclicity constraints. 
	Members:
		sol (LinearSolution): must have the linear solutions first
		index (dict from string to int): indices for operator pairs, e.g., index["AB,CD"] = 1
		param1 (sparse matrix): solution of the first operand; if index["AB,CD"] = i, then param1[i] is sol.get_solution("AB")
		param2 (sparse matrix): solution of the second operand; if index["AB,CD"] = i, then param2[i] is sol.get_solution("CD")
		matrix_line (sparse matrix): record of the linear part of the constraint
		matrix_quad (sparse matrix): record of the quadratic part of the constraint, e.g., if the quadratic part is
			2 * <tr AB> * <tr CD> - <tr AA> * <tr BB>, and index["AB,CD"] = 1, index["AA,BB"] = 2, the constraint is then 
			vectorized as [0, 2, -1, 0, ...]
	"""

	def __init__(self, sol):
		self.sol = sol
		num = sol.num_variables
		self.index = {}
		self.param1 = sparse.csr_matrix((0, num))
		self.param2 = sparse.csr_matrix((0, num))
		self.matrix_line = sparse.csr_matrix((0, num))
		self.matrix_quad = sparse.csr_matrix((0, 0))

	@property
	def num_constraints(self):
		expect(self.matrix_quad.shape[0] == self.matrix_line.shape[0])
		return self.matrix_quad.shape[0]

	def get_variable(self, op1, op2):
		"""
		Returns the index of the operator pair (op1, op2), and creates a new one if the pair did not appear before.
		Note the pair is unordered --- the index of (op1, op2) is the same as (op2, op1).
		Arguments:
			op1, op2 (string): two operators
		Returns:
			ind (int): index of the pair
		"""
		# comma to eliminate ambiguity e.g., op1 = "A", op2 = "BB" and op1 = "AB", op2 = "B"
		if (op1 + "," + op2) in self.index:
			return self.index[op1 + "," + op2]
		# unordered
		if (op2 + "," + op1) in self.index:
			return self.index[op2 + "," + op1]
		# not found
		ind = len(self.index)
		self.index[op1 + "," + op2] = ind
		self.param1 = sparse.vstack([self.param1, self.sol.get_solution(op1)])
		self.param2 = sparse.vstack([self.param2, self.sol.get_solution(op2)])
		self.matrix_quad = expand(self.matrix_quad, (self.matrix_quad.shape[0], ind + 1))
		return ind

	def add_constraint(self, quad, linear):
		"""
		Adds a quadratic constraint.
		Arguments:
			quad (list of (number, op1, op2) tuples): the quadratic part of the constraint,
				i.e., sum number * op1 * op2
			linear (sparse matrix of shape (1, num_variables)): linear part of the constraint
		"""
		# add the linear part
		self.matrix_line = sparse.vstack([self.matrix_line, expect_real(linear)])
		# add the quadratic part
		data, rows, cols = [], [], []
		for c, op1, op2 in quad:
			data.append(c)
			rows.append(0)
			cols.append(self.get_variable(op1, op2))
		res = sparse.csr_matrix((data, (rows, cols)), (1, self.matrix_quad.shape[-1]))
		self.matrix_quad = sparse.vstack([self.matrix_quad, expect_real(res)])

	def reduce_constraints(self):
		"""
		Eliminates linearly dependent constraints.
		"""
		expect(self.matrix_quad.shape[-1] == len(self.index))
		expect(self.matrix_line.shape[-1] == self.sol.num_variables)
		if self.matrix_quad.shape[0] == 0:
			return
		mat = sparse.hstack([self.matrix_quad, self.matrix_line])
		mat = range_space(mat)
		self.matrix_quad, self.matrix_line = mat[:, :len(self.index)], mat[:, len(self.index):]
		expect(self.matrix_line.shape[-1] == self.sol.num_variables)


class Solver():
	"""
	The matrix quantum mechanics bootstrap solver.
	Members:
		hamil (TraceOperator): the hamiltonian
		gauge (list of tuples (coefficient, string)): the gauge operator (which is not a trace operator)
		mats (string): the matrices to use to write down the operators
		conj (function string -> string): returns the hermitian conjugate of a matrix string
		solution (LinearSolution): the solutions found
	"""

	def __init__(self, hamil, gauge, mats, conj):
		self.hamil = hamil
		self.gauge = gauge
		self.mats = mats
		self.conj = conj
		self.solution = LinearSolution()

	def solve(self, ops):
		"""
		Solves for all the operators in ops.
		Arguments:
			ops (list of strings): the operators to solve
		Returns:
			sol (LinearSolution): solutions found
			cons (QuadraticSolution): the extra quadratic constraints from cyclicity
		"""
		# remove duplicates and sort by length
		seen = set()
		seen_add = seen.add
		ops = [x for x in ops if not (x in seen or seen_add(x))]
		ops = sorted(ops, key=len, reverse=True)
		debug("%d operators generated" % len(ops))
		debug_start("Solving the hamiltonian and gauge constraints")
		for i, op in enumerate(ops):
			if i % 100 == 0:
				debug("Operators %d/%d" % (i, len(ops)))
			expect(op not in self.solution)
			# solving op
			if self.conj(op) in self.solution:
				# hermitian conjugate of the operator is already known
				res = self.solution.get_solution(self.conj(op))
			else:
				# solve_hamiltonian_constraint is expensive
				# so do not have to run it if we know that None is returned
				res = self.solve_hamiltonian_constraint(op) if len(op) < len(ops[0]) else None
				if res is None:
					# no solution found; creating a new variable
					res = self.solution.get_variable()
			self.solution.add_solution(op, res)
			self.impose_gauge_constraint(op)
		debug_end("Number of variables: %d" % self.solution.num_variables)
		debug_start("Solving the cyclicity constraints")
		# solving linear constraints
		for i, op in enumerate(ops):
			if i % 100 == 0:
				debug("Operators %d/%d" % (i, len(ops)))
			res = self.cyclicity_constraint(op, linear_only=True)
			if res is not None:
				self.solution.add_constraints(res)
		self.solution.solve_constraints()
		debug("Linear cyclicity constraints solved. Number of variables now: %d" % self.solution.num_variables)
		# solving quadratic constraints
		self.quad = QuadraticSolution(self.solution)
		for i, op in enumerate(ops):
			if i % 100 == 0:
				debug("Operators %d/%d" % (i, len(ops)))
			res = self.cyclicity_constraint(op)
			if res is not None and res[0]:
				# a nontrivial quadratic constraint is found
				self.quad.add_constraint(*res)
		self.quad.reduce_constraints()
		debug("Number of independent quadratic constraints: %d" % self.quad.num_constraints)
		debug_end("Number of variables: %d" % self.solution.num_variables)
		return self.solution, self.quad

	def solve_hamiltonian_constraint(self, op):
		"""
		Solves the hamiltonian constraint <[H, tr op]> = 0 for the given operator.
		Arguments:
			op (string): the matrix operator to solve
		Returns:
			res (sparse matrix or None): the solution in terms of the variables in self.solution
				None if solution is not found
		"""
		# compute the commutator with the hamiltonian
		commutator = self.hamil.commutator(TraceOperator(self.hamil.mat_info, op))
		commutator = commutator.rewrite(self.mats)
		# solve for the operator: the constraint is coef * op + res = 0
		res = sparse.csr_matrix((1, self.solution.num_variables))
		coef = 0
		for c, s in commutator.components():
			if s == op:
				coef += np.real(c)
			else:
				expect(len(s) > len(op))
				if s not in self.solution:
					return None
				res = res + np.real(c) * self.solution.get_solution(s)
		if abs(coef) < 1e-8:
			# no solution for op; new constraints on res instead
			self.solution.add_constraints(res)
			return None
		# solution found
		return res * (-1 / coef)

	def impose_gauge_constraint(self, op):
		"""
		Imposes the gauge constraint <tr G op> = 0 to eliminate some variables.
		Arguments:
			op (string): a matrix operator
		Returns:
			success (bool): whether the constraint is found
		"""
		res = sparse.csr_matrix((1, self.solution.num_variables))
		for c, s in self.gauge:
			op_new = TraceOperator(self.hamil.mat_info, s + op).rewrite(self.mats)
			for d, t in op_new.components():
				if t not in self.solution:
					return False
				res = res + c * d * self.solution.get_solution(t)
		self.solution.add_constraints(res)
		return True

	def cyclicity_constraint(self, op, linear_only=False):
		"""
		Returns the cyclicity constraint by moving the first matrix in op to the last.
		WARNING: I assume that we know expectation value of an odd number of matrices is zero,
			and <1> = 1!
		Arguments:
			op (string): a matrix operator
			linear_only (bool): whether to return only if the constraint is linear
		Returns:
			None if no constraint; otherwise:
			quad (list of (number, op1, op2) tuples, if linear_only=False): 
				the quadratic part of the constraint, i.e., sum number * op1 * op2
			linear (sparse matrix of shape (1, solution.num_variables)): 
				the linear part of the constraint
		"""
		if len(op) < 2 or op not in self.solution or op[1:] + op[0] not in self.solution:
			return None
		linear = self.solution.get_solution(op) - self.solution.get_solution(op[1:] + op[0])
		quad = []
		for i, c in enumerate(op):
			# contribution from commuting op[0] and op[i]
			coef = self.hamil.mat_info.commutator(op[0], op[i])
			expect(float(np.abs(coef.imag)) < 1e-8)
			coef = np.real(coef)
			# assume that we know expectation value of an odd number of matrices is zero
			if i % 2 == 1 and np.abs(coef) > 1e-8:
				# the trace is split into two: coef tr op1 tr op2
				op1, op2 = op[1:i], op[i+1:]
				if op1 not in self.solution or op2 not in self.solution:
					return None
				if op1 == "" or op2 == "":
					# one of the two operators is actually identity
					# so the constraint is actually linear, assuming that <1> = 1
					linear = linear - coef * self.solution.get_solution(op1 or op2)
				elif linear_only:
					# the constraint is really quadratic but we only care about linear ones
					return None
				else:
					quad.append((-coef, op1, op2))
		return (quad, linear) if not linear_only else linear

	def table_with_constraints(self, maxlen, degree=lambda s: 0):
		"""
		Creates the bootstrap table with constraints.
		Arguments:
			maxlen (int): the maximal length of the trial operators
			degree (function string -> int): assigning a degree to each trial operator --- the expectation 
				value vanishes unless the total degree is zero.
		Returns:
			tables (list of (int, sparse array)): each tuple (size, sparse array) gives a square block in the 
				bootstrap matrix, which is sparse_array.dot(param).reshape((size, size))
			constraint (QuadraticSolution): the quadratic constraints
		"""
		# generates all trial operators with length <= maxlen and collect them by degree
		trial_ops = sum([["".join(s) for s in product(self.mats, repeat=l)] for l in range(maxlen+1)], [])
		trial_ops_by_degree = {}
		for op in trial_ops:
			d = degree(op)
			trial_ops_by_degree[d] = trial_ops_by_degree.get(d, []) + [op]
		# bootstrap tables for different degrees
		tables = [(len(trial_ops_by_degree[d]), 
					[self.conj(s1) + s2 for s1, s2 in product(trial_ops_by_degree[d], repeat=2)]
				  ) for d in trial_ops_by_degree]
		ops = sum([t for size, t in tables], [])
		# solves the constraints
		sol, cons = self.solve(ops)
		tables = [(size, sparse.vstack([self.solution.get_solution(ent) for ent in table])) for size, table in tables]
		return tables, cons


