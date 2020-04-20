import numpy as np
from utility import debug_start, debug_end, debug


class MatrixInfo():
	"""
	Information about the matrices we use and their commutators.
	Members:
		matrices (list of strings): the matrix basis to use
		coefficients (list of numpy arrays): the transformation matrix from the matrix basis
			to the matrix basis given in the constructor, matrices[i][j] = sum_k coefficients[i][j][k] * matrices[0][k]
		commutators (numpy array): the commutators of the matrix basis given in the constructor i.e., matrices[0]
	"""

	def __init__(self, matrices, commutators):
		"""
		Set the basic matrix basis.
		Arguments:
			matrices (string): symbols of the matrices to use, e.g., "PXQY"
			commutators (numpy array): commutators between these matrices (4*4 array in the previous example),
				[matrices[i], matrices[j]] = commutators[i, j]
		"""
		l = len(matrices)
		assert commutators.shape == (l, l), "dimension mismatch"
		assert len(set(matrices)) == l, "duplicate symbols in %s" % matrices
		debug_start("%d matrices with nonzero commutators:" % l)
		self.matrices = [matrices]
		self.coefficients = [np.eye(len(matrices))]
		self.commutators = commutators
		# print for verification
		for i, s in enumerate(matrices):
			for j, t in enumerate(matrices):
				if np.abs(commutators[i, j]) > 1e-8:
					debug("[{}_ij, {}_kl] = {} delta_il delta_jk".format(s, t, commutators[i, j]))
		debug_end("MatrixInfo constructed")

	def add_basis(self, matrices, coefficients):
		"""
		Add another matrix basis.
		Arguments:
			matrices (string): symbols of the matrices to use
			coefficients (numpy array): the coefficients when expanded into the matrix basis given in the constructor
		"""
		l = len(self.matrices[0])
		assert coefficients.shape == (l, l) and len(matrices) == l, "dimension mismatch"
		assert np.linalg.matrix_rank(coefficients) == l, "the coefficients matrix must be invertible"
		self.matrices.append(matrices)
		self.coefficients.append(coefficients)
		# check that no repeated characters
		assert len(set("".join(self.matrices))) == l * len(self.matrices), "duplicate symbols in %s" % self.matrices
		# print for verification
		debug_start("Adding a new matrix basis:")
		for i, s in enumerate(matrices):
			components = ["{} {}".format(coefficients[i, j], t) for j, t in enumerate(self.matrices[0]) if np.abs(coefficients[i, j]) > 1e-8]
			debug("{} = {}".format(s, " + ".join(components)))
		debug_end("add_basis complete")

	def get_coefficients(self, matrix):
		"""
		Get the coefficient when expanding the matrix onto the basis given in the constructor.
		Arguments:
			matrix (character): the matrix to expand
		Returns:
			coef (numpy vector): the coefficients
		"""
		assert isinstance(matrix, str) and len(matrix) == 1, "matrix must be a character"
		for i, matrices in enumerate(self.matrices):
			if matrix in matrices:
				return self.coefficients[i][matrices.index(matrix)]
		assert False, "matrix %s not found" % matrix

	def commutator(self, matrix1, matrix2):
		"""
		The quantum commutator between matrix1 and matrix2.
		Arguments:
			matrix1, matrix2 (character): the matrices
		Returns:
			res (complex number): [matrix1_ij, matrix2_kl] = res * delta_il * delta_jk
		"""
		assert isinstance(matrix1, str) and len(matrix1) == 1, "matrix1 must be a character"
		assert isinstance(matrix2, str) and len(matrix2) == 1, "matrix2 must be a character"
		return np.dot(self.get_coefficients(matrix1), np.dot(self.commutators, self.get_coefficients(matrix2)))

"""
A = P - i X - i (Q - i Y)
B = P + i X + i (Q + i Y)
C = P - i X + i (Q - i Y)
D = P + i X - i (Q + i Y)
"""

if __name__ == "__main__":
	mat_info = MatrixInfo("PXQY", np.array([[0, -1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]))
	mat_info.add_basis("ABCD", np.array([[1, -1j, -1j, -1], [1, 1j, 1j, -1], [1, -1j, 1j, 1], [1, 1j, -1j, 1]]))
	print("Commutator table:")
	print("\tA\tB\tC\tD")
	for s in "ABCD":
		print(s + "\t" + "\t".join(["{}".format(mat_info.commutator(s, t)) for t in "ABCD"]))


class TraceOperator():
	"""
	Class for a sum of single-trace matrix operators.
	"""

	def __init__(self, mat_info, ops=[]):
		"""
		Arguments:
			mat_info (MatrixInfo): information about the matrices involved
			ops (string or list of (number, string) tuples):
				if ops is a list, it is understood as a sum of single-trace operators, e.g., [(1, "XX"), (1, "YY")]
				means tr XX + tr YY; if ops is a string, it is understood as [(1, ops)]
		"""
		self.mat_info = mat_info
		if isinstance(ops, str):
			ops = [(1, ops)]
		assert isinstance(ops, list), "the argument must be a string or a list"
		# collect the coefficients of different operators (and remove duplicates or zeros if existing)
		coef = {}
		for c, s in ops:
			coef[s] = coef.get(s, 0) + c
		self.ops = [(coef[s], s) for s in coef if np.abs(coef[s]) > 1e-8]

	def __repr__(self):
		if self.ops:
			return " + ".join(["{} tr {}".format(c, s) for c, s in self.ops])
		else:
			return "0"

	def __neg__(self):
		return TraceOperator(self.mat_info, [(-c, s) for c, s in self.ops])

	def components(self):
		# generator for individual terms in the sum
		for c, s in self.ops:
			yield (c, s)

	def rewrite(self, matrices):
		"""
		Rewrite the operator string into the given matrix basis.
		Arguments:
			matrices (string): the matrix basis to convert to, must be in self.mat_info.matrices
		Returns:
			res (TraceOperator): the result
		"""
		assert matrices in self.mat_info.matrices, "unrecognized matrix basis %s" % matrices
		# self.mat_info.matrices[0][i] = sum_j table[i, j] * matrices[j]
		table = np.linalg.inv(self.mat_info.coefficients[self.mat_info.matrices.index(matrices)])
		in_process = self.ops
		final = {}
		while in_process:
			result = []
			for c, s in in_process:
				to_be_converted = [i for i, m in enumerate(s) if m not in matrices]
				if not to_be_converted:
					# finished
					final[s] = final.get(s, 0) + c
				else:
					# convert the first unconverted matrix
					pos = min(to_be_converted)
					coef = np.dot(self.mat_info.get_coefficients(s[pos]), table)
					for i, m in enumerate(matrices):
						result.append((c * coef[i], s[:pos] + m + s[pos+1:]))
			in_process = result
		return TraceOperator(self.mat_info, [(final[s], s) for s in final])

	def commutator(self, other):
		"""
		Compute the commutator [self, other].
		Arguments:
			self, other (TraceOperator): matrix trace operators; must have the same mat_info
		Returns:
			res (TraceOperator): the quantum commutator [self, other].
		"""
		assert self.mat_info is other.mat_info, "cannot take commutator of different matrix types"
		res = []
		for c1, s1 in self.components():
			for c2, s2 in other.components():
				for i, m1 in enumerate(s1):
					for j, m2 in enumerate(s2):
						for o1 in s1[:i]:
							for o2 in s1[i+1:]:
								assert np.abs(self.mat_info.commutator(o1, o2)) < 1e-8, "commutator not supported for operator %s" % self
						s = s2[:j] + s1[i+1:] + s1[:i] + s2[j+1:]
						res.append((c1 * c2 * self.mat_info.commutator(m1, m2), s))
		return TraceOperator(self.mat_info, res)


if __name__ == "__main__":
	# test basic functions
	g, h = 0, 0
	hamil = TraceOperator(mat_info, [(1, "PP"), (1, "XX"), (1, "QQ"), (1, "YY"), (-2*g, "XYXY"), (2*g, "XXYY"), (h, "XXXX"), (h, "YYYY"), (2*h, "XXYY")])
	hamil_rewrite = hamil.rewrite("ABCD")
	_hamil = hamil_rewrite.rewrite("PXQY")
	print("H =", hamil)
	print("-H =", -hamil)
	print("H =", hamil_rewrite)
	print("H =", _hamil)
	print("H =", _hamil.rewrite("ABCD"))
	# test commutators
	print("[H, H] =", hamil.commutator(hamil))
	rot = TraceOperator(mat_info, [(1, "XQ"), (-1, "YP")])
	print("S =", rot)
	print("S =", rot.rewrite("ABCD"))
	print("[H, S] =", hamil.commutator(rot))
	print("[S, H] =", rot.commutator(hamil))
	for s in "ABCD":
		op = TraceOperator(mat_info, s)
		comm = hamil.commutator(op).rewrite("ABCD")
		print("[H, {}] = {}".format(op, comm))
	for s in "ABCD":
		op = TraceOperator(mat_info, s)
		comm = rot.commutator(op).rewrite("ABCD")
		print("[S, {}] = {}".format(op, comm))

