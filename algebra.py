from utility import debug
import numpy as np
import scipy.sparse as sparse
import sparseqr


def null_space(mat):
	"""
	Computes the right null space of a sparse matrix.
	Arguments:
		mat (sparse matrix): the matrix to compute, of shape (M, N)
	Returns:
		null (sparse matrix): a basis of the null space, of shape (N, K)
	"""
	debug("null_space: input shape {}".format(mat.shape))
	q, _, _, rank = sparseqr.qr(mat.transpose())
	null = sparse.csc_matrix(q)[:, rank:]
	# sanity check
	debug("Error: {}".format(mat.dot(null).max()))
	assert null.shape[0] == mat.shape[1], "dimension mismatch"
	debug("null_space: solution shape {}".format(null.shape))
	return null

def range_space(mat):
	"""
	Computes the row space of a sparse matrix.
	Arguments:
		mat (sparse matrix): the matrix to compute, of shape (M, N)
	Returns:
		r (sparse matrix): a basis of the row space, of shape (K, N)
	"""
	debug("range_space: input shape {}".format(mat.shape))
	q, _, _, rank = sparseqr.qr(mat.transpose())
	r = sparse.csc_matrix(q)[:, :rank].transpose()
	# sanity check
	assert r.shape[1] == mat.shape[1], "dimension mismatch"
	debug("range_space: solution shape {}".format(r.shape))
	return r


if __name__ == "__main__":
	identity = sparse.csr_matrix(np.eye(5))
	print(null_space(identity))
	print(range_space(identity))
	zeros = sparse.csr_matrix(np.zeros((5, 5)))
	print(null_space(zeros))
	print(range_space(zeros))