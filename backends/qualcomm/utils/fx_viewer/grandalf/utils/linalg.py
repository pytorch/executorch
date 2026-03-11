from math import sqrt
from array import array as _array

constants = (int, float)


def coerce_(types):
    if types is None:
        types = []
    if str in types:
        raise TypeError
    if complex in types:
        raise TypeError
    dtype = ("i", int)
    if float in types:
        dtype = ("d", float)
    return dtype


def _mkslice(i, n):
    if not isinstance(i, slice):
        i = slice(i, i + 1, 1)
    start, stop, stride = i.indices(n)
    return slice(start, stop, stride)


def make_ij_slices(f):
    def wrapper(self, ij, *args):
        # I = slice(0,self.n,1)
        J = slice(0, self.p, 1)
        if isinstance(ij, tuple):
            I = _mkslice(ij[0], self.n)
            J = _mkslice(ij[1], self.p)
        else:
            I = _mkslice(ij, self.n)
        return f(self, (I, J), *args)

    return wrapper


# minimalistic numpy.array replacement class used as fallback
# when numpy is not found in geometry module
class array(object):
    def __init__(self, data, dtype=None, copy=True):
        self.dim = len(data)
        types = None
        if self.dim > 0:
            types = set([type(x) for x in data])
        if dtype is not None:
            types = (dtype,)
        tc, self.dtype = coerce_(types)
        data = [self.dtype(x) for x in data]
        if copy is True:
            self.data = _array(tc, data)
        else:
            raise NotImplementedError

    def coerce(self, dtype):
        data = [dtype(x) for x in self.data]
        tc, dtype = coerce_((dtype,))
        self.data = _array(tc, data)
        self.dtype = dtype

    @property
    def typecode(self):
        return self.data.typecode

    def __len__(self):
        return self.dim

    def __str__(self):
        s = " ".join(("%.12s" % x).ljust(12) for x in self)
        return "[%s]" % s.strip()

    def copy(self):
        return array(self.data, self.dtype)

    def __add__(self, v):
        if isinstance(v, constants):
            v = array([v] * self.dim)
        assert v.dim == self.dim
        return array([x + y for (x, y) in zip(self.data, v.data)])

    def __sub__(self, v):
        if isinstance(v, constants):
            v = array([v] * self.dim)
        assert v.dim == self.dim
        return array([x - y for (x, y) in zip(self.data, v.data)])

    def __neg__(self):
        return array([-x for x in self.data], dtype=self.dtype)

    def __radd__(self, v):
        return self + v

    def __rsub__(self, v):
        return (-self) + v

    def dot(self, v):
        assert v.dim == self.dim
        return sum([x * y for (x, y) in zip(self.data, v.data)])

    def __rmul__(self, k):
        return array([k * x for x in self.data])

    def __mul__(self, v):
        if isinstance(v, constants):
            v = array([v] * self.dim)
        assert v.dim == self.dim
        return array([x * y for (x, y) in zip(self.data, v.data)])

    def __truediv__(self, v):
        if isinstance(v, constants):
            v = array([v] * self.dim)
        assert v.dim == self.dim
        return array([x / y for (x, y) in zip(self.data, v.data)])

    __div__ = __truediv__

    def __rtruediv__(self, v):
        if isinstance(v, constants):
            v = array([v] * self.dim)
        assert v.dim == self.dim
        return array([x / y for (x, y) in zip(v.data, self.data)])

    __rdiv__ = __rtruediv__

    def __floordiv__(self, v):
        if isinstance(v, constants):
            v = array([v] * self.dim)
        assert v.dim == self.dim
        return array([x // y for (x, y) in zip(self.data, v.data)])

    def __rfloordiv__(self, v):
        if isinstance(v, constants):
            v = array([v] * self.dim)
        assert v.dim == self.dim
        return array([x // y for (x, y) in zip(v.data, self.data)])

    def norm(self):
        return sqrt(self.dot(self))

    def max(self):
        return max(self.data)

    def min(self):
        return min(self.data)

    def __iter__(self):
        for x in self.data:
            yield x

    def __setitem__(self, i, v):
        assert isinstance(i, int)
        self.data[i] = self.dtype(v)

    def __getitem__(self, i):
        i = _mkslice(i, self.dim)
        res = self.data[i]
        if len(res) == 1:
            return res[0]
        return array(res)

    def transpose(self):
        return matrix(self.data, self.dtype)

    def __float__(self):
        assert self.dim == 1
        return float(self.data[0])


# ------------------------------------------------------------------------------
# minimalistic numpy.matrix replacement class used as fallback
# when numpy is not found in geometry module
class matrix(object):
    def __init__(self, data, dtype=None, copy=True, transpose=False):
        # check input data types:
        types = set([type(v) for v in data])
        if len(types) > 1:
            raise TypeError
        t = types.pop()
        # import data:
        if t in constants:
            self.data = [array(data, dtype, copy)]
        else:
            if transpose:
                data = zip(*data)
            self.data = [array(v, dtype, copy) for v in data]
        # define matrix sizes:
        self.n = len(self.data)
        sizes = set([len(v) for v in self.data])
        if len(sizes) > 1:
            raise ValueError
        self.p = sizes.pop()
        if dtype is None:
            # coerce types of arrays of matrix:
            types = set([v.dtype for v in self.data])
            tc, dtype = coerce_(types)
            for v in self.data:
                v.coerce(dtype)
        self.dtype = dtype

    def __len__(self):
        return self.n * self.p

    def __str__(self):
        s = "\n ".join([str(v) for v in self.data])
        return "[%s]" % s.strip()

    @property
    def shape(self):
        return (self.n, self.p)

    def lvecs(self):
        return self.data

    def cvecs(self):
        return [array(v, self.dtype) for v in zip(*self.data)]

    def copy(self):
        return matrix(self.data, self.dtype)

    def transpose(self):
        return matrix(self.data, dtype=self.dtype, transpose=True)

    def sum(self):
        return sum([sum(v) for v in self.data])

    @make_ij_slices
    def __getitem__(self, ij):
        I, J = ij
        l = self.lvecs()[I]
        m = matrix([v[J] for v in l])
        if m.n == 1:
            v = m.data[0]
            if v.dim == 1:
                return v[0]
            if len(l) > 1:
                return v
        return m

    @make_ij_slices
    def __setitem__(self, ij, v):
        I, J = ij
        Ri = range(I.start, I.stop, I.step)
        Rj = range(J.start, J.stop, J.step)
        if type(v) in constants:
            v = (v,)
        value = (x for x in v)
        for i in Ri:
            for j in Rj:
                self.data[i][j] = next(value)

    def __add__(self, m):
        if isinstance(m, constants):
            return matrix([u + m for u in self.data])
        else:
            assert self.shape == m.shape
            return matrix([u + v for (u, v) in zip(self.data, m.data)])

    def __sub__(self, m):
        if isinstance(m, constants):
            return matrix([u - m for u in self.data])
        else:
            assert self.shape == m.shape
            return matrix([u - v for (u, v) in zip(self.data, m.data)])

    def __neg__(self):
        return matrix([-x for x in self.data], dtype=self.dtype)

    def __float__(self):
        assert self.n == 1 and self.p == 1
        return self[0, 0]

    def __radd__(self, v):
        return self + v

    def __rsub__(self, v):
        return (-self) + v

    def __rmul__(self, k):
        if not isinstance(k, constants):
            raise TypeError
        return matrix([k * v for v in self.data])

    def __mul__(self, X):
        if isinstance(X, constants):
            return X * self
        if isinstance(X, array):
            assert X.dim == self.p
            return array([v.dot(X) for v in self.data])
        if isinstance(X, matrix):
            assert X.n == self.p
            return matrix([self * v for v in X.cvecs()])

    def __pow__(self, v):
        S = [self] * v
        assert len(S) > 0
        return reduce(lambda x, y: x * y, S)

    def __iter__(self):
        for l in self.data:
            for v in l:
                yield v


class SimplexMin(object):
    def __init__(self, A, b, c):
        self.A = A
        self.b = b
        self.c = c
        self.tableau()

    def tableau(self):
        self.T = []

    def setup(self):
        self.enter = []
        delf.outer = []
