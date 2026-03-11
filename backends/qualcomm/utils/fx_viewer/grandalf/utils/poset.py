# This code is part of Grandalf
# Copyright (C) 2008-2015 Axel Tillequin (bdcht3@gmail.com) and others
# published under GPLv2 license or EPLv1 license
# Contributor(s): Axel Tillequin

from collections import OrderedDict

__all__ = ["Poset"]

# ------------------------------------------------------------------------------
# Poset class implements a set but allows to interate over the elements in a
# deterministic way and to get specific objects in the set.
# Membership operator defaults to comparing __hash__  of objects but Poset
# allows to check for __cmp__/__eq__ membership by using contains__cmp__(obj)
class Poset(object):
    def __init__(self, L):
        self.o = OrderedDict()
        for obj in L:
            self.add(obj)

    def __repr__(self):
        return "Poset(%r)" % (self.o,)

    def __str__(self):
        f = "%%%dd" % len(str(len(self.o)))
        s = []
        for i, x in enumerate(self.o.values()):
            s.append(f % i + ".| %s" % repr(x))
        return "\n".join(s)

    def add(self, obj):
        if obj in self:
            return self.get(obj)
        else:
            self.o[obj] = obj
            return obj

    def remove(self, obj):
        if obj in self:
            obj = self.get(obj)
            del self.o[obj]
            return obj
        return None

    def index(self, obj):
        return list(self.o.values()).index(obj)

    def get(self, obj):
        return self.o.get(obj, None)

    def __getitem__(self, i):
        return list(self.o.values())[i]

    def __len__(self):
        return len(self.o)

    def __iter__(self):
        for obj in iter(self.o.values()):
            yield obj

    def __cmp__(self, other):
        s1 = set(other.o.values())
        s2 = set(self.o.values())
        return cmp(s1, s2)

    def __eq__(self, other):
        s1 = set(other.o.values())
        s2 = set(self.o.values())
        return s1 == s2

    def __ne__(self, other):
        s1 = set(other.o.values())
        s2 = set(self.o.values())
        return s1 != s2

    def copy(self):
        return Poset(self.o.values())

    __copy__ = copy

    def deepcopy(self):
        from copy import deepcopy

        L = deepcopy(self.o.values())
        return Poset(L)

    def __or__(self, other):
        return self.union(other)

    def union(self, other):
        p = Poset([])
        p.o.update(self.o)
        p.o.update(other.o)
        return p

    def update(self, other):
        self.o.update(other.o)

    def __and__(self, other):
        s1 = set(self.o.values())
        s2 = set(other.o.values())
        return Poset(s1.intersection(s2))

    def intersection(self, *args):
        p = self
        for other in args:
            p = p & other
        return p

    def __xor__(self, other):
        s1 = set(self.o.values())
        s2 = set(other.o.values())
        return Poset(s1.symmetric_difference(s2))

    def symmetric_difference(self, *args):
        p = self
        for other in args:
            p = p ^ other
        return p

    def __sub__(self, other):
        s1 = set(self.o.values())
        s2 = set(other.o.values())
        return Poset(s1.difference(s2))

    def difference(self, *args):
        p = self
        for other in args:
            p = p - other
        return p

    def __contains__(self, obj):
        return obj in self.o

    def contains__cmp__(self, obj):
        return obj in self.o.values()

    def issubset(self, other):
        s1 = set(self.o.values())
        s2 = set(other.o.values())
        return s1.issubset(s2)

    def issuperset(self, other):
        s1 = set(self.o.values())
        s2 = set(other.o.values())
        return s1.issuperset(s2)

    __le__ = issubset
    __ge__ = issuperset

    def __lt__(self, other):
        return self <= other and len(self) != len(other)

    def __gt__(self, other):
        return self >= other and len(self) != len(other)
