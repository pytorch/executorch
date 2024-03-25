# AOT Util

Ahead-of-time (AOT) utility library. Contains native code used by the AOT lowering and delegation logic. Note 
that this library should build independently of the runtime code, and as such, should not have dependencies 
on runtime targets.

This library is intended to be built and distributed as part of the Python pip package, such that it can be
loaded by AOT Python code.

