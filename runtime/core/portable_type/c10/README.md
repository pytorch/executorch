This directory contains header files from `c10` in PyTorch core that
need to be used in ExecuTorch core. They are copied here rather than
being found through the torch pip package to keep the core build
hermetic for embedded use cases. The headers should be exact copies
from PyTorch core; if they are out of sync, please send a PR!

We added an extra c10 directory so that `runtime/core/portable_type/c10`
can be the directory to put on your include path, rather than
`runtime/core/portable_type`, because using `runtime/core/portable_type`
would cause all headers in that directory to be includeable with
`#include <foo.h>`. In particular, that includes
`runtime/core/portable_type/complex.h`, which would shadow the C99
`complex.h` standard header.
