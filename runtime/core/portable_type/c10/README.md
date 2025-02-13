We added an extra c10 directory so that runtime/core/portable_type/c10
can be the directory to put on your include path, rather than
runtime/core/portable_type, because using runtime/core/portable_type
would cause all headers in that directory to be includeable with
`#include <foo.h>`. In particular, that includes
runtime/core/portable_type/complex.h, which would shadow the C99
complex.h standard header.
