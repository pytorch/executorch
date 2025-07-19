#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

/*
 * We are adding a custom syscall_stubs.c file to provide dummy implementations for syscalls that are not 
 * available on the Pico platform. This is necessary because the Pico does not have an operating system, 
 * and therefore does not support standard C library functions like _exit, _sbrk, _read, etc.
 * By adding these stubs, we can resolve linker errors that occur when building our project for the Pico. 
 * The stubs will be compiled and linked into our final executable, allowing it to run on the target hardware.
*/
#ifdef __cplusplus
extern "C" {
#endif
void *__dso_handle = 0;
int fnmatch(const char *pattern, const char *string, int flags) { return 1; }
ssize_t pread(int fd, void *buf, size_t count, off_t offset) { return -1; }
void _fini(void) {}
void _exit(int status) { while (1) {} }
void* _sbrk(ptrdiff_t incr) { return (void*)-1; }
int _read(int file, char *ptr, int len) { return -1; }
int _write(int file, char *ptr, int len) { return -1; }
int _close(int file) { return -1; }
int _fstat(int file, void *st) { return 0; }
int _lseek(int file, int ptr, int dir) { return 0; }
int _isatty(int file) { return 1; }
int _kill(int pid, int sig) { return -1; }
int _getpid(void) { return 1; }
int _open(const char *name, int flags, int mode) { return -1; }
int _gettimeofday(void *tv, void *tz) { return -1; }
#ifdef __cplusplus
}
#endif
