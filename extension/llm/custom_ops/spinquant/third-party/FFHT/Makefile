CC = gcc
CFLAGS = -O3 -march=native -std=c99 -pedantic -Wall -Wextra -Wshadow -Wpointer-arith -Wcast-qual -Wstrict-prototypes -Wmissing-prototypes

all: test_float test_double fast_copy.o fht.o

OBJ := fast_copy.o fht.o

%.o: %.c
	$(CC) $< -o $@ -c $(CFLAGS)

test_%: test_%.c $(OBJ)
	$(CC) $< $(OBJ) -o $@ $(CFLAGS)

test_double_header_only: test_double_header_only.c
	$(CC) $< -o $@ $(CFLAGS)

test_float_header_only: test_double_header_only.c
	$(CC) $< -o $@ $(CFLAGS)

clean:
	rm -f test_float test_double test_float_header_only test_double_header_only $(OBJ)
