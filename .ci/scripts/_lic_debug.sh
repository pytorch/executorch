#!/usr/bin/env bash
# TEMPORARY diagnostic: why XTENSA_XCC_TIE fails to check out on OSDC but not
# on EC2. Prints host identity and licence metadata only -- never a SIGN= key.
set -u

echo "=== flexnet diagnostic ==="
echo "--- host identity"
echo "hostname:        $(hostname)"
echo "hostname length: ${#HOSTNAME}"
echo "hostname -f:     $(hostname -f 2>&1)"
echo "id:              $(id)"
echo "--- interfaces"
for n in /sys/class/net/*/address; do
  echo "  $n = $(cat "$n" 2>/dev/null)"
done
echo "--- licence files on the search path"
for f in $(echo "${XTENSAD_LICENSE_FILE}:${XTENSA_TOOLCHAIN}/${TOOLCHAIN_VER}/XtensaTools/Tools/lic/license.dat" | tr ':' ' '); do
  echo "== $f"
  ls -l "$f" 2>&1
  md5sum "$f" 2>/dev/null
  awk '/^(FEATURE|INCREMENT)/ { print "   feature:", $1, $2, $3, $4, $5 }' "$f" 2>/dev/null | sort -u
  grep -o 'HOSTID=[^ ]*' "$f" 2>/dev/null | sort -u | sed 's/^/   /'
done
echo "--- core resolution"
echo "XTENSA_CORE=${XTENSA_CORE:-unset}"
echo "XTENSA_SYSTEM=${XTENSA_SYSTEM:-unset}"
xt-run --show-config=cores 2>&1 | head -8
echo "--- direct compile attempt"
echo 'int main(void){return 0;}' > /tmp/_lic_t.c
xt-clang -c /tmp/_lic_t.c -o /tmp/_lic_t.o 2>&1 | head -20
echo "compile rc=$?"
echo "=== end flexnet diagnostic ==="
