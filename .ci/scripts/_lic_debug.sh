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
echo "--- core config fingerprints (is the key input identical?)"
CORE_DIR=$(dirname "$(dirname "${XTENSAD_LICENSE_FILE}")")/$(basename "$(dirname "$(dirname "${XTENSAD_LICENSE_FILE}")")")
CORE_DIR=$(dirname "$(dirname "${XTENSAD_LICENSE_FILE}")")
echo "CORE_DIR=${CORE_DIR}"
echo "core dir aggregate: $(find "${CORE_DIR}" -type f -print0 2>/dev/null | sort -z | xargs -0 md5sum 2>/dev/null | md5sum)"
echo "core file count:    $(find "${CORE_DIR}" -type f 2>/dev/null | wc -l)"
for f in "${CORE_DIR}"/config/*-params "${XTENSA_SYSTEM}"/*-params; do
  [ -f "$f" ] && echo "  params $(md5sum "$f")"
done
TC="${XTENSA_TOOLCHAIN}/${TOOLCHAIN_VER}/XtensaTools"
echo "toolchain lic dir:  $(find "${TC}/Tools/lic" -type f -print0 2>/dev/null | sort -z | xargs -0 md5sum 2>/dev/null | md5sum)"
echo "xt-clang binary:    $(md5sum "${TC}/bin/xt-clang" 2>/dev/null)"
echo "--- cpu"
grep -m1 vendor_id /proc/cpuinfo; grep -m1 "model name" /proc/cpuinfo
echo "--- core resolution"
echo "XTENSA_CORE=${XTENSA_CORE:-unset}"
echo "XTENSA_SYSTEM=${XTENSA_SYSTEM:-unset}"
xt-run --show-config=cores 2>&1 | head -8
echo "--- direct compile attempt"
echo 'int main(void){return 0;}' > /tmp/_lic_t.c
xt-clang -c /tmp/_lic_t.c -o /tmp/_lic_t.o 2>&1 | head -20
echo "compile rc=$?"
echo "--- same compile with FLEXLM_DIAGNOSTICS=3"
FLEXLM_DIAGNOSTICS=3 LM_BORROW="" xt-clang -c /tmp/_lic_t.c -o /tmp/_lic_t.o 2>&1 | head -60
echo "diag compile rc=$?"
echo "=== end flexnet diagnostic ==="
