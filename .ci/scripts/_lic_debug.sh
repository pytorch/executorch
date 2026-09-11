#!/usr/bin/env bash
# TEMPORARY diagnostic: xt-clang (RJ-2025.5) computes licence feature
# XT_XCC_TIE_ED4DB539 on the mt-l-x86iavx512-8-64 pod but XT_XCC_TIE_ED4D1230
# (the one the bundled licence grants) on mt-l-x86iamx-8-64 and on EC2, from
# byte-identical config inputs. Two candidate triggers remain: the host CPU, and
# the hostname (48 chars on the failing label, 45 on the working one, 12 on
# EC2). This probe runs the same compile under faked hostnames to tell them
# apart. Prints host identity and licence metadata only -- never a SIGN= key.
set -u

echo "=== flexnet diagnostic ==="
echo "--- host identity"
echo "hostname:        $(hostname)"
echo "hostname length: ${#HOSTNAME}"
echo "id:              $(id)"
echo "machine-id:      $(cat /etc/machine-id 2>/dev/null)"
echo "boot-id:         $(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"
echo "--- cpu"
grep -m1 vendor_id /proc/cpuinfo
grep -m1 "model name" /proc/cpuinfo
grep -m1 "cpu family" /proc/cpuinfo
grep -m1 "^model" /proc/cpuinfo
grep -m1 stepping /proc/cpuinfo
grep -m1 microcode /proc/cpuinfo
echo "nproc:           $(nproc) / online $(getconf _NPROCESSORS_ONLN)"
grep -m1 flags /proc/cpuinfo | tr ' ' '\n' | grep -xE 'avx512f|avx512_bf16|amx_tile|amx_bf16|sha_ni|vaes|movdir64b|serialize' | tr '\n' ' '
echo
echo "--- licence file features"
awk '/^(FEATURE|INCREMENT)/ { print "   feature:", $2 }' "${XTENSAD_LICENSE_FILE}" 2>/dev/null | sort -u

echo "--- what config id do the tools report?"
xt-run --show-config=config-id 2>&1 | head -3
xt-clang --show-config=config-id 2>&1 | head -3
# An invalid key makes xt-run print the list of valid ones; look for id/licence.
xt-run --show-config=__bogus__ 2>&1 | tr ' ' '\n' | grep -iE 'id$|lic' | sort -u | head -20

echo 'int main(void){return 0;}' > /tmp/_lic_t.c

# gethostname/uname shim so the same binary on the same host sees a different
# hostname. If the requested feature name tracks the fake hostname, the trigger
# is the hostname, not the CPU.
cat > /tmp/_hn.c <<'EOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>

static const char *fake(void) {
  const char *e = getenv("FAKE_HOSTNAME");
  return (e && *e) ? e : "fake";
}

int gethostname(char *name, size_t len) {
  const char *f = fake();
  if (len == 0) return 0;
  strncpy(name, f, len);
  name[len - 1] = '\0';
  return 0;
}

int uname(struct utsname *buf) {
  static int (*real)(struct utsname *);
  if (!real) real = dlsym(RTLD_NEXT, "uname");
  int rc = real(buf);
  if (rc == 0) {
    strncpy(buf->nodename, fake(), sizeof(buf->nodename) - 1);
    buf->nodename[sizeof(buf->nodename) - 1] = '\0';
  }
  return rc;
}
EOF
gcc -shared -fPIC -o /tmp/_hn.so /tmp/_hn.c -ldl 2>&1 | head -5
echo "shim built: $(ls -l /tmp/_hn.so 2>&1)"
echo "shim works: $(FAKE_HOSTNAME=shim-check LD_PRELOAD=/tmp/_hn.so python3 -c 'import socket; print(socket.gethostname())' 2>&1)"

probe() {
  # $1 = label, $2 = FAKE_HOSTNAME ("" = no shim)
  local label="$1" hn="${2:-}" out
  if [[ -z "${hn}" ]]; then
    out=$(FLEXLM_DIAGNOSTICS=3 xt-clang -c /tmp/_lic_t.c -o /tmp/_lic_t.o 2>&1)
  else
    out=$(FAKE_HOSTNAME="${hn}" LD_PRELOAD=/tmp/_hn.so FLEXLM_DIAGNOSTICS=3 \
          xt-clang -c /tmp/_lic_t.c -o /tmp/_lic_t.o 2>&1)
  fi
  printf '  %-28s len=%-3s %s\n' "${label}" "${#hn}" \
    "$(echo "${out}" | grep -oE '(Checkout succeeded: )?XT_XCC_TIE[A-Z0-9_]*' | head -1)"
  echo "${out}" | grep -qE 'Checkout succeeded' && echo "      -> OK" || echo "      -> FAILED"
}

echo "--- feature name vs hostname"
probe "real hostname (no shim)" ""
probe "shim, real hostname"     "$(hostname)"
probe "shim, short"             "shorty"
probe "shim, 45ch (iamx-like)"  "mt-l-x86iamx-8-64-2lfxk-runner-4bg8n-workflow"
probe "shim, 48ch (avx512-like)" "mt-l-x86iavx512-8-64-r6gdl-runner-6525q-workflow"
probe "shim, 47ch"              "mt-l-x86iavx512-8-64-r6gdl-runner-6525-workflow"
probe "shim, 64ch"              "mt-l-x86iavx512-8-64-r6gdl-runner-6525q-workflow-padding-xyz"
echo "=== end flexnet diagnostic ==="
