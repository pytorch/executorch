# aoti_cuda_shims.lib

Import library for `aoti_cuda_shims.dll`. Lowering a CUDA model for a Windows target
links the generated wrapper against this file, so it has to advertise every shim name
the wrapper can reference. It is checked in because the build that produces the DLL
does not run on the machines that lower for Windows.

Regenerate it whenever a shim is added, which in practice means whenever the PyTorch
pin moves and the generated wrapper starts calling something new:

```
nm --defined-only aoti_cuda_shims.lib \
  | sed -n 's/.* T _\?\(aoti_torch_[a-z_0-9]*\)$/\1/p' \
  | sort -u > exports.txt
# add the new names to exports.txt, then
{ echo 'LIBRARY aoti_cuda_shims.dll'; echo 'EXPORTS'; sed 's/^/    /' exports.txt; } \
  > aoti_cuda_shims.def
x86_64-w64-mingw32-dlltool -d aoti_cuda_shims.def -l aoti_cuda_shims.lib \
  --dllname aoti_cuda_shims.dll
# zero the archive metadata so the file is reproducible
mkdir extract && cd extract && x86_64-w64-mingw32-ar x ../aoti_cuda_shims.lib \
  && rm -f ../aoti_cuda_shims.lib \
  && x86_64-w64-mingw32-ar rcsD ../aoti_cuda_shims.lib $(ls | sort)
```

The archive has to be built at a fresh path. Updating it in place keeps the reverse
member order the generator produced, so the same export list would not give the same
bytes twice.

Then check the result is a superset of what it replaced and that no member carries a
timestamp, since this file ships in the wheel and a stamped one makes two builds of
the same source differ.

A name only resolves if something in the DLL defines it. The DLL is built from the
CUDA shims plus the SlimTensor common shims, so a shim added to the ETensor common
shims will link and then fail to load.
