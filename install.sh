#!/bin/bash

# Creates a pip package named "executorch" containing Executorch python modules.

set -o errexit
set -o pipefail
set -o nounset

PIP="${PIP:=pip}"

main() {
  if [ ! -d 'executorch' ]; then
    echo "ERROR: Must be run from the parent of an 'executorch' subdir" >&2
    exit 1
  fi

  # Create a temp directory.
  local pip_root
  # Works on mac or linux.
  pip_root="$(mktemp -d 2>/dev/null || mktemp -d -t 'et-pip')"
  echo "Working dir: ${pip_root}" >&2

  local et_root="${pip_root}/src/executorch"

  # Create a temporary tree containing all the executorch/ files,
  # except with the src layout. It'll look something like:
  #
  # ${pip_root}
  # |   pyproject.toml
  # |__ src/
  #     |__ executorch/
  #         |__ exir/
  #         |   |...
  #         |
  #         |__ backends/
  #             |...
  #
  # This way the pip package user will be able to `import executorch.exir`
  # and `import executorch.backends`.

  mkdir -p "${et_root}"
  rsync -r --exclude='.*' "executorch/" "${et_root}"
  mv "${et_root}/pyproject.toml" "${pip_root}"

  # We need to copy over the schema.fbs files into executorch/exir/serialize/...
  # since the `serialize_to_flatbuffer` function looks for the schema.fbs files
  # in that directory.
  cp "${et_root}/schema/"*.fbs "${et_root}/exir/serialize/"

  # Uninstall older pip package if present.
  "${PIP}" uninstall -y executorch

  # Install the tree as a pip package.
  cd "${et_root}/../../"
  "${PIP}" install .

  # Clean up.
  rm -rf "${pip_root}"
}

main "$@"
