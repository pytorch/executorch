#!/usr/bin/env bash
set -eu

ethos_u_dir=${1:-/tmp/ethos-u}
script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

function patch_repo() {
    echo -e "\nPreparing ${name}..."
    cd ${ethos_u_dir}/${name}
   
    git reset --hard ${base_rev}
    
    patch_dir=${script_dir}/${name}/patches/
    [[ -e ${patch_dir} && $(ls -A ${patch_dir}) ]] && \
        git am -3 ${patch_dir}/*.patch
    
    echo -e "Patched ${name} @ $(git describe --all --long 2> /dev/null) in ${ethos_u_dir}/${name} dir.\n"
}

name="core_platform"
base_rev=204210b1074071532627da9dc69950d058a809f4
patch_repo 

name="core_software"
base_rev=74c514a5b50a19197a64a86095bc0429188adcbe
patch_repo 

exit $?
