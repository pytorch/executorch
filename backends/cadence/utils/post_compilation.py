#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

XTENSA_ROOT = os.environ["XTENSA_TOOLCHAIN"]
XTENSA_VER = os.environ["TOOLCHAIN_VER"]
XTENSA_TOOLS = os.path.join(XTENSA_ROOT, f"{XTENSA_VER}/XtensaTools")
XTENSA_SYSTEM = os.path.join(XTENSA_TOOLS, "config")
XTENSA_CORE = os.environ["XTENSA_CORE"]


def parse_sections(dsp_dir, dsp_exe):
    """
    >>> xt-size dsp_mu_polling_hifi4 --radix=16 -A
    dsp_mu_polling_hifi4  :
    section                           size         addr
    .UserExceptionVector.literal       0x4   0x24000000
    .ResetVector.text                0x14c   0x24020000
    .WindowVectors.text              0x16c   0x24020400
    .Level2InterruptVector.text        0x8   0x2402057c
    .Level3InterruptVector.text        0x8   0x2402059c
    .DebugExceptionVector.text         0xc   0x240205bc
    .NMIExceptionVector.text           0x4   0x240205dc
    .KernelExceptionVector.text        0x8   0x240205fc
    .UserExceptionVector.text          0xc   0x2402061c
    .DoubleExceptionVector.text        0x8   0x2402063c
    .rodata                          0x398     0x200000
    .text                           0x4c6c     0x2003a0
    .clib.data                         0x4     0x20500c
    .rtos.percpu.data                0x310     0x205010
    .data                            0x880     0x205320
    .bss                             0x6b8     0x205ba0
    .debug_aranges                   0x2c8          0x0
    .debug_info                     0x4367          0x0
    .debug_abbrev                    0xf54          0x0
    .debug_line                     0x36e9          0x0
    .debug_frame                     0x1f4          0x0
    .debug_str                      0x1986          0x0
    .debug_loc                      0x15ba          0x0
    .xt.prop                        0x4a10          0x0
    .xt.lit                          0x110          0x0
    .xtensa.info                     0x218          0x0
    .comment                          0x5f          0x0
    .debug_ranges                    0x100          0x0
    Total                          0x1717f
    """
    cmd = f"{XTENSA_TOOLS}/bin/xt-size {dsp_dir}/{dsp_exe} --radix=16 -A"
    print(f"Executing command:\n {cmd}\n")
    p = subprocess.run(cmd.split(" "), capture_output=True)
    print(p.stdout.decode())
    lines = p.stdout.decode().strip().split("\n")
    print(lines)
    lines = [line for line in lines if line != ""]
    lines = lines[2:-1]
    lines = "\n".join(lines)

    f = io.StringIO(lines)
    reader = csv.reader(f, delimiter=" ", skipinitialspace=True)
    ret_list = [(section, int(addr, 16)) for section, size, addr in reader]
    print(lines)
    print(ret_list)
    return ret_list


def xt_objcopy_sections(dsp_dir, dsp_exe, output_obj_path, obj_name, sections):
    xt_objcopy = f"{XTENSA_TOOLS}/bin/xt-objcopy"
    xtensa_args = f"--xtensa-system={XTENSA_SYSTEM} --xtensa-core={XTENSA_CORE}"

    cmd = f"{xt_objcopy} {xtensa_args} -O binary {dsp_dir}/{dsp_exe} {output_obj_path}/{obj_name} "
    cmd += " ".join([f"--only-section={section}" for section in sections])

    print(cmd)
    subprocess.run(cmd.split(" "), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dsp_exe_path", help="Xtensa DSP executable path")
    parser.add_argument("output_dir", help="output directory")
    args = parser.parse_args()

    dsp_exe_dir = os.path.dirname(args.dsp_exe_path)
    dsp_exe = os.path.basename(args.dsp_exe_path)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    sections = parse_sections(dsp_exe_dir, dsp_exe)

    DATA_SECTION_START = 0x200000
    NCACHE_SECTION_START = 0x20060000
    TEXT_SECTION_START = 0x24000000

    ncache_sections = [
        section
        for section, addr in sections
        if addr >= NCACHE_SECTION_START and addr < TEXT_SECTION_START
    ]

    data_sections = [
        section
        for section, addr in sections
        if addr < NCACHE_SECTION_START and addr >= DATA_SECTION_START
    ]

    text_sections = [
        section for section, addr in sections if addr >= TEXT_SECTION_START
    ]

    if len(ncache_sections):
        dirpath = tempfile.mkdtemp()
        xt_objcopy_sections(
            dsp_exe_dir,
            dsp_exe,
            dirpath,
            "dsp_ncache_release.bin",
            ncache_sections,
        )
        # copy files over
        shutil.copyfile(
            os.path.join(dirpath, "dsp_ncache_release.bin"),
            os.path.join(args.output_dir, "dsp_ncache_release.bin"),
        )
        shutil.rmtree(dirpath)

    dirpath = tempfile.mkdtemp()

    xt_objcopy_sections(
        dsp_exe_dir,
        dsp_exe,
        dirpath,
        "dsp_text_release.bin",
        text_sections,
    )

    # copy files over
    shutil.copyfile(
        os.path.join(dirpath, "dsp_text_release.bin"),
        os.path.join(args.output_dir, "dsp_text_release.bin"),
    )
    shutil.rmtree(dirpath)

    dirpath = tempfile.mkdtemp()

    xt_objcopy_sections(
        dsp_exe_dir,
        dsp_exe,
        dirpath,
        "dsp_data_release.bin",
        data_sections,
    )

    # copy files over
    shutil.copyfile(
        os.path.join(dirpath, "dsp_data_release.bin"),
        os.path.join(args.output_dir, "dsp_data_release.bin"),
    )
    shutil.rmtree(dirpath)


if __name__ == "__main__":
    main()
