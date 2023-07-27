import contextlib
import functools
import logging
import os
import selectors
import subprocess
import sys
from typing import List, Tuple

# If quiet is true, suppress the printing of stdout and stderr output.
quiet = False


def execute_cmd(cmd: List[str]) -> Tuple[str, str]:
    """
    `subprocess.run(cmd, capture_output=True)` captures stdout/stderr and only
    returns it at the end. This functions not only does that, but also prints out
    stdout/stderr non-blockingly when running the command.
    """
    logging.debug(f"cmd = \33[33m{cmd}\33[0m, cwd = {os.getcwd()}")
    stdout = ""
    stderr = ""

    PIPE = subprocess.PIPE
    with subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE) as p:
        sel = selectors.DefaultSelector()
        if p.stdout is not None:
            sel.register(p.stdout, selectors.EVENT_READ)
        if p.stderr is not None:
            sel.register(p.stderr, selectors.EVENT_READ)

        done = False
        while not done:
            for key, _ in sel.select():
                # pyre-ignore Undefined attribute [16]: Item `_typeshed.HasFileno` of
                # `typing.Union[_typeshed.HasFileno, int]` has no attribute `read1`.
                data = key.fileobj.read1().decode()
                if not data:
                    done = True
                    break

                if key.fileobj is p.stdout:
                    if not quiet:
                        print(data, end="")
                    stdout += data
                else:
                    if not quiet:
                        print(data, end="", file=sys.stderr)
                    stderr += data

    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, p.args, stdout, stderr)

    return stdout, stderr


@contextlib.contextmanager
def change_directory(path: str):
    # record cwd (current working directory)
    cwd = os.getcwd()
    try:
        os.chdir(path)
        yield os.getcwd()
    finally:
        # restore cwd
        os.chdir(cwd)


@functools.lru_cache(maxsize=None)
def get_hg_root():
    def _get_hg_root():
        return execute_cmd(["hg", "root"])[0].rstrip()

    try:
        hg_root = _get_hg_root()
    except subprocess.CalledProcessError as e:
        if "not inside a repository" not in e.stderr:
            raise e

        # If we get "not inside a repository" error, try again with $HOME/fbsource,
        # which is the most common hg root
        possible_hg_root = os.path.join(os.environ["HOME"], "fbsource")
        logging.warning(f"{e}. Trying again with {possible_hg_root}")

        with change_directory(possible_hg_root):
            hg_root = _get_hg_root()

    return hg_root
