# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import cProfile
import io
import logging
import os
import pstats
import re
from pstats import Stats

from snakeviz.stats import json_stats, table_rows
from tornado import template

module_found = True
try:
    import snakeviz
except ImportError:
    module_found = False

snakeviz_dir = os.path.dirname(os.path.abspath(snakeviz.__file__))
snakeviz_templates_dir = os.path.join(snakeviz_dir, "templates")


def _from_pstat_to_static_html(stats: Stats, html_filename: str):
    """
    Parses pstats data and populates viz.html template stored under templates dir.
    This utility allows to export html file without kicking off webserver.

    Note that it relies js scripts stored at rawgit cdn. This is not super
    reliable, however it does allow one to not have to rely on webserver and
    local rendering. On the other hand, for local rendering please follow
    the main snakeviz tutorial

    Inspiration for this util is from https://gist.github.com/jiffyclub/6b5e0f0f05ab487ff607.

    Args:
        stats: Stats generated from cProfile data
        html_filename: Output filename in which populated template is rendered
    """
    RESTR = r'(?<!] \+ ")/static/'
    REPLACE_WITH = "https://cdn.rawgit.com/jiffyclub/snakeviz/v0.4.2/snakeviz/static/"

    if not isinstance(html_filename, str):
        raise ValueError("A valid file name must be provided.")

    viz_html_loader = template.Loader(snakeviz_templates_dir)
    html_bytes_renderer = viz_html_loader.load("viz.html")
    file_split = html_filename.split(".")
    if len(file_split) < 2:
        raise ValueError(
            f"\033[0;32;40m Provided filename \033[0;31;47m {html_filename} \033[0;32;40m does not contain . separator."
        )
    profile_name = file_split[0]
    html_bytes = html_bytes_renderer.generate(
        profile_name=profile_name,
        table_rows=table_rows(stats),
        callees=json_stats(stats),
    )
    html_string = html_bytes.decode("utf-8")
    html_string = re.sub(RESTR, REPLACE_WITH, html_string)
    with open(html_filename, "w") as f:
        f.write(html_string)


class CProfilerFlameGraph:
    def __init__(self, filename: str):
        if not module_found:
            raise Exception(
                "Please install snakeviz to use CProfilerFlameGraph. Follow cprofiler_flamegraph.md for more information."
            )
        self.filename = filename

    def __enter__(self):
        self.pr = cProfile.Profile()
        self.pr.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logging.error("Exception occurred", exc_info=(exc_type, exc_val, exc_tb))

        self.pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(self.pr, stream=s)
        _from_pstat_to_static_html(ps, self.filename)
