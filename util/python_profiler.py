import cProfile, io, pstats
import logging
from pstats import SortKey

module_found = True
try:
    import snakeviz
    from snakeviz.export_static_html import from_pstat_to_static_html
except ImportError:
    module_found = False


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
        from_pstat_to_static_html(ps, self.filename)
