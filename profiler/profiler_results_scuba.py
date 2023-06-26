import getpass
import hashlib
import uuid

from typing import Dict, List, Optional

from executorch.exir.schema import FrameList
from executorch.exir.serialize import deserialize_from_flatbuffer

from executorch.profiler.parse_profiler_results import (
    frame_list_normal_vector_short,
    frame_list_short_repr,
    get_frame_list,
    ProfileEvent,
)

from libfb.py.fburl import FBUrlError, get_fburl, resolve_fburl
from libfb.py.scuba_url import ScubaDrillstate, ScubaURL
from rfe.scubadata.scubadata_py3 import Sample, ScubaData


def generate_scuba_fburl(scuba_url):
    scuba_url = str(scuba_url)
    key = hashlib.md5(scuba_url.encode()).hexdigest()[:20]
    try:
        fb_url = "https://fburl.com/{}".format(key)
        resolve_fburl(fb_url)
        return fb_url
    except FBUrlError:
        return get_fburl(scuba_url, string_key=key)


def upload_to_scuba(profile_data: Dict[str, List[ProfileEvent]], model_path: str):
    scuba_table = "executorch_profile"
    scuba_data = ScubaData(scuba_table)
    run_id = str(uuid.uuid4())

    program = None
    model_bytes = None
    if model_path is not None:
        with open(model_path, "rb") as model_file:
            model_bytes = model_file.read()
            program = deserialize_from_flatbuffer(model_bytes)

    def frame_list_normal_vector(frame_list: FrameList) -> Optional[List[str]]:
        if frame_list is None:
            return None
        return [
            f'File "{frame.filename}", line {frame.lineno}, in {frame.name}\n{frame.context}\n'
            for frame in frame_list.items
        ]

    execution_plan_idx = 0
    samples = []

    for _, profile_data_list in profile_data.items():
        for d in profile_data_list:
            sample = Sample()
            sample.setTimeColumnNow()
            sample.addNormalValue("run_id", run_id)
            sample.addNormalValue("unixname", getpass.getuser())

            sample.addNormalValue("profile_event_name", d.name)

            sample.addIntValue("execution_plan_idx", execution_plan_idx)
            sample.addIntValue("chain_idx", d.chain_idx)
            sample.addIntValue("instruction_idx", d.instruction_idx)
            sample.addFloatValue("duration", d.duration[0])

            if model_bytes is not None:
                sample.addNormalValue("model_file_path", model_path)
                sample.addIntValue("model_file_size_bytes", len(model_bytes))

            frame_list = get_frame_list(
                program, execution_plan_idx, d.chain_idx, d.instruction_idx
            )
            stacktrace = frame_list_normal_vector(frame_list)
            stacktrace_short = frame_list_normal_vector_short(frame_list)

            sample.addNormVectorValue("model_stacktrace", stacktrace)
            sample.addNormVectorValue("model_stacktrace_short", stacktrace_short)
            sample.addNormalValue("model_frame", frame_list_short_repr(frame_list))

            samples.append(sample)

    start_time = samples[0].time() - 1
    end_time = samples[-1].time() + 1

    try:
        print(f"Uploading to scuba with run_id: {run_id}")
        scuba_data.add_samples(samples)
        drillstate = (
            ScubaDrillstate()
            .setStartTime(str(start_time))
            .setEndTime(str(end_time))
            .setGroupBy(["profile_event_name"])
            .setAggregationField("sum")
            .setMetric("sum")
            .setLimit("20")
            .setOrderColumn("duration")
            .setOrderDesc(True)
            .addEqConstraint("run_id", run_id)
            .addMatchRegexConstraint("profile_event_name", "^native_call.*")
        )
        scuba_url = generate_scuba_fburl(ScubaURL(scuba_table, drillstate, view="Pie"))
        print(f"Successfully uploaded to scuba: {scuba_url}")
        print("It may take up to 5 minutes to see it in scuba")
    except Exception as e:
        print(f"Error logging result to Scuba table: {e}")
        raise
