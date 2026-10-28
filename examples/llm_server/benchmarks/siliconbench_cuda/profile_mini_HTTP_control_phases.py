"""Observe real HTTP admission phases using a saved mini and Release worker."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import traceback


def instrument(source):
    source = source.replace('#include <cmath>', '#include <cmath>\n#include <chrono>\n#include <cstdio>', 1)
    marker = '  LifecycleResult prepare_text(const Request& request) {'
    helper = '''  static void trace_phase(const Request& request, const char* phase,
      batching::MetricsTime begin) {
    const auto end = batching::MetricsClock::now();
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    const auto age = request->text ? std::chrono::duration_cast<std::chrono::microseconds>(end - request->text->submitted).count() : 0;
    std::fprintf(stderr, "ET_HTTP_CONTROL_PHASE %llu %s %lld %lld\\n",
        static_cast<unsigned long long>(request->id), phase,
        static_cast<long long>(us), static_cast<long long>(age));
  }

'''
    assert source.count(marker) == 1
    source = source.replace(marker, helper + marker)
    before = '    auto prepared = detail::prepare_prompt('
    assert source.count(before) == 1
    source = source.replace(before, '    const auto prepare_begin = batching::MetricsClock::now();\n' + before)
    after = '        model_preparer_);\n    if (std::holds_alternative<std::monostate>(prepared)) {'
    assert source.count(after) == 1
    source = source.replace(after, '        model_preparer_);\n    trace_phase(request, "prompt_prepare", prepare_begin);\n    if (std::holds_alternative<std::monostate>(prepared)) {')
    before = '    const auto acceptance = runner_->accepts_async(view).get();'
    assert source.count(before) == 1
    source = source.replace(before, '    const auto accept_begin = batching::MetricsClock::now();\n' + before + '\n    trace_phase(request, "accept_wait", accept_begin);')
    before = '          opened = runner_->open_session_async().get();'
    assert source.count(before) == 1
    source = source.replace(before, '          const auto open_begin = batching::MetricsClock::now();\n' + before + '\n          trace_phase(request, "open_wait", open_begin);')
    before = '  void generate(const Request& request) {'
    assert source.count(before) == 1
    source = source.replace(before, before + '\n    trace_phase(request, "control_queue", request->text ? request->text->submitted : batching::MetricsClock::now());')
    before = '      request->bind(std::move(handle));'
    assert source.count(before) == 1
    source = source.replace(before, before + '\n      trace_phase(request, "start_submitted", batching::MetricsClock::now());')
    return source


def main(args):
    job, base = args.job, args.base
    rc = 1
    try:
        current = base/'four-opt-MG-observable-full-native-worker-20261028-v5'
        commands = json.loads((current/'build-commands.json').read_text())
        previous = json.loads((base/'four-opt-MG-combined-full-native-worker-20261028-v2/build-commands.json').read_text())
        source = Path(previous[1][previous[1].index('-c')+1]).read_text()
        (job/'original-serving_runtime.cpp').write_text(source)
        (job/'profile-serving_runtime.cpp').write_text(instrument(source))
        compile_command = previous[1][:]
        compile_command[compile_command.index('-c')+1] = str(job/'profile-serving_runtime.cpp')
        compile_command[compile_command.index('-o')+1] = str(job/'serving_runtime.o')
        link = commands[-1][:]
        old = str(base/'four-opt-MG-combined-full-native-worker-20261028-v2/1-serving_runtime.o')
        assert link.count(old) == 1
        link[link.index(old)] = str(job/'serving_runtime.o')
        link[link.index('-o')+1] = str(job/'worker')
        (job/'build-commands.json').write_text(json.dumps([compile_command, link], indent=2))
        with (job/'build.log').open('x') as log:
            for command in [compile_command, link]:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        (job/'build-receipt.json').write_text(json.dumps(dict(
            worker=str(job/'worker'), SHA256=hashlib.sha256((job/'worker').read_bytes()).hexdigest(),
            phase_observation_only=True, dispatch_unchanged=True), indent=2))
        # The original saved harness remains immutable. This isolated copy only binds our worker.
        harness = (job/'profile_mini_HTTP_routing_prefix_off.py').read_text()
        old = "str(base/'four-opt-MG-observable-full-native-worker-20261028-v5/worker')"
        assert harness.count(old) == 1
        harness = harness.replace(old, "str(job/'worker')")
        (job/'HTTP-phase-harness.py').write_text(harness)
        command = [str(base/'venv/bin/python'), '-B', '-u', str(job/'HTTP-phase-harness.py'),
                   '--base', str(base), '--job', str(job), '--port', str(args.port)]
        with (job/'HTTP-controller.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert json.loads((job/'exit-receipt.json').read_text())['returncode'] == 0
        rc = 0
    except BaseException:
        (job/'phase-failure.txt').write_text(traceback.format_exc())
    finally:
        (job/'phase-exit-receipt.json').write_text(json.dumps(dict(returncode=rc,
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), mini_only=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--port', type=int, default=20271)
    main(parser.parse_args())
