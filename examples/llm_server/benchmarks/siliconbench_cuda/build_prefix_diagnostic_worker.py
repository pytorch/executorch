"""CPU-only observation build: trace prefix candidates and synchronous waits."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import subprocess
import traceback
from profile_mini_HTTP_control_phases import instrument


def header_trace(source):
    source=source.replace('#include <algorithm>','#include <algorithm>\n#include <chrono>\n#include <cstdio>',1)
    anchor='namespace batching {\n'
    helper=r'''
inline std::int64_t prefix_diag_tick() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline void prefix_diag_event(const char* phase, std::size_t demand,
    std::size_t matched, std::size_t entries, std::int64_t elapsed = 0) {
  std::fprintf(stderr, "ET_PREFIX_DIAGNOSTIC %s %zu %zu %zu %lld %lld\n",
      phase, demand, matched, entries, static_cast<long long>(elapsed),
      static_cast<long long>(prefix_diag_tick()));
}
'''
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+helper,1)
    before='        auto snapshot = state->snapshot.get();'
    assert source.count(before)==1
    source=source.replace(before,'        const auto wait_begin = prefix_diag_tick();\n'+before+
        '\n        prefix_diag_event("capture_wait", state->tokens.size(), state->captured_tokens, 0, prefix_diag_tick()-wait_begin);',1)
    before='    if (request.size() < 2 || entries_.empty()) {'
    assert source.count(before)==1
    source=source.replace(before,before+'\n      prefix_diag_event("lookup_empty", request.size(), 0, entries_.size());',1)
    before='      for (const auto& candidate : candidates) {'
    assert source.count(before)==1
    source=source.replace(before,'      prefix_diag_event("lookup_theoretical_retained", request.size(), candidates.empty() ? 0 : candidates.front().matched, entries_.size());\n'+before+'\n        const auto clone_begin = prefix_diag_tick();',1)
    before='                .get();\n        if (session) {'
    assert source.count(before)==1
    source=source.replace(before,'                .get();\n        prefix_diag_event(session ? "lookup_clone_ok" : "lookup_clone_failed", request.size(), candidate.matched, candidate.entry->tokens.size(), prefix_diag_tick()-clone_begin);\n        if (session) {',1)
    before='      entries_.push_front(\n          Entry{tokens, std::move(snapshot), std::move(metadata)});'
    assert source.count(before)==1
    source=source.replace(before,'      prefix_diag_event(victim != entries_.end() ? "insert_evict" : "insert", tokens.size(), 0, entries_.size());\n'+before,1)
    before='      state.snapshot = state.request_clone(position);'
    assert source.count(before)==1
    source=source.replace(before,'      const auto clone_begin = prefix_diag_tick();\n'+before+'\n      prefix_diag_event("capture_clone_submit", state.tokens.size(), position, 0, prefix_diag_tick()-clone_begin);',1)
    return source


def main(args):
    rc=1
    try:
        old=args.base/'four-opt-Gemma-combined-full-native-worker-20261028-v2'
        current=args.base/'four-opt-Gemma-observable-full-native-worker-20261028-v5'
        compile=json.loads((old/'build-commands.json').read_text())[1]
        link=json.loads((current/'build-commands.json').read_text())[-1]
        original=Path(compile[compile.index('-c')+1]).read_text()
        headers=[Path(flag[2:])/'executorch/extension/llm/batching/prefix_cache.h'
                 for flag in compile if flag.startswith('-I')]
        header=next(path for path in headers if path.is_file())
        source=instrument(original)
        marker='#include <executorch/extension/llm/batching/prefix_cache.h>'
        assert source.count(marker)==1
        source=source.replace(marker,'#include "'+str(args.job/'prefix_cache.h')+'"',1)
        before='      return prefix_cache_.lookup_identity(*request->text->prompt);'
        assert source.count(before)==1
        source=source.replace(before,'      const auto lookup_begin = batching::MetricsClock::now();\n      auto match = prefix_cache_.lookup_identity(*request->text->prompt);\n      trace_phase(request, "prefix_lookup_inclusive", lookup_begin);\n      return match;',1)
        before='        request->text->prefix_capture->collect();'
        assert source.count(before)==1
        source=source.replace(before,'        const auto capture_begin = batching::MetricsClock::now();\n'+before+'\n        trace_phase(request, "prefix_capture_collect_inclusive", capture_begin);',1)
        before='                return !request->cancelled.load() &&\n                    !capture_lane_busy_.exchange(true);'
        assert source.count(before)==1
        source=source.replace(before,'                if (request->cancelled.load()) return false;\n                const bool acquired = !capture_lane_busy_.exchange(true);\n                if (!acquired) std::fprintf(stderr, "ET_PREFIX_CAPTURE_BUSY %llu\\n", static_cast<unsigned long long>(request->id));\n                return acquired;',1)
        (args.job/'original-serving_runtime.cpp').open('x').write(original)
        (args.job/'original-prefix_cache.h').open('x').write(header.read_text())
        (args.job/'serving_runtime.cpp').open('x').write(source)
        (args.job/'prefix_cache.h').open('x').write(header_trace(header.read_text()))
        compile[compile.index('-c')+1]=str(args.job/'serving_runtime.cpp')
        compile[compile.index('-o')+1]=str(args.job/'serving_runtime.o')
        assert '-DNDEBUG' in compile
        old_object=str(old/'1-serving_runtime.o');assert link.count(old_object)==1
        link[link.index(old_object)]=str(args.job/'serving_runtime.o')
        link[link.index('-o')+1]=str(args.job/'worker')
        (args.job/'build-commands.json').open('x').write(json.dumps([compile,link],indent=2))
        with (args.job/'build.log').open('x') as log:
            for command in [compile,link]:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        (args.job/'build-receipt.json').open('x').write(json.dumps(dict(worker=str(args.job/'worker'),
            worker_SHA256=hashlib.sha256((args.job/'worker').read_bytes()).hexdigest(),observation_only=True,
            same_route_control=True,kernels_and_all_other_objects_unchanged=True,
            original_prefix_header=str(header),original_header_SHA256=hashlib.sha256(header.read_bytes()).hexdigest()),indent=2))
        rc=0
    except BaseException:(args.job/'build-failure.txt').open('x').write(traceback.format_exc())
    finally:(args.job/'build-exit-receipt.json').open('x').write(json.dumps(dict(returncode=rc,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())))
    return rc


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--job',type=Path,required=True)
    raise SystemExit(main(parser.parse_args()))
