import os, re
from models import model_dict
import glob
import subprocess
import json
from inspector import parse_dispatch_statistics

def make_input_args(fname):
    with open(fname, "r") as f:
        next(f)  # 첫 번째 줄 건너뛰기
        definition = next(f).split("->")[0].split("!")[1:]

    print(len(definition))

    def process_definition(d):
        d = re.sub(r"(^.*<)|(>.*)", "", d)
        shape, _type = d.split("]")
        shape = re.sub(",", "x", shape)[1:] if shape else ""
        _type = _type[1:]
        _type = {"si64": "i64", "si32": "i32"}.get(_type, _type)

        return f"    --input={shape}x{_type}=0.5 \\" if shape else f"    --input={_type}=0.5 \\"

    return "\n".join(process_definition(d) for d in definition)

def run_executorch(model_name, train=False, xnn=True, num_executions=100, multithreaded=True, profiling=False):
    prefix = f"./{model_name}/{'train' if train else 'inf'}"
    model_path = f"{prefix}.pte"
    threads = 48 if multithreaded else 1
    app = "/root/executorch/cmake-out/executor_runner"
    cmd = f"{app} --model_path={model_path} \
        --num_executions={num_executions} \
        --cpu_threads={threads}"
    if profiling:
        cmd += f" --etdump_path={prefix}_{threads}.etdp"
    print(cmd)
    os.system(cmd)
    if profiling:
        from executorch.devtools import Inspector
        inspector = Inspector(etdump_path=f"{prefix}_{threads}.etdp")
        inspector.save_data_to_tsv(f"{prefix}_{threads}.tsv", include_delegate_debug_data = True)

def run_iree(model_name, train=False, num_executions=100, multithreaded=True):
    prefix = f"./{model_name}/{'train' if train else 'inf'}"
    app = "/root/iree-latest-build/install/bin/iree-benchmark-module"
    cmd = f"{app} \
        --module={prefix}.vmfb \
        --device={'local-task' if multithreaded else 'local-sync'} \
        --function=main \
        --benchmark_repetitions={num_executions} \
        {make_input_args(prefix + '.mlir')}"
    print(cmd)
    os.system(cmd)

def run_iree_prof(model_name, train=False, num_executions=100, multithreaded=True):
    prefix = f"./{model_name}/{'train' if train else 'inf'}"

    files = list(glob.glob(f"{prefix}_dispatch/*.mlir"))
    def get_idx(f):
        return int(f.split("async_dispatch_")[1].split("_")[0])
    files = [(get_idx(f), f) for f in files]
    files.sort(key=lambda x: x[0])

    stats = parse_dispatch_statistics(prefix)
    real_time_sum = 0
    cpu_time_sum = 0
    for j, (i, f) in enumerate(files):
        try:
            result = subprocess.run(
                f"/root/iree-latest-build/install/bin/iree-compile '{f}'"
                " | /root/iree-latest-build/install/bin/iree-benchmark-module "
                "--benchmark_format=json --benchmark_repetitions=1 --module=-"
                f" --device={'local-task' if multithreaded else 'local-sync'}",
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )
            bench = json.loads(result.stdout)
            for bench_item in bench["benchmarks"]:
                name = bench_item["name"].split("async_dispatch_")[2].split("/process_time")[0]
                real_time = bench_item["real_time"]
                cpu_time = bench_item["cpu_time"]
                # time_unit = bench_item["time_unit"]
                rep = stats[j] if j in stats else 0
                print(f"{i:<5}\t{name:<40}\t{real_time:>15.2f}\t{cpu_time:>15.2f}\t{rep}\t{real_time * rep:>15.2f}\t{cpu_time * rep:>15.2f}")
                real_time_sum += real_time * rep
                cpu_time_sum += cpu_time * rep
        except subprocess.CalledProcessError as e:
            print("ERR")
    print(f"real_time_sum: {real_time_sum}, cpu_time_sum: {cpu_time_sum}")

# for train in [True, False]:
#     run_iree("vit_b_16", train=train, num_executions=10, multithreaded=True)
#     run_iree("vit_tiny_patch16_224", train=train, num_executions=10, multithreaded=True)
#     run_iree("mobilevit_s", train=train, num_executions=10, multithreaded=True)
# print()

# run_executorch("vit_b_16", train=False, xnn=True, num_executions=100, multithreaded=False)
# run_executorch("vit_tiny_patch16_224", train=False, xnn=True, num_executions=100, multithreaded=False)
# run_executorch("mobilevit_s", train=False, xnn=True, num_executions=100, multithreaded=False)

# next: single-threaded
# run_iree("vit_tiny_patch16_224", train=False,  multithreaded=False)
# run_executorch("vit_tiny_patch16_224", train=False,  multithreaded=True, profiling=True, num_executions=1)
run_iree("resnet18", train=False, multithreaded=True, num_executions=10)