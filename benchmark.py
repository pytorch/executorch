import argparse
import os
import subprocess

qnn_sdk = os.getenv("QNN_SDK_ROOT")
htp_arch = "79"
workspace = "/data/local/tmp/et_ga_benchmark"
memory_script_file = "peak_memory.sh"
perf_file = "statistics.txt"


def get_artifacts(backend, pte_path):
    def get_build_dir(backend):
        build_dir = {
            "qnn": "build-android",
            "xnn": "build-xnnpack",
        }
        return build_dir[backend]

    memory_script = """$@ 2> /dev/null &

PROCESS=$1
PEAK_MEM=0
SAMPLES=0
TOTAL=0
while true; do
    PID=$(pidof $PROCESS)
    if [ "$PID" != "" ]; then
        DMA=$(dmabuf_dump $PID | grep "PROCESS TOTAL" | awk '{ print $3 }')
        PSS=$(dumpsys meminfo -s $PID | grep "TOTAL PSS" | awk '{ print $3 }')
        if [ "$PSS" == "" ]; then
            continue
        fi
        CURRENT=$(($DMA+$PSS))
        if [ CURRENT -gt PEAK_MEM ]; then
            PEAK_MEM=$CURRENT
        fi
        SAMPLES=$(($SAMPLES+1))
        TOTAL=$(($TOTAL+$CURRENT))
    else
        break
    fi
done

rm -rf memory_usage.txt
echo "peak_mem: $PEAK_MEM" >> statistics.txt
AVG_MEM=$(awk -- 'BEGIN{printf "%.3f", ARGV[1]/ARGV[2]}' "$TOTAL" "$SAMPLES")
echo "avg_mem: $AVG_MEM" >> statistics.txt
    """
    with open(memory_script_file, "w") as f:
        f.write(memory_script)

    runner = {
        "qnn": f"{get_build_dir(backend)}/examples/qualcomm/executor_runner/qnn_executor_runner",
        "xnn": f"{get_build_dir(backend)}/backends/xnnpack/xnn_executor_runner",
    }
    artifacts = {
        "qnn": [
            pte_path,
            f"{qnn_sdk}/lib/aarch64-android/libQnnHtp.so",
            (
                f"{qnn_sdk}/lib/hexagon-v{htp_arch}/"
                f"unsigned/libQnnHtpV{htp_arch}Skel.so"
            ),
            (f"{qnn_sdk}/lib/aarch64-android/" f"libQnnHtpV{htp_arch}Stub.so"),
            f"{qnn_sdk}/lib/aarch64-android/libQnnHtpPrepare.so",
            f"{qnn_sdk}/lib/aarch64-android/libQnnSystem.so",
            f"{get_build_dir(backend)}/backends/qualcomm/libqnn_executorch_backend.so",
            f"{qnn_sdk}/lib/aarch64-android/libQnnModelDlc.so",
            runner[backend],
            memory_script_file,
        ],
        "xnn": [
            pte_path,
            runner[backend],
            memory_script_file,
        ],
    }
    return artifacts[backend]


def get_cmds(backend, pte_path, iteration):
    cmd_args = {
        "qnn": (
            [
                f"--model_path {os.path.basename(pte_path)}",
                f"--iteration {iteration}",
                "--dump_statistics",
            ]
        ),
        "xnn": (
            [
                f"--model_path {os.path.basename(pte_path)}",
                f"--num_executions {iteration}",
                "--dump_statistics",
            ]
        ),
    }
    cmds_for_inference = {
        "qnn": (
            " ".join(
                [
                    f"cd {workspace} &&",
                    "chmod +x ./qnn_executor_runner &&",
                    f"./qnn_executor_runner {' '.join(cmd_args[backend])}",
                ]
            )
        ),
        "xnn": (
            " ".join(
                [
                    f"cd {workspace} &&",
                    "chmod +x ./xnn_executor_runner &&",
                    f"./xnn_executor_runner {' '.join(cmd_args[backend])}",
                ]
            )
        ),
    }
    # do not dump inference metrics during profiling memory
    for _, v in cmd_args.items():
        v.pop()
    cmds_for_memory = {
        "qnn": (
            " ".join(
                [
                    f"cd {workspace} &&",
                    "chmod +x ./qnn_executor_runner &&",
                    f"chmod +x {memory_script_file} &&",
                    f"./{memory_script_file} ./qnn_executor_runner {' '.join(cmd_args[backend])}",
                ]
            )
        ),
        "xnn": (
            " ".join(
                [
                    f"cd {workspace} &&",
                    "chmod +x ./xnn_executor_runner &&",
                    f"chmod +x {memory_script_file} &&",
                    f"./{memory_script_file} ./xnn_executor_runner {' '.join(cmd_args[backend])}",
                ]
            )
        ),
    }
    return [cmds_for_inference[backend], cmds_for_memory[backend]]


def start_benchmark(artifacts, cmds, device, host):
    def adb(action):
        if not host:
            actions = ["adb", "-s", device]
        else:
            actions = ["adb", "-H", host, "-s", device]
        actions.extend(action)
        subprocess.run(actions, stdout=subprocess.DEVNULL)

    def post_process():
        subprocess.run(["rm", "-rf", perf_file], stdout=subprocess.DEVNULL)
        for file_name in [perf_file]:
            adb(["pull", f"{workspace}/{file_name}", "."])
            with open(file_name, "r") as f:
                print(f.read())

    adb(["shell", "rm", "-rf", workspace])
    adb(["shell", "mkdir", "-p", workspace])
    for artifact in artifacts:
        adb(["push", artifact, workspace])
    for cmd in cmds:
        adb(["shell", cmd])
    post_process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--backend",
        help="either 'qnn' or 'xnn'",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--pte",
        help="path to generated .pte file",
        required=True,
    )
    parser.add_argument(
        "-H",
        "--host",
        help="hostname for adb gateway",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--device",
        help="serial number for adb device",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--iteration",
        help="total number of inferences",
        default=100,
        required=False,
    )
    args = parser.parse_args()
    start_benchmark(
        artifacts=get_artifacts(args.backend, args.pte),
        cmds=get_cmds(args.backend, args.pte, args.iteration),
        device=args.device,
        host=args.host,
    )
