from __future__ import annotations

import argparse
import getpass
import logging
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

from tqdm import tqdm
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e 

from executorch.backends.qualcomm.export_utils import (
    QnnConfig,
    SimpleADB,
    get_backend_type,
    make_quantizer,
)

from executorch.examples.qualcomm.robotics.export.vla_flow_matching_exportable_module import (
    VLAFlowMatchingExportableModule,
)
from executorch.examples.qualcomm.robotics.policies.smolvla.config import (
    ACTION_EXPERT_GRAPH_NAME,
    PREFIX_MODEL_GRAPH_NAME,
    VISION_ENCODER_GRAPH_NAME,
    SmolVLAConfig,
)
from executorch.examples.qualcomm.robotics.policies.smolvla.policy import (
    SmolVLAPolicy,
)
from executorch.examples.qualcomm.robotics.policies.smolvla.prefix_decoder import (
    SmolVLAPrefixEmbedding,
)
from executorch.examples.qualcomm.robotics.policies.smolvla.smolvla_flow_matching import (
    SmolVLAFlowMatching,
)
from executorch.examples.qualcomm.robotics.remote_inference.server import (
    RemoteServer,
)
from executorch.examples.qualcomm.robotics.remote_inference.worker import (
    WORKER_COMPLETE,
    WorkerType,
    select_worker,
)
from executorch.examples.qualcomm.robotics.runtime.htp_io import (
    ACTION_CHUNK_FILENAME,
    RUNNER,
    parse_runner_timing,
    push_inputs,
    read_action_chunk,
    runner_command,
    write_raw_inputs,
)
from executorch.examples.qualcomm.robotics.scripts.smolvla.quant_recipe import (
    SmolVLAActionExpertQuantRecipe,
    SmolVLAPrefixQuantRecipe,
)

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5555


def report_rollout_timing(
    name: str, plan_ms: list[float], rollout_started_at: float
) -> None:
    """Print the end-to-end timing of one rollout served by Terminal A."""
    rollout_seconds = time.perf_counter() - rollout_started_at
    summary = f"[master] {name} timing plans={len(plan_ms)} rollout_s={rollout_seconds:.2f}"
    if plan_ms:
        summary += (
            f" mean_plan_e2e_ms={sum(plan_ms) / len(plan_ms):.2f}"
            f" min_plan_e2e_ms={min(plan_ms):.2f}"
            f" max_plan_e2e_ms={max(plan_ms):.2f}"
        )
    print(summary, flush=True)


def quantize(
    args: argparse.Namespace,
    module: VLAFlowMatchingExportableModule,
    prefix_embedding: SmolVLAPrefixEmbedding,
    config: SmolVLAConfig,
    *,
    soc_model: str,
) -> None:
    """Export, calibrate, and convert fixed Prefix and Action Expert modules."""

    # Step 1. Make quantizers
    from executorch.backends.qualcomm.quantizer.quantizer import (
        QnnQuantizer,
        QuantDtype,
    )
    prefix_decoder_quantizer = make_quantizer(
        quant_dtype=QuantDtype.use_16a8w,
        per_channel_linear=True,
        per_channel_conv=True,
        backend=get_backend_type("htp"), soc_model=soc_model
    )
    # prefix_decoder_quantizer.set_recipe(SmolVLAPrefixQuantRecipe(True).recipe)
    action_expert_quantizer = make_quantizer(
        quant_dtype=QuantDtype.use_16a8w,
        per_channel_linear=True,
        per_channel_conv=True,
        backend=get_backend_type("htp"), soc_model=soc_model
    )
    # action_expert_quantizer.set_recipe(SmolVLAActionExpertQuantRecipe(True).recipe)

    # Step 2. Export model
    logging.info("Exported model")
    module.export()
    
    # Step 3. Simulate exported module
    skip_rollout = False
    if not skip_rollout:
        logging.info("Simulate Exported module") 
        policy = SmolVLAPolicy(
            SmolVLAFlowMatching(
                module.exported_modules[VISION_ENCODER_GRAPH_NAME],
                module.exported_modules[PREFIX_MODEL_GRAPH_NAME],
                module.exported_modules[ACTION_EXPERT_GRAPH_NAME],
                num_steps=module.model.num_steps,
            ),
            prefix_embedding,
            config,
            seed=args.seed,
        )
        with RemoteServer(
            host=SERVER_HOST,
            port=SERVER_PORT,
        ) as server:
            server.accept()

            select_worker(
                server,
                worker=WorkerType.ROLLOUT,
                name="exported_module_rollout",
            )
            rollout_started_at = time.perf_counter()
            plan_ms: list[float] = []

            while True:
                try:
                    inputs = server.recv()
                except ConnectionError:
                    break
                
                
                # Check if worker complete
                if inputs == WORKER_COMPLETE:
                    server.send(True)
                    print(
                        f"[master] Rollout worker completed ",
                        flush=True,
                    )
                    break

                plan_started_at = time.perf_counter()
                action_chunk = policy.predict_action_chunk(inputs)
                plan_ms.append((time.perf_counter() - plan_started_at) * 1000)
                print(
                    f"[master] exported_module_rollout plan={len(plan_ms)} "
                    f"e2e_ms={plan_ms[-1]:.2f}",
                    flush=True,
                )

                server.send(action_chunk)
            report_rollout_timing(
                "exported_module_rollout", plan_ms, rollout_started_at
            )
        logging.info("Finish exported module simulation")

    # Step 4. Prepare for quantization
    prepared_prefix_model = prepare_pt2e(
        module.exported_modules[PREFIX_MODEL_GRAPH_NAME],
        prefix_decoder_quantizer,
    )
    prepared_action_expert = prepare_pt2e(
        module.exported_modules[ACTION_EXPERT_GRAPH_NAME],
        action_expert_quantizer,
    )

    # Step 5. quantize
    logging.info("Calibrating Prefix and Action Expert module")
    policy = SmolVLAPolicy(
        SmolVLAFlowMatching(
            module.exported_modules[VISION_ENCODER_GRAPH_NAME],
            prepared_prefix_model,
            prepared_action_expert,
            num_steps=module.model.num_steps,
        ),
        prefix_embedding,
        config,
        seed=args.seed,
    )
    # Due to the seperate enviroment the client will send the lerobot dataset
    # and server will recievd it to perferom calibration
    with RemoteServer(
        host=SERVER_HOST,
        port=SERVER_PORT,
        read_timeout_seconds=3000,
    ) as server:
        server.accept()
        
        select_worker(
            server,
            worker=WorkerType.CALIBRATION,
        )
        
        calibration_description = server.recv()
        observation_count = calibration_description[
            "observation_count"
        ]
        server.send(True)
        for observation_index in tqdm(range(
            observation_count
        )):
            inputs = server.recv()

            # Check if worker complete
            if inputs == WORKER_COMPLETE:
                print(
                    f"[master] Calibration worker completed ",
                    flush=True,
                )
                break
            
            policy.predict_action_chunk(inputs)

            server.send(True)
            

            print(
                f"[quantization] Calibrated "
                f"observation={observation_index + 1}/"
                f"{observation_count}",
                flush=True,
            )
        
        # Check calibration data complete
        completion = server.recv()
        if completion != WORKER_COMPLETE:
            raise RuntimeError(
                "Expected WORKER_COMPLETE after calibration, "
                f"got {completion!r}"
            )
        server.send(True)
        print(
            f"[master] Calibration worker completed "
            f"observations={observation_count}",
            flush=True,
        )

    # Step 6. Convert to quantized modules
    module.exported_modules[PREFIX_MODEL_GRAPH_NAME] = convert_pt2e(prepared_prefix_model)
    module.exported_modules[ACTION_EXPERT_GRAPH_NAME] = convert_pt2e(prepared_action_expert)

    # Step 7. Simultae quantized module
    logging.info("Simulate quantized module") 
    policy = SmolVLAPolicy(
        SmolVLAFlowMatching(
            module.exported_modules[VISION_ENCODER_GRAPH_NAME],
            module.exported_modules[PREFIX_MODEL_GRAPH_NAME],
            module.exported_modules[ACTION_EXPERT_GRAPH_NAME],
            num_steps=module.model.num_steps,
        ),
        prefix_embedding,
        config,
        seed=args.seed,
    )
    with RemoteServer(
        host=SERVER_HOST,
        port=SERVER_PORT,
    ) as server:
        server.accept()

        select_worker(
            server,
            worker=WorkerType.ROLLOUT,
            name="qdq_module_rollout",
        )
        rollout_started_at = time.perf_counter()
        plan_ms: list[float] = []

        while True:
            try:
                inputs = server.recv()
            except ConnectionError:
                break

            # Check if worker complete
            if inputs == WORKER_COMPLETE:
                server.send(True)
                print(
                    f"[master] Rollout worker completed ",
                    flush=True,
                )
                break

            plan_started_at = time.perf_counter()
            action_chunk = policy.predict_action_chunk(inputs)
            plan_ms.append((time.perf_counter() - plan_started_at) * 1000)
            print(
                f"[master] qdq_module_rollout plan={len(plan_ms)} "
                f"e2e_ms={plan_ms[-1]:.2f}",
                flush=True,
            )

            server.send(action_chunk)
        report_rollout_timing("qdq_module_rollout", plan_ms, rollout_started_at)

    logging.info("Finish quantized module simulation") 

def compile(
    module: VLAFlowMatchingExportableModule,
    config: SmolVLAConfig,
    *,
    artifact_dir: Path,
    soc_model: str,
    profile_level: int,
    online_prepare: bool,
) -> dict[str, Path]:
    """Lower Vision, Prefix, and Action Expert and write their PTE artifacts."""
    from executorch.backends.qualcomm._passes.build_quant_io import BuildQuantIo
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        get_soc_to_chipset_map,
        to_edge_transform_and_lower_to_qnn,
    )
    from executorch.exir import ExecutorchBackendConfig

    # Compile quatnized modules to PTE artifacts
    chipset = get_soc_to_chipset_map()[soc_model]
    quantized_specs = generate_qnn_executorch_compiler_spec(
        soc_model=chipset,
        backend_options=generate_htp_compiler_spec(use_fp16=False),
        profile_level=profile_level,
        online_prepare=online_prepare,
    )
    compile_specs = generate_qnn_executorch_compiler_spec(
        soc_model=chipset,
        backend_options=generate_htp_compiler_spec(use_fp16=True),
        profile_level=profile_level,
        online_prepare=online_prepare,
    )
    if (
        module.exported_modules[VISION_ENCODER_GRAPH_NAME] is None
        or module.exported_modules[PREFIX_MODEL_GRAPH_NAME] is None
        or module.exported_modules[ACTION_EXPERT_GRAPH_NAME] is None
    ):
        raise RuntimeError("Quantized modules were not created")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    vision_edge_program = to_edge_transform_and_lower_to_qnn(
        module.exported_modules[VISION_ENCODER_GRAPH_NAME],
        module.model.vision_encoder.get_example_inputs(),
        compile_specs,
    )
    prefix_edge_program = to_edge_transform_and_lower_to_qnn(
        module.exported_modules[PREFIX_MODEL_GRAPH_NAME],
        module.model.prefix_model.get_example_inputs(),
        quantized_specs
    )
    action_expert_edge_program = to_edge_transform_and_lower_to_qnn(
        module.exported_modules[ACTION_EXPERT_GRAPH_NAME],
        module.model.action_expert.get_example_inputs(),
        quantized_specs
    )

    vision_path = artifact_dir / config.pte_filenames[VISION_ENCODER_GRAPH_NAME]
    prefix_path = artifact_dir / config.pte_filenames[PREFIX_MODEL_GRAPH_NAME]
    action_expert_path = artifact_dir / config.pte_filenames[ACTION_EXPERT_GRAPH_NAME]
    vision_program = vision_edge_program.to_executorch(
        ExecutorchBackendConfig(passes=[BuildQuantIo()])
    )
    prefix_program = prefix_edge_program.to_executorch(
        ExecutorchBackendConfig(passes=[BuildQuantIo()])
    )
    action_expert_program = action_expert_edge_program.to_executorch(
        ExecutorchBackendConfig(passes=[BuildQuantIo()])
    )
    with vision_path.open("wb") as file:
        vision_program.write_to_file(file)
    with prefix_path.open("wb") as file:
        prefix_program.write_to_file(file)
    with action_expert_path.open("wb") as file:
        action_expert_program.write_to_file(file)
    return {
        VISION_ENCODER_GRAPH_NAME: vision_path,
        PREFIX_MODEL_GRAPH_NAME: prefix_path,
        ACTION_EXPERT_GRAPH_NAME: action_expert_path,
    }


def inference(
    args: argparse.Namespace,
    prefix_embedding: SmolVLAPrefixEmbedding,
    config: SmolVLAConfig,
    pte_paths: dict[str, Path],
) -> None:
    """Serve a rollout whose action chunks are produced on HTP, one adb execute per plan."""
    qnn_config = QnnConfig(
        soc_model=args.soc_model,
        build_folder=args.build_folder,
        host=args.adb_host,
        device=args.device,
    )
    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/smolvla"
    adb = SimpleADB(
        qnn_config=qnn_config,
        pte_path=[str(path) for path in pte_paths.values()],
        workspace=workspace,
        runner=RUNNER,
    )
    cmd = runner_command(
        workspace,
        vision_pte=pte_paths[VISION_ENCODER_GRAPH_NAME].name,
        prefix_pte=pte_paths[PREFIX_MODEL_GRAPH_NAME].name,
        action_pte=pte_paths[ACTION_EXPERT_GRAPH_NAME].name,
        num_steps=config.num_steps,
    )
    # The policy only prepares inputs here; the model runs on device.
    policy = SmolVLAPolicy(None, prefix_embedding, config, seed=args.seed)
    action_chunk_shape = (1, config.chunk_size, config.max_action_dim)
    plan_timings_ms: list[dict[str, float]] = []
    runner_stdout: list[str] = []

    def capture_runner_output(result) -> None:
        print(result.stdout, end="", flush=True)
        runner_stdout.append(result.stdout)

    # Step 1. Push QNN libraries, PTEs, and the runner once
    adb.push()

    # Step 2. Serve the rollout, one adb execute per plan
    with tempfile.TemporaryDirectory() as tmp_dir, RemoteServer(
        host=SERVER_HOST,
        port=SERVER_PORT,
    ) as server:
        input_dir = Path(tmp_dir) / "inputs"
        host_action_chunk = Path(tmp_dir) / ACTION_CHUNK_FILENAME
        server.accept()

        select_worker(
            server,
            worker=WorkerType.ROLLOUT,
            name="htp_module_rollout",
        )
        rollout_started_at = time.perf_counter()

        while True:
            try:
                inputs = server.recv()
            except ConnectionError:
                break

            # Check if worker complete
            if inputs == WORKER_COMPLETE:
                server.send(True)
                print(
                    f"[master] Rollout worker completed ",
                    flush=True,
                )
                break

            started_at = time.perf_counter()
            push_inputs(
                qnn_config,
                write_raw_inputs(policy.prepare_inputs(inputs), input_dir),
                workspace,
            )
            pushed_at = time.perf_counter()
            runner_stdout.clear()
            adb.execute(
                custom_runner_cmd=cmd, output_callback=capture_runner_output
            )
            executed_at = time.perf_counter()
            adb.pull(
                host_output_path=str(host_action_chunk),
                device_output_path=f"{workspace}/{ACTION_CHUNK_FILENAME}",
            )
            action_chunk = read_action_chunk(host_action_chunk, action_chunk_shape)
            finished_at = time.perf_counter()

            device_plan_ms, device_execute_ms = parse_runner_timing(
                "".join(runner_stdout)
            )
            timing_ms = {
                "e2e": (finished_at - started_at) * 1000,
                "push": (pushed_at - started_at) * 1000,
                "adb_execute": (executed_at - pushed_at) * 1000,
                "pull": (finished_at - executed_at) * 1000,
                "device_plan": device_plan_ms,
                "device_execute": device_execute_ms,
            }
            plan_timings_ms.append(timing_ms)
            print(
                f"[master] htp_module_rollout plan={len(plan_timings_ms)} "
                + " ".join(f"{name}_ms={value:.2f}" for name, value in timing_ms.items()),
                flush=True,
            )

            server.send(action_chunk[:, :, : config.action_feature[0]])
        report_rollout_timing(
            "htp_module_rollout",
            [timing["e2e"] for timing in plan_timings_ms],
            rollout_started_at,
        )
    if plan_timings_ms:
        print(
            f"[master] htp_module_rollout breakdown plans={len(plan_timings_ms)} "
            + " ".join(
                f"mean_{name}_ms="
                f"{sum(timing[name] for timing in plan_timings_ms) / len(plan_timings_ms):.2f}"
                for name in plan_timings_ms[0]
            ),
            flush=True,
        )
    logging.info("Finish HTP module rollout")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smolvla")
    parser.add_argument("--soc-model", default="SM8850")
    parser.add_argument("--artifact", type=Path, default=Path("smolvla_qnn"))
    parser.add_argument("--compile_only", action="store_true")
    parser.add_argument("--pre_gen_pte", type=Path)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--profile-level", choices=(0, 2, 3), type=int, default=0)
    parser.add_argument("--online-prepare", action="store_true")
    parser.add_argument("--server-host", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=5555)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adb-host")
    parser.add_argument("--device")
    parser.add_argument("--build-folder", default="build-android")
    parser.add_argument("--device-output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.compile_only and args.pre_gen_pte:
        raise RuntimeError("Cannot set both compile_only and pre_gen_pte")
    if args.compile_only and args.verbose:
        raise RuntimeError("Cannot set both compile_only and verbose")

    if not args.compile_only and args.device is None:
        raise RuntimeError("--device is required unless --compile_only is set")

    config = SmolVLAConfig.from_hf_policy("lerobot/smolvla_libero")
    policy_state_dict = config.load_policy_state_dict()
    prefix_embedding = SmolVLAPrefixEmbedding.from_hf_policy(
        config, policy_state_dict
    ).eval()
    if args.pre_gen_pte:
        inference(
            args,
            prefix_embedding,
            config,
            {
                name: args.pre_gen_pte / filename
                for name, filename in config.pte_filenames.items()
            },
        )
        return

    # Load Policy
    policy = VLAFlowMatchingExportableModule(
        SmolVLAFlowMatching.from_hf_policy(config, policy_state_dict)
    )

    # Quantize
    quantize(args, policy, prefix_embedding, config, soc_model=args.soc_model)

    # Compile
    pte_paths = compile(
        policy,
        config,
        artifact_dir=args.artifact,
        soc_model=args.soc_model,
        profile_level=args.profile_level,
        online_prepare=args.online_prepare,
    )
    
    if args.compile_only:
        return

    # Inference
    inference(args, prefix_embedding, config, pte_paths)


if __name__ == "__main__":
    main()
