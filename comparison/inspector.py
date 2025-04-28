# %%
from iree.compiler import ir
from collections import defaultdict
import os


os.chdir("/root/executorch/comparison")
def cvalue_to_int(cvalue):
    """Operand 이름을 int로 변환하거나 원래 이름 반환."""
    name = str(cvalue.get_name())
    if name.startswith("%zero"):
        return 0
    if name.startswith("%c"):
        try:
            return int(name[2:])
        except ValueError:
            return name
    return name

def parse_dispatch_statistics(prefix):
    """vm.call.variadic -> hal.command_buffer.dispatch 호출 통계를 수집."""
    train_or_test = os.path.basename(prefix)
    mlir_path = f"{prefix}_phases/{train_or_test}.12.vm.mlir"

    with ir.Context():
        with open(mlir_path, "r") as f:
            module = ir.Module.parse(f.read())

        # 모듈 첫 번째 operation의 첫 번째 블록
        top_block = list(module.body.operations)[0].body.blocks[0]

        # vm.func 찾기
        func_op = next((op for op in top_block.operations if op.operation.name == "vm.func"), None)
        if func_op is None:
            raise RuntimeError("No vm.func found in module.")

        stats = defaultdict(int)
        for op in func_op.body.blocks[0].operations:
            op_name = op.operation.name

            if op_name == "vm.call.variadic":
                target_attr = str(op.attributes[0].attr)
                if target_attr == "@hal.command_buffer.dispatch":
                    dispatch_id = cvalue_to_int(op.operands[2])
                    stats[dispatch_id] += 1

            elif op_name == "vm.call":
                target_attr = str(op.attributes[0].attr)
                if target_attr == "@hal.command_buffer.execution_barrier":
                    # 아직 별도 처리 없음 (pass)
                    pass

        return stats


