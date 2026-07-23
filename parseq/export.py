
from executorch.parseq.export_utils import get_dummy_input, prepare_export_model

from executorch.examples.qualcomm.utils import build_executorch_binary


if __name__ == '__main__':
    
    model_name = "parseq"
    export_mode = 'executorch'
    model = prepare_export_model(model_name, export_mode)
    image = get_dummy_input()
    inputs = (image,)

    build_executorch_binary(
        model,
        inputs,
        "SM8650",
        "parseq_qualcomm.pte",
        [inputs],
        skip_node_op_set={"aten.full.default", "aten.where.self"},
        skip_node_id_set={"aten_view_copy_default_235"},
    )
