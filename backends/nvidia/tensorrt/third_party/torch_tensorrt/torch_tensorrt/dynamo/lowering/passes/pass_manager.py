import os
from typing import Any, Callable, List, Optional

import torch
from torch.fx import passes
from torch.fx.passes.pass_manager import PassManager
from torch_tensorrt.dynamo._defaults import DEBUG_LOGGING_DIR
from torch_tensorrt.dynamo._settings import CompilationSettings


def _generate_draw_fx_graph_pass(
    output_path_prefix: str, name: str
) -> Callable[[torch.fx.GraphModule, CompilationSettings], torch.fx.GraphModule]:
    def draw_fx_graph_pass(
        gm: torch.fx.GraphModule, settings: CompilationSettings
    ) -> torch.fx.GraphModule:
        if not os.path.exists(f"{output_path_prefix}/"):
            os.makedirs(f"{output_path_prefix}/")
        path = f"{output_path_prefix}/{name}.svg"
        g = passes.graph_drawer.FxGraphDrawer(gm, name)
        with open(path, "wb") as f:
            f.write(g.get_dot_graph().create_svg())
        return gm

    return draw_fx_graph_pass


class DynamoPassManager(PassManager):  # type: ignore[misc]
    def __init__(
        self,
        passes: Optional[
            List[
                Callable[
                    [torch.fx.GraphModule, CompilationSettings], torch.fx.GraphModule
                ]
            ]
        ] = None,
        constraints: Optional[List[Callable]] = None,
    ):
        super().__init__(passes, constraints)

    @classmethod
    def build_from_passlist(
        cls,
        passes: Optional[
            List[
                Callable[
                    [torch.fx.GraphModule, CompilationSettings], torch.fx.GraphModule
                ]
            ]
        ],
    ) -> Any:
        pm = DynamoPassManager(passes)
        return pm

    def add_pass_with_index(
        self,
        lowering_pass: Callable[
            [torch.fx.GraphModule, CompilationSettings], torch.fx.GraphModule
        ],
        index: Optional[int] = None,
    ) -> None:
        if index is None:
            self.passes.append(lowering_pass)
            index = -1
        else:
            self.passes.insert(index, lowering_pass)

    def remove_pass_with_index(self, index: int) -> None:
        del self.passes[index]

    def insert_debug_pass_before(
        self, passes: List[str], output_path_prefix: str = DEBUG_LOGGING_DIR
    ) -> None:
        """Insert debug passes in the PassManager pass sequence prior to the execution of a particular pass.

        Args:
            passes: List of pass names to insert debug passes before
            output_path_prefix: Prefix to use for generated debug files

        Debug passes generate SVG visualizations of the FX graph at specified points
        in the pass sequence.
        """
        self.check_pass_names_valid(passes)
        new_pass_list = []
        for ps in self.passes:
            if ps.__name__ in passes:
                new_pass_list.append(
                    _generate_draw_fx_graph_pass(
                        output_path_prefix, f"before_{ps.__name__}"
                    )
                )
            new_pass_list.append(ps)

        self.passes = new_pass_list
        self._validated = False

    def insert_debug_pass_after(
        self, passes: List[str], output_path_prefix: str = DEBUG_LOGGING_DIR
    ) -> None:
        """Insert debug passes in the PassManager pass sequence after the execution of a particular pass.

        Args:
            passes: List of pass names to insert debug passes after
            output_path_prefix: Prefix to use for generated debug files

        Debug passes generate SVG visualizations of the FX graph at specified points
        in the pass sequence.
        """
        self.check_pass_names_valid(passes)
        new_pass_list = []
        for ps in self.passes:
            new_pass_list.append(ps)
            if ps.__name__ in passes:
                new_pass_list.append(
                    _generate_draw_fx_graph_pass(
                        output_path_prefix, f"after_{ps.__name__}"
                    )
                )

        self.passes = new_pass_list
        self._validated = False

    def check_pass_names_valid(self, debug_pass_names: List[str]) -> None:
        pass_names_str = [p.__name__ for p in self.passes]
        for name in debug_pass_names:
            assert (
                name in pass_names_str
            ), f"{name} is not a valid pass! Passes: {pass_names_str}"

    def __call__(self, gm: Any, settings: CompilationSettings) -> Any:
        self.validate()
        out = gm
        for _pass in self.passes:
            out = _pass(out, settings)
        return out

    def __str__(self) -> str:
        return str(self.passes)
