#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import asyncio
from datetime import datetime

from executorch.sdk.edir.tests.exported_op_graph_test import (
    generate_op_graph,
    MultiOutputNodeModule,
)
from executorch.sdk.visualizer.generator import Generator


async def main() -> int:
    # Get the input, which is an example op_graph for testing
    model = MultiOutputNodeModule()
    op_graph = generate_op_graph(model, model.get_random_inputs())
    op_graph.attach_metadata(model.gen_inference_run())

    # Initialize the TB URLs generator
    generator = Generator()

    # Call gen() on the generator
    tb_url = await generator.gen(
        op_graph=op_graph,
        run_name=model._get_name() + "_" + datetime.now().strftime("%b%d_%H-%M-%S"),
    )

    # Process the returned URLs
    print(f"\nTo view the graph of this run, go to {tb_url}\n")

    return 0


if __name__ == "__main__":
    asyncio.run(main())
