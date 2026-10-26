"""Preserve literal cond operands during export placeholder naming.

The pinned PyTorch helper assumes every higher-order operand is an FX node.
Full MG export also lifts integer closure arguments into cond operands. Keep
their existing subgraph placeholder names; graph arguments, predicates and
operators are unchanged. The replacement is scoped to this export process.
"""
import hashlib
import inspect
import json
import operator
from pathlib import Path

import torch
import torch._export.utils as utils


ORIGINAL_SHA256 = '9ef3327995d9e0aa6960af5d7e631aaa8358cbd6019db98bee41c99ab5c43517'


def _fixture(literal):
    root = torch.nn.Module()
    for name in ['true_branch', 'false_branch']:
        graph = torch.fx.Graph()
        x = graph.placeholder('local_x')
        y = graph.placeholder('local_y')
        graph.output(graph.call_function(operator.add, (x, y)))
        root.add_module(name, torch.fx.GraphModule(torch.nn.Module(), graph))
    graph = torch.fx.Graph()
    x = graph.placeholder('parent_x')
    y = 16 if literal else graph.placeholder('parent_y')
    t = graph.get_attr('true_branch')
    f = graph.get_attr('false_branch')
    graph.output(graph.call_function(torch.ops.higher_order.cond, (True, t, f, (x, y))))
    return torch.fx.GraphModule(root, graph)


def install(job_dir):
    original = utils._name_hoo_subgraph_placeholders
    source = inspect.getsource(original)
    assert hashlib.sha256(source.encode()).hexdigest() == ORIGINAL_SHA256
    assert source.count('hoo_phs[i].name') == 2
    events = Path(job_dir) / 'literal-cond-placeholder-naming.jsonl'
    modified = source.replace('hoo_phs[i].name', '_operand_placeholder_name(hoo_phs[i], node)')

    def operand_name(operand, placeholder):
        assert placeholder.op == 'placeholder'
        if isinstance(operand, torch.fx.Node):
            return operand.name
        assert type(operand) is int, 'An unverified non-node operand needs separate handling'
        with events.open('a') as stream:
            stream.write(json.dumps(dict(type='int', value=operand,
                retained_placeholder=placeholder.name)) + '\n')
        return placeholder.name

    namespace = dict(original.__globals__)
    namespace['_operand_placeholder_name'] = operand_name
    exec(compile(modified, '<roofline-cond-placeholder-naming>', 'exec'), namespace)
    replacement = namespace['_name_hoo_subgraph_placeholders']
    # Confirm ordinary FX operands retain exactly the pinned naming behavior.
    normal_before, normal_after = _fixture(False), _fixture(False)
    original(normal_before)
    replacement(normal_after)
    for name in ['true_branch', 'false_branch']:
        assert getattr(normal_before, name).code == getattr(normal_after, name).code
    # Reproduce the original failure, then check scalar branch execution and
    # graph argument identity after the narrow naming correction.
    broken = _fixture(True)
    try:
        original(broken)
    except AttributeError as exc:
        assert "'int' object has no attribute 'name'" in str(exc)
    else:
        raise AssertionError('The pinned literal-operand failure was not reproduced')
    fixed = _fixture(True)
    graph_arguments = [node.args for node in fixed.graph.nodes]
    x = torch.arange(3, dtype=torch.float32)
    expected = {name: getattr(fixed, name)(x, 16) for name in ['true_branch', 'false_branch']}
    replacement(fixed)
    fixed.graph.lint()
    assert [node.args for node in fixed.graph.nodes] == graph_arguments
    for name in ['true_branch', 'false_branch']:
        branch = getattr(fixed, name)
        branch.graph.lint()
        assert torch.equal(branch(x, 16), expected[name])
        assert [node.target for node in branch.graph.nodes if node.op == 'placeholder'] == ['parent_x', 'local_y']
    events.unlink()  # Synthetic checks are distinct from actual export events.
    utils._name_hoo_subgraph_placeholders = replacement
    return dict(original_helper_SHA256=ORIGINAL_SHA256,
        modified_helper_SHA256=hashlib.sha256(modified.encode()).hexdigest(),
        process_local=True, installed_framework_files_changed=False,
        original_failure_reproduced=True, ordinary_FX_naming_identical=True,
        literal_branch_outputs_equal=True, cond_operands_and_predicates_unchanged=True,
        actual_literal_events=str(events))
