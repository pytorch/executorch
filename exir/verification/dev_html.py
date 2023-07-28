# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from xml.etree import ElementTree as ET

from executorch.exir.verification.interpreter import Interpreter


def get_header(name):
    h2 = ET.Element("h2")
    span = ET.Element("span")
    span.text = name
    h2.append(span)

    return h2


def iterable_to_html(s):
    res = ET.Element("span")
    val_node = ET.SubElement(res, "p")

    for e in s:
        br = ET.SubElement(val_node, "br")
        br.text = str(e)
    return res


def gen_html(program):
    test = Interpreter(program)
    prog_opset = set(test.get_operators_list())

    prog_vals = test.get_constant_tensors()
    html = ET.Element("html")

    head = ET.Element("head")
    html.append(head)

    title = ET.Element("title")
    head.append(title)
    title.text = "Model"

    body = ET.Element("body")
    html.append(body)
    div = ET.Element(
        "div",
        attrib={
            "id": "main_content",
            "style": "position: absolute; width: 99%; height: 100%; overflow: scroll;",
        },
    )
    body.append(div)

    # Header: Operator List
    div.append(get_header("Operator List"))

    # Burn in data
    div.append(iterable_to_html(prog_opset))

    # Header: Constant Tensor List
    div.append(get_header("Constant Tensor List"))

    # Burn in data
    div.append(iterable_to_html(prog_vals))

    return ET.tostring(html)
