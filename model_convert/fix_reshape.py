#!/usr/bin/env python3
"""Fold dynamic Reshape/Expand shape inputs into constants.

torch.onnx emits `Reshape(shape=Concat(Unsqueeze(Shape(x)), ...))` even when the
target shape is static. Pulsar2's shape inference cannot resolve those and fails
with `Reshape, shapefn failed`. The output shape is known from ONNX shape
inference, so replace the shape input with a constant initializer.

    python export/fix_reshape.py [graphs...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = ROOT / "export" / "onnx"
SHAPE_OPS = {"Reshape": 1, "Expand": 1, "Tile": 1}


def fold(path):
    model = onnx.load(str(path))
    model = onnx.shape_inference.infer_shapes(model)
    inits = {i.name for i in model.graph.initializer}
    values = {i.name: i for i in model.graph.value_info}
    for out in model.graph.output:
        values[out.name] = out
    producers = {n.output[0]: n for n in model.graph.node if n.output}
    changed = 0
    extra = []
    for node in model.graph.node:
        index = SHAPE_OPS.get(node.op_type)
        if index is None:
            continue
        name = node.input[index]
        if name in inits:
            continue
        info = values.get(node.output[0])
        if info is None or not info.type.HasField("tensor_type"):
            continue
        dims = info.type.tensor_type.shape.dim
        if not dims or any(not d.HasField("dim_value") or d.dim_value <= 0 for d in dims):
            continue
        target = np.array([d.dim_value for d in dims], np.int64)
        const_name = f"{node.name or node.output[0]}_p2shape"
        extra.append(numpy_helper.from_array(target, const_name))
        node.input[index] = const_name
        changed += 1
    if changed:
        model.graph.initializer.extend(extra)
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
    print(f"{path.name:24s} folded {changed} shape inputs")
    return changed


def main():
    names = sys.argv[1:] or [p.name for p in sorted(ONNX_DIR.glob("*.onnx"))]
    for name in names:
        fold(ONNX_DIR / name)


if __name__ == "__main__":
    main()
