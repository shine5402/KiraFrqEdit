"""Build the tiny ONNX fixture used by kirafrq-ml-provider's integration tests.

The graph has RMVPE's exact I/O contract but a deterministic body: given
`mel [1, 128, T]` it returns `salience [1, T, 360]` with one non-zero bin per
row, cycling with a period of 4 native frames:

  slot 0 -> bin 228, salience 1.0   (~441.5 Hz, voiced)
  slot 1 -> bin 168, salience 1.0   (~220.4 Hz, voiced)
  slot 2 -> bin 289, salience 1.0   (~891.3 Hz, above the 800 Hz ceiling)
  slot 3 -> bin 228, salience 0.0   (below the 0.03 confidence threshold)

The input's values are ignored; only its time extent matters. That is enough
to exercise session create/run plus the decoder, the #53 policy and the #47
mapping end to end without a real model in CI.

Run from the repo root with any Python that has `onnx` installed:
    python crates/kirafrq-ml-provider/tests/fixtures/make_fixture.py
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

BINS = 360
MELS = 128

BINS_CYCLE = [228, 168, 289, 228]
VALUES_CYCLE = [1.0, 1.0, 1.0, 0.0]


def build() -> onnx.ModelProto:
    mel = helper.make_tensor_value_info("mel", TensorProto.FLOAT, [1, MELS, None])
    salience = helper.make_tensor_value_info(
        "salience", TensorProto.FLOAT, [1, None, BINS]
    )

    def i64(value, name):
        return numpy_helper.from_array(np.array(value, dtype=np.int64), name=name)

    initializers = [
        i64(0, "zero"),
        i64(1, "one"),
        i64(4, "cycle"),
        i64(2, "axis2"),
        i64(BINS, "n_bins"),
        i64(BINS_CYCLE, "bins_cycle"),
        i64([0, 2], "unsqueeze_axes"),
        numpy_helper.from_array(
            np.array(VALUES_CYCLE, dtype=np.float32), name="values_cycle"
        ),
    ]

    nodes = [
        helper.make_node("Shape", ["mel"], ["mel_shape"]),
        helper.make_node("Gather", ["mel_shape", "axis2"], ["n_frames"]),
        helper.make_node("Range", ["zero", "n_frames", "one"], ["frame_index"]),
        helper.make_node("Mod", ["frame_index", "cycle"], ["slot"]),
        helper.make_node("Gather", ["bins_cycle", "slot"], ["peak_bin"]),
        helper.make_node("Gather", ["values_cycle", "slot"], ["peak_value"]),
        helper.make_node("Unsqueeze", ["peak_bin", "unsqueeze_axes"], ["peak_bin_2d"]),
        helper.make_node(
            "Unsqueeze", ["peak_value", "unsqueeze_axes"], ["peak_value_2d"]
        ),
        helper.make_node("Unsqueeze", ["one", "zero_axis"], ["batch_axis"]),
        helper.make_node("Unsqueeze", ["n_frames", "zero_axis"], ["frames_axis"]),
        helper.make_node("Unsqueeze", ["n_bins", "zero_axis"], ["bins_axis"]),
        helper.make_node(
            "Concat",
            ["batch_axis", "frames_axis", "bins_axis"],
            ["salience_shape"],
            axis=0,
        ),
        helper.make_node(
            "ConstantOfShape",
            ["salience_shape"],
            ["salience_zeros"],
            value=numpy_helper.from_array(np.array([0.0], dtype=np.float32)),
        ),
        helper.make_node(
            "ScatterElements",
            ["salience_zeros", "peak_bin_2d", "peak_value_2d"],
            ["salience"],
            axis=2,
        ),
    ]
    initializers.append(i64([0], "zero_axis"))

    graph = helper.make_graph(
        nodes, "rmvpe_tiny_fixture", [mel], [salience], initializers
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    path = Path(__file__).with_name("rmvpe_tiny.onnx")
    onnx.save(build(), path)
    print(f"wrote {path} ({path.stat().st_size} bytes)")
