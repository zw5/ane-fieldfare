#!/usr/bin/env python3

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from package_cross_layer import build_manifest, load_identities


def tensor(offset, size, shape, bits=None):
    value = {"offset": offset, "size": size, "shape": shape,
             "dtype": "U32" if bits else "BF16"}
    if bits is not None:
        value["bits"] = bits
    return value


class CrossLayerPackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        packed = self.root / "packed_experts"
        packed.mkdir()
        (packed / "layer0.bin").write_bytes(b"\0" * 128)
        (packed / "layer1.bin").write_bytes(b"\0" * 128)
        tensors = {}
        offset = 0
        for name in ("gate", "up", "down"):
            tensors[name] = tensor(offset, 4, [2, 4], 4); offset += 4
            tensors[name + "_scales"] = tensor(offset, 2, [2, 1]); offset += 2
            tensors[name + "_biases"] = tensor(offset, 2, [2, 1]); offset += 2
        layout = {"layers": [
            {"layer": layer, "file": f"layer{layer}.bin", "experts": [
                {"expert": 3, "offset": 16, "size": 32, "tensors": tensors}
            ]} for layer in (0, 1)
        ]}
        (packed / "layout.json").write_text(json.dumps(layout))

    def tearDown(self):
        self.temporary.cleanup()

    def identities(self, value):
        path = self.root / "identities.json"
        path.write_text(json.dumps(value))
        return load_identities(path)

    def test_order_names_and_exact_arithmetic(self):
        manifest = build_manifest(self.root, self.identities([[1, 3], {"layer": 0, "expert": 3}]))
        self.assertEqual([p["procedure"] for p in manifest["procedures"]],
                         ["expert_l01_e003", "expert_l00_e003"])
        self.assertEqual(manifest["stored_byte_arithmetic"]["checkpoint_tensor_payload_bytes"], 48)
        self.assertEqual(manifest["stored_byte_arithmetic"]["checkpoint_expert_spans_bytes"], 64)
        self.assertEqual(manifest["stored_byte_arithmetic"]["fp16_matrix_equivalent_bytes"], 96)
        self.assertEqual(manifest["compile_count"], 0)

    def test_rejects_duplicates_and_more_than_96(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.identities([[0, 3], [0, 3]])
        with self.assertRaisesRegex(ValueError, "96-procedure cap"):
            self.identities([[index, 0] for index in range(97)])

    def test_rejects_out_of_span_tensor(self):
        layout_path = self.root / "packed_experts/layout.json"
        layout = json.loads(layout_path.read_text())
        layout["layers"][0]["experts"][0]["tensors"]["down_biases"]["size"] = 99
        layout_path.write_text(json.dumps(layout))
        with self.assertRaisesRegex(ValueError, "exceeds expert"):
            build_manifest(self.root, [(0, 3)])


if __name__ == "__main__":
    unittest.main()
