import io, json, os, sys, unittest
from contextlib import redirect_stdout
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import fitsheet as fs
EX = os.path.join(HERE, "..", "examples")


def ex(name):
    with open(os.path.join(EX, name)) as f:
        return json.load(f)


def run(args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = fs.main(args)
    return rc, buf.getvalue()


class T(unittest.TestCase):
    def test_dense_params_from_config(self):
        m = fs.from_config(ex("config-dense-8b.json"))
        self.assertAlmostEqual(m["total_params_b"], 8.03, delta=0.05)
        self.assertIsNone(m["active_params_b"])

    def test_moe_active_params(self):
        m = fs.from_config(ex("config-moe-30b.json"))
        self.assertAlmostEqual(m["total_params_b"], 30.5, delta=0.3)
        self.assertLess(m["active_params_b"], 4)

    def test_kv_per_token_gqa(self):
        m = fs.from_config(ex("config-dense-8b.json"))
        # 32 layers x 2 (K,V) x 8 heads x 128 dim x 2 bytes = 131072 bytes
        self.assertEqual(fs.kv_bytes_per_token(m, "fp16", 4096), 131072)
        self.assertEqual(fs.kv_bytes_per_token(m, "q8", 4096), 65536)

    def test_sliding_layers_cap_kv(self):
        m = dict(fs.MODEL, layers=4, kv_layers=1, sliding_layers=3, sliding_window=1024, kv_heads=1, head_dim=1)
        short = fs.kv_bytes_per_token(m, "fp16", 1024)
        long = fs.kv_bytes_per_token(m, "fp16", 8192)
        self.assertGreater(short, long)

    def test_file_size_beats_bpw(self):
        m = dict(fs.MODEL, total_params_b=8, file_gb={"Q4_K_M": 4.9})
        self.assertEqual(fs.weight_gb(m, "Q4_K_M"), (4.9, "file"))
        self.assertEqual(fs.weight_gb(m, "Q8_0")[1], "bpw")

    def test_unified_memory_reserve(self):
        self.assertAlmostEqual(fs.usable_gb({"memory_gb": 32, "unified": True}), 22.4)
        self.assertAlmostEqual(fs.usable_gb({"memory_gb": 32, "unified": True, "usable_fraction": 0.8}), 25.6)

    def test_quantized_kv_unlocks_long_context(self):
        mach, mod = ex("machine-rtx-24gb.json"), fs.from_config(ex("config-dense-8b.json"))
        self.assertNotEqual(fs.row(mach, mod, "Q8_0", 131072, 1, "fp16")["fit"], "FITS")
        self.assertEqual(fs.row(mach, mod, "Q8_0", 131072, 1, "q8")["fit"], "FITS")

    def test_decode_slows_with_context(self):
        mach, mod = ex("machine-rtx-24gb.json"), fs.from_config(ex("config-dense-8b.json"))
        r = fs.row(mach, mod, "Q4_K_M", 32768, 1, "fp16")
        self.assertGreater(r["decode_short_tps"], r["decode_full_tps"])

    def test_moe_decode_faster_than_dense_same_bytes(self):
        mach = ex("machine-rtx-24gb.json")
        dense = dict(fs.MODEL, total_params_b=30, layers=48, kv_heads=4, head_dim=128)
        moe = dict(dense, active_params_b=3)
        a = fs.row(mach, dense, "Q4_K_M", 4096, 1, "fp16")["decode_short_tps"]
        b = fs.row(mach, moe, "Q4_K_M", 4096, 1, "fp16")["decode_short_tps"]
        self.assertGreater(b, 5 * a)

    def test_lossy_flag_and_pick_skips_it(self):
        mach, mod = ex("machine-laptop-32gb-unified.json"), fs.from_config(ex("config-moe-30b.json"))
        rows = [fs.row(mach, mod, q, 8192, 1, "q8") for q in ("Q4_K_M", "Q3_K_M")]
        self.assertIn("LOSSY-RISK", rows[1]["flags"])
        self.assertIsNone(fs.pick(rows, 10))
        self.assertEqual(fs.pick(rows, 10, allow_tight=True)["quant"], "Q4_K_M")

    def test_sheet_cli_pick(self):
        rc, out = run(["sheet", os.path.join(EX, "machine-rtx-24gb.json"), os.path.join(EX, "model-dense-8b.json"),
                       "--contexts", "32768,131072", "--kv", "q8"])
        self.assertEqual(rc, 0); self.assertIn("Pick: Q8_0 at 131072", out)

    def test_check_verdicts(self):
        self.assertEqual(run(["check", "--predicted", "100", "--measured", "70"])[0], 0)
        rc, out = run(["check", "--predicted", "100", "--measured", "130"]); self.assertIn("ABOVE CEILING", out)
        rc, out = run(["check", "--predicted", "100", "--measured", "10"]); self.assertIn("BROKEN", out)


if __name__ == "__main__":
    unittest.main()
