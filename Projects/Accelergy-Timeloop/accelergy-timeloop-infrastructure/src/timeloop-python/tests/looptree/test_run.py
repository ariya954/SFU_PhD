from pathlib import Path
import unittest

from pytimeloop.looptree.run import run_looptree

from tests.util import TEST_TMP_DIR


class TestCompleteRun(unittest.TestCase):
    def test_fused_sequential(self):
        BINDINGS = {
            0: 'MainMemory',
            1: 'GlobalBuffer',
            2: 'GlobalBuffer',
            3: 'GlobalBuffer',
            4: 'MACC'
        }

        stats = run_looptree(
            Path(__file__).parent.parent / 'test_configs',
            [
                'looptree-test-fused.yaml',
                'cascaded_mm.workload.yaml',
                'three_level.arch.yaml'
            ],
            TEST_TMP_DIR,
            BINDINGS,
            True
        )

        self.assertEqual(54, stats.latency)

        ENERGY_REFS = {
            ('MainMemory', 'read'): 118784,
            ('MainMemory', 'write'): 147456,
            ('GlobalBuffer', 'read'): 124842.654,
            ('GlobalBuffer', 'write'): 89549.356,
            ('MACC', 'compute'): 304.2
        }

        for k, v in stats.energy.items():
            self.assertAlmostEqual(ENERGY_REFS[k], v, 1)
