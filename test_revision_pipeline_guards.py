"""Fail-closed guards for revision artifacts and experiment provenance."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiment import revision_pipeline as pipeline
from experiment.revision_training import atomic_json, require_current_gain_convention
from utils.directional_service import BEAM_AVERAGE_DB_CONVENTION


class PipelineGuards(unittest.TestCase):
    def test_smoke_cache_rejected_for_formal_grid(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root/'cache.pkl'
            cache.write_bytes(b'smoke fixture')
            atomic_json(cache.with_suffix('.json'),dict(interference_label='beam-average',
                cache_sha256=pipeline.digest(cache),smoke=True,frames=9,
                interference_db_convention=BEAM_AVERAGE_DB_CONVENTION))
            args = SimpleNamespace(cache=cache,allow_smoke=False,o_mappo_policy=root/'policy.pt')
            with patch('utils.o_mappo.OMAPPPolicy.load',return_value=SimpleNamespace(config=None)), \
                 self.assertRaisesRegex(ValueError,'Smoke-trained'):
                pipeline.protocol(args)

    def test_modified_cache_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root/'cache.pkl'
            cache.write_bytes(b'original')
            atomic_json(cache.with_suffix('.json'),dict(interference_label='beam-average',
                cache_sha256=pipeline.digest(cache),smoke=False,frames=301))
            cache.write_bytes(b'altered')
            with patch('utils.o_mappo.OMAPPPolicy.load',return_value=SimpleNamespace(config=None)), \
                 self.assertRaisesRegex(ValueError,'Unvalidated'):
                pipeline.protocol(SimpleNamespace(cache=cache,allow_smoke=False,o_mappo_policy=root/'policy.pt'))

    def test_old_minus300_artifacts_are_not_reused_as_corrected(self):
        for metadata in ({},{'interference_db_convention':'10*log10(max(power,1e-30))'}):
            with self.assertRaisesRegex(ValueError,'dB convention mismatch'):
                require_current_gain_convention(metadata)
        require_current_gain_convention({'interference_db_convention':BEAM_AVERAGE_DB_CONVENTION})

    def test_modified_source_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            atomic_json(root/'protocol.json',dict(backend='cuda',code_sha256={
                'experiment/revision_pipeline.py':'not-the-current-hash'}))
            with self.assertRaisesRegex(ValueError,'Frozen code changed'):
                pipeline.validate(root)
            with self.assertRaisesRegex(ValueError,'Physical backend changed'):
                pipeline.validate(root,'cpu')

    def test_only_intact_committed_case_is_skipped(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('raw','runs','diagnostics'):
                (root/name).mkdir()
            self.assertFalse(pipeline.completed(root,'case','protocol'))
            raw = root/'raw/case.npz'
            diagnostic = root/'diagnostics/case.json'
            raw.write_bytes(b'complete raw')
            diagnostic.write_text('{}')
            atomic_json(root/'runs/case.json',dict(protocol_sha256='protocol',
                raw_sha256=pipeline.digest(raw),diagnostics_sha256=pipeline.digest(diagnostic)))
            self.assertTrue(pipeline.completed(root,'case','protocol'))
            with self.assertRaisesRegex(ValueError,'Wrong protocol'):
                pipeline.completed(root,'case','other protocol')
            raw.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Corrupt result'):
                pipeline.completed(root,'case','protocol')


if __name__ == '__main__':
    unittest.main()
