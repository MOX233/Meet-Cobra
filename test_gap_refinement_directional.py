"""The R1C6 harness must change only the local iteration configuration."""
import unittest
from unittest.mock import Mock, patch

from experiment.gap_refinement_directional import CONFIGS, iteration_override
from utils import alg_utils


class IterationOverrideTests(unittest.TestCase):
    def test_only_configuration_is_replaced_and_hook_is_restored(self):
        marker = object()
        original = Mock(return_value=marker)
        common = dict(gap_cap_rb_usage=True, ho_capacity_correction=True,
                      gap_refinement_config=CONFIGS['fixed2'], payload=marker)
        with patch.object(alg_utils, 'HO_EE_GAP_APX_SINR_conservative_adaptive', original):
            with iteration_override(CONFIGS['upto5']):
                output = alg_utils.HO_EE_GAP_APX_SINR_conservative_adaptive('args', **common)
            self.assertIs(output, marker)
            self.assertIs(alg_utils.HO_EE_GAP_APX_SINR_conservative_adaptive, original)
        original.assert_called_once_with('args', **dict(common, gap_refinement_config=CONFIGS['upto5']))
        self.assertEqual(common['gap_refinement_config'], CONFIGS['fixed2'])

    def test_wrong_default_or_disabled_capacity_correction_is_rejected(self):
        for config, capped, corrected in ((CONFIGS['upto3'], True, True),
                                           (CONFIGS['fixed2'], False, True),
                                           (CONFIGS['fixed2'], True, False)):
            original = Mock()
            with patch.object(alg_utils, 'HO_EE_GAP_APX_SINR_conservative_adaptive', original):
                with iteration_override(CONFIGS['upto3']):
                    with self.assertRaises(AssertionError):
                        alg_utils.HO_EE_GAP_APX_SINR_conservative_adaptive(
                            gap_refinement_config=config, gap_cap_rb_usage=capped,
                            ho_capacity_correction=corrected)
                self.assertIs(alg_utils.HO_EE_GAP_APX_SINR_conservative_adaptive, original)
            original.assert_not_called()


if __name__ == '__main__':
    unittest.main()
