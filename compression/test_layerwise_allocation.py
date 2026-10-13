import itertools
import unittest

from layerwise_allocation import allocate_widths, overlay_widths, validate_layer_widths, evaluation_candidate


class AllocationTests(unittest.TestCase):
    def test_dynamic_program_matches_exhaustive_budget_search(self):
        curves = [[{'width': w, 'calibration_nmse': e} for w, e in values] for values in
                  [[(128, .8), (256, .3), (384, .1)],
                   [(128, .2), (256, .18), (384, .17)],
                   [(128, .9), (256, .4), (384, .15)]]]
        result = allocate_widths(curves, channel_budget=768, minimum_width=128)
        feasible = [rows for rows in itertools.product(*curves) if sum(r['width'] for r in rows) <= 768]
        best = min(feasible, key=lambda rows: (sum(r['calibration_nmse'] for r in rows),
                                             sum(r['width'] for r in rows), tuple(r['width'] for r in rows)))
        self.assertEqual(result['layer_widths'], [r['width'] for r in best])
        self.assertGreater(result['layer_widths'][0], result['layer_widths'][1])

    def test_protection_floor_and_infeasible_budget(self):
        curves = [[{'width': 128, 'calibration_nmse': .2}, {'width': 256, 'calibration_nmse': .1}]] * 2
        result = allocate_widths(curves, channel_budget=384, minimum_width=128,
                                 protected_layers=[1], protected_minimum=256)
        self.assertEqual(result['layer_widths'], [128, 256])
        with self.assertRaises(ValueError):
            allocate_widths(curves, channel_budget=256, minimum_width=128,
                            protected_layers=[1], protected_minimum=256)

    def test_nonmonotonic_measured_error_is_not_faked(self):
        curves = [[{'width': 128, 'calibration_nmse': .1}, {'width': 256, 'calibration_nmse': .2}]]
        self.assertEqual(allocate_widths(curves, channel_budget=256, minimum_width=128)['layer_widths'], [128])

    def test_invalid_scores_and_widths(self):
        for error in [float('nan'), float('inf'), -.1]:
            with self.assertRaises(ValueError):
                allocate_widths([[{'width': 128, 'calibration_nmse': error}]], channel_budget=128)
        for widths in [[128] * 35, [True] * 36, [1409] * 36, [12416] * 36]:
            with self.assertRaises(ValueError):
                validate_layer_widths(widths)

    def test_uniform_and_layerwise_overlay_checks(self):
        widths = [2560, 5376] * 18
        layers = [{'layer': i, 'original_width': 12288, 'keep_width': w} for i, w in enumerate(widths)]
        manifest = {'keep_width': 'layerwise', 'layer_widths': widths, 'layers': layers}
        selected = {'keep_width': 'layerwise', 'layer_widths': widths}
        self.assertEqual(overlay_widths(manifest, selected), widths)
        manifest['layers'][2]['keep_width'] = 5376
        with self.assertRaises(ValueError):
            overlay_widths(manifest, selected)
        uniform = {'keep_width': 5376, 'layers': [{'layer': i, 'original_width': 12288, 'keep_width': 5376} for i in range(36)]}
        self.assertEqual(overlay_widths(uniform, {'keep_width': 5376}), [5376] * 36)

    def test_diagnostic_does_not_fabricate_a_passing_selection(self):
        selection = {'status': 'no_validation_passing_candidate', 'selection_uses_holdout': False, 'selected': None}
        candidate = {'status': 'completed', 'keep_width': 1408, 'params': 6265562354,
                     'vlm_params': 3984969968, 'overlay_directory': '/tmp/overlay',
                     'validation_mean_minADE_m': 8.0, 'validation_pass': False}
        with self.assertRaises(ValueError):
            evaluation_candidate(selection)
        selected = evaluation_candidate(selection, diagnostic_label='1408', candidate=candidate)
        self.assertFalse(selected['validation_pass'])
        self.assertIsNone(selection['selected'])
        with self.assertRaises(ValueError):
            evaluation_candidate(selection, diagnostic_label='2560', candidate=candidate)
        with self.assertRaises(ValueError):
            evaluation_candidate(selection, diagnostic_label='1408', candidate=dict(candidate, status='failed'))


if __name__ == '__main__':
    unittest.main()
