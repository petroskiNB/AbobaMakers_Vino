import unittest
from scripts.live_evaluation import summarize


class LiveEvaluationTest(unittest.TestCase):
    def test_no_labels_means_no_quality(self):
        result = summarize([dict(ok=True, latency_ms=100, query_id='a')])
        self.assertIsNone(result['quality'])

    def test_errors_and_extra_predicted_classes_are_counted(self):
        rows = [dict(query_id='a', predicted_slug='x', top5=['x', 'a'], ok=True, latency_ms=3000),
                dict(query_id='b', predicted_slug=None, top5=[], ok=False, latency_ms=20)]
        result = summarize(rows, {'a': 'a', 'b': 'b'})
        self.assertEqual(result['quality']['classes_in_macro_average'], 3)
        self.assertEqual(result['quality']['macro_f1_top1'], 0)
        self.assertEqual(result['quality']['recall_at_5'], .5)
        self.assertEqual(result['quality']['micro_set_f1_at_5'], .5)
        self.assertEqual(result['successful_under_3s'], 0)

    def test_perfect_top5_is_not_perfect_set_f1(self):
        rows = [dict(query_id='a', predicted_slug='a', top5=['a','b','c','d','e'], ok=True, latency_ms=100)]
        quality = summarize(rows, {'a':'a'})['quality']
        self.assertEqual(quality['macro_f1_top1'], 1)
        self.assertEqual(quality['recall_at_5'], 1)
        self.assertAlmostEqual(quality['micro_set_f1_at_5'], 1/3)
