import unittest
import torch
from PIL import Image
from wine_ml.core import Adapter, prepare
from wine_ml.experiment import evaluate


class MLTest(unittest.TestCase):
    def test_identity_and_ranking(self):
        torch.manual_seed(0)
        sources = torch.nn.functional.normalize(torch.randn(10, 768), dim=-1)
        features = sources[:, None].repeat(1, 4, 1)
        adapter = Adapter()
        self.assertTrue(torch.allclose(adapter(sources), sources, atol=1e-6))
        scores = evaluate(adapter, features, torch.arange(10))
        self.assertEqual(scores['accuracy_at_1'], 1.0)
        self.assertEqual(scores['recall_at_5'], 1.0)
        # Every source has two queries but occupies one top-k identity.
        self.assertEqual(scores['queries'], 20)

    def test_loss_updates_weights(self):
        torch.manual_seed(0)
        adapter = Adapter()
        x = torch.randn(8, 768)
        before = adapter(x).detach().clone()
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=.001)
        (adapter(x)[:, 0].sum()).backward()
        optimizer.step()
        self.assertFalse(torch.allclose(adapter(x), before))

    def test_aspect_ratio_padding(self):
        image = prepare(Image.new('RGB', (40, 200), 'red'))
        self.assertEqual(image.size, (384, 384))
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(image.getpixel((192, 192)), (255, 0, 0))


if __name__ == '__main__':
    unittest.main()
