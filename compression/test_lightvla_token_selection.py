import math
import unittest
import torch
import torch.nn.functional as F
from lightvla_token_selection import select_visual_positions


class SelectionTests(unittest.TestCase):
    def test_chunked_winners_match_dense_and_preserve_nonvisual(self):
        torch.manual_seed(42)
        hidden = torch.randn(1, 47, 32)
        visual = torch.tensor([2, 3, 6, 7, 8, 15, 16, 17, 23, 24, 31])
        task = torch.tensor([35, 36, 38, 40])
        def norm(x):
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        patches, prompt = norm(hidden[:, visual]), norm(hidden[:, task])
        query = norm(F.scaled_dot_product_attention(patches, prompt, prompt))
        winners = (query @ patches.transpose(-2, -1) / math.sqrt(32)).argmax(-1).unique()
        expected = torch.ones(47, dtype=torch.bool)
        expected[visual] = False
        expected[visual[winners]] = True
        actual = select_visual_positions(hidden, visual, task, 3, 4)
        self.assertTrue(torch.equal(actual, expected.nonzero().flatten()))

    def test_invalid_span_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            select_visual_positions(torch.ones(1, 4, 8), torch.tensor([1, 2]), torch.tensor([2]))


if __name__ == '__main__':
    unittest.main()
