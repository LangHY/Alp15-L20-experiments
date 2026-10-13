import unittest
from types import SimpleNamespace
import torch
from torch import nn
from student_interface_distill import RecoveryLinear, interface_targets, interface_loss, logits_kl


class InterfaceRecoveryTests(unittest.TestCase):
    def test_zero_adapter_and_merge(self):
        torch.manual_seed(12)
        base = nn.Linear(7, 9, bias=False)
        x = torch.randn(3, 7)
        module = RecoveryLinear(base, rank=3)
        torch.testing.assert_close(module(x), base(x), rtol=0, atol=0)
        with torch.no_grad():
            module.b.normal_()
        expected = module(x).detach()
        actual = module.merged()(x)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(module.base.weight.requires_grad)

    def test_selected_kv_receives_gradient(self):
        hidden = torch.randn(1, 6, 8, requires_grad=True)
        key = torch.randn(1, 2, 6, 4, requires_grad=True)
        value = torch.randn(1, 2, 6, 4, requires_grad=True)
        output = SimpleNamespace(last_hidden_state=hidden, past_key_values=[(key, value)])
        selected = interface_targets(output, torch.tensor([1, 4]))
        targets = {'hidden': torch.ones_like(selected['hidden']),
                   'keys': [torch.ones_like(selected['keys'][0])],
                   'values': [torch.ones_like(selected['values'][0])]}
        loss, _ = interface_loss(selected, targets)
        loss.backward()
        for tensor in (hidden, key, value):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        self.assertEqual(key.grad[:, :, 0].abs().sum().item(), 0)

    def test_logit_kl_identity_and_gradient(self):
        torch.manual_seed(3)
        teacher = torch.randn(1, 4, 17)
        self.assertAlmostEqual(logits_kl(teacher, teacher).item(), 0.0)
        student = (teacher + torch.randn_like(teacher) * 0.2).requires_grad_()
        loss = logits_kl(student, teacher)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(student.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
