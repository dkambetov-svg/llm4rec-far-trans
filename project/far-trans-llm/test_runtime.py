"""CPU-only tests of LocalGenerator dtype selection with fake torch/transformers modules.

No model is loaded; these check which dtype is requested from from_pretrained.
"""
import sys
import unittest
from unittest.mock import MagicMock, patch

import runtime

# Metadata keys before --dtype existed. They feed the generation cache key, so the
# set must not change, or default runs would stop matching existing caches.
ORIGINAL_METADATA_KEYS = {'model', 'requested_revision', 'resolved_revision', 'device', 'dtype',
                          'torch_version', 'transformers_version', 'tokenizer_class',
                          'do_sample', 'thinking', 'context_limit'}


class Dtype:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f'torch.{self.name}'


def fake_modules(cuda_available):
    torch = MagicMock(name='torch')
    torch.__version__ = 'fake-torch'
    torch.cuda.is_available.return_value = cuda_available
    for name in ('bfloat16', 'float16', 'float32'):
        setattr(torch, name, Dtype(name))
    transformers = MagicMock(name='transformers')
    transformers.__version__ = 'fake-transformers'
    model = MagicMock(name='model')
    model.config._commit_hash = 'fake-commit'
    model.to.return_value = model
    transformers.AutoModelForCausalLM.from_pretrained.return_value = model
    return {'torch': torch, 'transformers': transformers}


class DtypeTests(unittest.TestCase):
    def load(self, device, dtype=None, cuda_available=True):
        modules = fake_modules(cuda_available)
        kwargs = {} if dtype is None else {'dtype': dtype}
        with patch.dict(sys.modules, modules):
            generator = runtime.LocalGenerator('fake/model', 'rev', device, False, **kwargs)
        call = modules['transformers'].AutoModelForCausalLM.from_pretrained.call_args
        return generator, call.kwargs['torch_dtype']

    def test_auto_is_original_behaviour(self):
        for device, kwargs, expected in (('cuda', {}, 'torch.bfloat16'),
                                         ('cuda', {'dtype': 'auto'}, 'torch.bfloat16'),
                                         ('cpu', {}, 'torch.float32'),
                                         ('cpu', {'dtype': 'auto'}, 'torch.float32')):
            with self.subTest(device=device, kwargs=kwargs):
                generator, requested = self.load(device, **kwargs)
                self.assertEqual(str(requested), expected)
                self.assertEqual(generator.metadata['dtype'], expected)
                self.assertEqual(set(generator.metadata), ORIGINAL_METADATA_KEYS)

    def test_explicit_dtypes(self):
        for dtype in ('float16', 'bfloat16'):
            with self.subTest(dtype=dtype):
                generator, requested = self.load('cuda', dtype)
                self.assertEqual(str(requested), f'torch.{dtype}')
                self.assertEqual(generator.metadata['dtype'], f'torch.{dtype}')

    def test_invalid_dtype_rejected_before_torch_import(self):
        with patch.dict(sys.modules, {'torch': None}), self.assertRaises(ValueError):
            runtime.LocalGenerator('fake/model', dtype='float32')


if __name__ == '__main__':
    unittest.main()
