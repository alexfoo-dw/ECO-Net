"""Explicit Compositional Network."""
__all__ = ['ECONet']


def __getattr__(name):
    if name == 'ECONet':
        from .model import ECONet
        return ECONet
    raise AttributeError(name)
