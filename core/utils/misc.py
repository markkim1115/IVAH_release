from typing import Mapping, TypeVar
KT = TypeVar("KT")  # key type
VT = TypeVar("VT")  # value type

class DotDict(dict, Mapping[KT, VT]):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    def update(self, dct=None, **kwargs):
        if dct is None:
            dct = kwargs
        else:
            dct.update(kwargs)
        for k, v in dct.items():
            if k in self:
                target_type = type(self[k])
                if not isinstance(v, target_type):
                    # NOTE: bool('False') will be True
                    if target_type == bool and isinstance(v, str):
                        dct[k] = v == 'True'
                    else:
                        dct[k] = target_type(v)
        dict.update(self, dct)

    def __hash__(self):
        return hash(''.join([str(self.values().__hash__())]))

    def __init__(self, dct=None, **kwargs):
        if dct is None:
            dct = kwargs
        else:
            dct.update(kwargs)
        if dct is not None:
            for key, value in dct.items():
                if hasattr(value, 'keys'):
                    value = DotDict(value)
                self[key] = value

    """
    Uncomment following lines and 
    comment out __getattr__ = dict.__getitem__ to get feature:
    
    returns empty numpy array for undefined keys, so that you can easily copy things around
    TODO: potential caveat, harder to trace where this is set to np.array([], dtype=np.float32)
    """

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError as e:
            raise AttributeError(e)
    # MARK: Might encounter exception in newer version of pytorch
    # Traceback (most recent call last):
    #   File "/home/xuzhen/miniconda3/envs/torch/lib/python3.9/multiprocessing/queues.py", line 245, in _feed
    #     obj = _ForkingPickler.dumps(obj)
    #   File "/home/xuzhen/miniconda3/envs/torch/lib/python3.9/multiprocessing/reduction.py", line 51, in dumps
    #     cls(buf, protocol).dump(obj)
    # KeyError: '__getstate__'
    # MARK: Because you allow your __getattr__() implementation to raise the wrong kind of exception.
    __getattr__ = __getitem__  # overidden dict.__getitem__
    # __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def count_parameters(module):
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {'total': total_params, 'trainable': trainable_params}