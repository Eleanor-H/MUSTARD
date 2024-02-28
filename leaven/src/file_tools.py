import collections
import os

def is_sequence(obj):
    return isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str)

def pack_varargs(args):
    assert isinstance(args, tuple), "please input the tuple `args` as in *args"
    if len(args) == 1 and is_sequence(args[0]):
        return args[0]
    else:
        return args

def f_not_empty(*fpaths):
    fpath = f_join(*fpaths)
    if not os.path.exists(fpath):
        return False

    if os.path.isdir(fpath):
        return len(os.listdir(fpath)) > 0
    else:
        return os.path.getsize(fpath) > 0


def f_expand(fpath):
    return os.path.expandvars(os.path.expanduser(fpath))

def f_join(*fpaths):
    fpaths = pack_varargs(fpaths)
    fpath = f_expand(os.path.join(*fpaths))
    if isinstance(fpath, str):
        fpath = fpath.strip()
    return fpath

def f_mkdir(*fpaths):
    fpath = f_join(*fpaths)
    os.makedirs(fpath, exist_ok=True)
    return fpath