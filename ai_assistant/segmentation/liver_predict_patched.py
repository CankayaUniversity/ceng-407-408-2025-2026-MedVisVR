def _patch_pools():
    class _SeqPool:
        def __init__(self, *args, **kwargs):
            pass

        def map(self, fn, iterable, **kw):
            return list(map(fn, iterable))

        def imap(self, fn, iterable, **kw):
            return iter(list(map(fn, iterable)))

        def imap_unordered(self, fn, iterable, **kw):
            return iter(list(map(fn, iterable)))

        def starmap(self, fn, iterable, **kw):
            return [fn(*a) for a in iterable]

        def close(self):   pass
        def join(self):    pass
        def terminate(self): pass
        def restart(self, *a, **kw): pass

        def __enter__(self): return self
        def __exit__(self, *a): pass

    try:
        import pathos.multiprocessing as _pmp
        _pmp.Pool         = _SeqPool
        _pmp.ProcessPool  = _SeqPool
        _pmp.ThreadPool   = _SeqPool
        _pmp._ProcessPool = _SeqPool
    except ImportError:
        pass

    try:
        import multiprocess
        import multiprocess.pool as _mpp
        multiprocess.Pool = _SeqPool
        _mpp.Pool         = _SeqPool
    except ImportError:
        pass

    try:
        import multiprocessing
        import multiprocessing.pool as _mp
        multiprocessing.Pool = _SeqPool
        _mp.Pool             = _SeqPool
    except ImportError:
        pass


_patch_pools()

from nnunet.inference.predict_simple import main  
main()
