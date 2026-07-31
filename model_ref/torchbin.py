"""
torchbin.py -- read a legacy (pre-1.6) PyTorch .bin state_dict into numpy
arrays, with no torch dependency.

WHY THIS EXISTS

The reference model for this accelerator is Helsinki-NLP/opus-mt-en-ar, and
the only weight file that repository publishes is `pytorch_model.bin` -- no
safetensors. Installing torch to read 300 MB of float32 would pull ~1 GB of
CUDA-less wheels onto a machine whose entire job here is to run GHDL and
numpy. The file format is documented and stable, so we read it directly.

FORMAT (torch.serialization._legacy_save)

    pickle: magic number 0x1950a86a20f9469cfc6c
    pickle: protocol version (1001)
    pickle: sys_info dict
    pickle: the object graph, where every storage is a *persistent id*
            ('storage', storage_type, root_key, location, numel, view_meta)
    pickle: the list of storage keys, in the order their bytes follow
    raw   : for each key -- int64 numel, then numel * itemsize bytes

Tensors appear in the graph as calls to torch._utils._rebuild_tensor_v2,
which carries (storage, offset, shape, stride). We record those calls, read
the storage bytes afterwards, and assemble with numpy stride tricks so that
non-contiguous tensors (transposed weights, shared embeddings) come out
correct rather than merely correctly shaped.
"""

import pickle
import struct
import numpy as np


# torch storage class name -> numpy dtype. Only the ones a published
# checkpoint can actually contain.
_STORAGE_DTYPE = {
    "FloatStorage": np.dtype("<f4"),
    "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"),
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("<i1"),
    "ByteStorage": np.dtype("<u1"),
    "BoolStorage": np.dtype("?"),
}


class _Storage:
    """Placeholder for one contiguous block of tensor data."""

    def __init__(self, key, dtype, numel):
        self.key = key
        self.dtype = dtype
        self.numel = numel
        self.data = None        # filled in by the raw-data pass


class _TensorSpec:
    """A _rebuild_tensor_v2 call, deferred until the storages are read."""

    def __init__(self, storage, offset, shape, stride):
        self.storage = storage
        self.offset = offset
        self.shape = tuple(shape)
        self.stride = tuple(stride)

    def materialise(self):
        buf = self.storage.data
        itemsize = self.storage.dtype.itemsize
        if not self.shape:
            return buf[self.offset]
        # PyTorch strides count ELEMENTS; numpy's count BYTES.
        arr = np.lib.stride_tricks.as_strided(
            buf[self.offset:],
            shape=self.shape,
            strides=tuple(s * itemsize for s in self.stride),
        )
        # as_strided returns a view onto the storage; copy so the result is
        # independent and contiguous. Transposed weights are non-contiguous
        # here, and every consumer downstream assumes C order.
        return np.ascontiguousarray(arr)


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *_rest):
    return _TensorSpec(storage, storage_offset, size, stride)


def _rebuild_tensor(storage, storage_offset, size, stride, *_rest):
    return _TensorSpec(storage, storage_offset, size, stride)


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, storages):
        super().__init__(f, encoding="utf-8")
        self._storages = storages

    def find_class(self, module, name):
        if module.startswith("torch") and name.endswith("Storage"):
            return name                       # the *name* is all we need
        if module == "torch._utils":
            if name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if name == "_rebuild_tensor":
                return _rebuild_tensor
        if module == "collections" and name == "OrderedDict":
            # Must be the real OrderedDict: pickle reconstructs it with a
            # REDUCE followed by a BUILD that writes __dict__, and a plain
            # dict has no __dict__ to write.
            import collections

            return collections.OrderedDict
        if module == "torch" and name == "device":
            return lambda *a, **k: None
        return super().find_class(module, name)

    def persistent_load(self, pid):
        # ('storage', storage_type, root_key, location, numel, view_meta)
        assert pid[0] == "storage", f"unexpected persistent id {pid[0]!r}"
        storage_type, root_key, _location, numel = pid[1], pid[2], pid[3], pid[4]
        if isinstance(storage_type, bytes):
            storage_type = storage_type.decode("ascii")
        if root_key not in self._storages:
            dtype = _STORAGE_DTYPE.get(storage_type)
            if dtype is None:
                raise ValueError(f"unhandled storage type {storage_type!r}")
            self._storages[root_key] = _Storage(root_key, dtype, numel)
        return self._storages[root_key]


def load(path):
    """Return {name: np.ndarray} for a legacy PyTorch .bin checkpoint."""
    storages = {}
    with open(path, "rb") as f:
        magic = pickle.load(f, encoding="utf-8")
        if magic != 0x1950A86A20F9469CFC6C:
            raise ValueError(
                "not a legacy torch checkpoint (magic %#x). A zip-format "
                ".bin needs the >=1.6 reader, not this one." % magic
            )
        proto = pickle.load(f, encoding="utf-8")
        if proto != 1001:
            raise ValueError(f"unsupported protocol version {proto}")
        pickle.load(f, encoding="utf-8")           # sys_info, unused

        graph = _Unpickler(f, storages).load()
        order = pickle.load(f, encoding="utf-8")   # storage keys, in file order

        for key in order:
            st = storages[key]
            (numel,) = struct.unpack("<q", f.read(8))
            if numel != st.numel:
                raise ValueError(
                    f"storage {key}: header says {numel}, graph says {st.numel}"
                )
            raw = f.read(numel * st.dtype.itemsize)
            if len(raw) != numel * st.dtype.itemsize:
                raise ValueError(f"storage {key}: truncated")
            st.data = np.frombuffer(raw, dtype=st.dtype)

    out = {}
    for name, val in graph.items():
        if isinstance(val, _TensorSpec):
            out[name] = val.materialise()
    return out


if __name__ == "__main__":
    import sys

    sd = load(sys.argv[1])
    print(f"{len(sd)} tensors")
    total = 0
    for k, v in sd.items():
        total += v.size
        print(f"  {k:60s} {str(v.shape):20s} {v.dtype}")
    print(f"total elements: {total:,}")
