"""Prepared process-local export adapter; not deployed or GPU validated.

The caller must first accept actual-GGUF numerical/timing and dynamic-M AOTI
controls and capture device capability before export. No other runtime,
native source file, quantizer, weight packer or static bucket is changed.
"""


def prepare_family(native_family, *, capability, accepted_routes=None):
    if capability != (12, 0):
        return native_family  # Exact native object/ops on A100 and other GPUs.
    from torch.fx.experimental.symbolic_shapes import statically_known_true
    from candidates_w4a16_v8.aoti import candidate_op, config_index
    if not accepted_routes:
        raise ValueError('Require current actual-GGUF/AOTI validated winning routes')
    routes = dict(accepted_routes)
    for (bucket, n, k), config in routes.items():
        if bucket not in [8, 16] or n <= 0 or k <= 0 or k % 256:
            raise ValueError('Invalid accepted W4 route')
        config_index(config)

    class Family:
        name = native_family.name
        buckets = native_family.buckets

        def __init__(self):
            self._ops = {}
            for bucket in self.buckets:
                original = native_family.op(bucket)
                if bucket not in [8, 16]:
                    self._ops[bucket] = original
                    continue

                def launch(x, q, sc, ss, z, zs, group_size,
                           bucket=bucket, original=original):
                    lower = 5 if bucket == 8 else 9
                    m = x.shape[0]
                    # No guard that narrows an unproven dynamic-M interval.
                    if (statically_known_true(m >= lower)
                            and statically_known_true(m <= bucket)):
                        n, k = q.shape[0], x.shape[1]
                        if isinstance(n, int) and isinstance(k, int):
                            config = routes.get((bucket, n, k))
                            if config is not None:
                                return candidate_op(x, q, sc, ss, z, zs,
                                                    config_index(config))
                    return original(x, q, sc, ss, z, zs, group_size)

                self._ops[bucket] = launch

        def op(self, bucket):
            return self._ops[bucket]

        def supports(self, bucket, *args):
            return native_family.supports(bucket, *args)

        def validate(self, bucket, *args):
            return native_family.validate(bucket, *args)

    return Family()
