"""Log the original capacity error for this isolated MG benchmark server."""
import logging
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.models.muse_glimmer.serving import serve_batching

original_generation_error = ServingChat._generation_error

def log_generation_error(error):
    logging.getLogger('siliconbench.capacity_diagnostic').warning(
        'ET_NATIVE_ERROR type=%s code=%s detail=%s',
        type(error).__name__, getattr(error, 'code', None), str(error))
    return original_generation_error(error)

ServingChat._generation_error = staticmethod(log_generation_error)

if __name__ == '__main__':
    serve_batching.main()
