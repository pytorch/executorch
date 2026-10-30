# Open-source builds have no Meta CI configuration, so targets that link a ggml
# GPU backend set get no CI labels.
def ggml_ci_labels(_backend):
    return []
