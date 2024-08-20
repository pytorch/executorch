

class NativePreprocessRunner():
  """
  Runs preprocess via ExecuTorch with provided pte file.
  """
  def __init__(self, pte_file):
    super().__init__()
    self.model = _load_for_executorch(pte_file)
  
  def forward(
    self,
    image: torch.Tensor,
    target_size: torch.Tensor,
    canvas_size: torch.Tensor,
  ):
    return self.model.forward((image, target_size, canvas_size))
