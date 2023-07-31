## Dependencies

Various models listed in this directory have dependencies on some other packages, e.g. torchvision, torchaudio.
In order to make sure model's listed in examples are importable, e.g. via
```
from executorch.examples.models.mobilenet_v3d import MV3Model
m = MV3Model.get_model()
```
we need to have appropriate packages installed. You should install these deps via install_requirements.sh
