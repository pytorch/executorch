import torchvision
import yaml

from executorch.parseq.parseq import PARSeq, Tokenizer
from PIL import Image


def get_transform(img_size=(32, 128)):
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                img_size, torchvision.transforms.InterpolationMode.BICUBIC
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.5, 0.5),
        ]
    )


# Gray dummy image
def get_dummy_input(img_h=32, img_w=128, n_channels=3):
    transform = get_transform((img_h, img_w))
    image = Image.new("RGB", (img_w, img_h), color=128)
    image = transform(image)
    image = image.view(1, *image.size())
    return image


def prepare_export_model(model_name, export_mode):
    with open(f"parseq/{model_name}.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    print(cfg)
    cfg.pop("name")
    cfg.pop("_target_")
    cfg.pop("lr")
    cfg.pop("perm_num")
    cfg.pop("perm_forward")
    cfg.pop("perm_mirrored")
    charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    tokenizer = Tokenizer(charset)
    lightning_model = PARSeq(
        tokenizer,
        25,
        (32, 128),
        **cfg,
    )

    model = lightning_model
    model.export_mode = export_mode
    return model.eval()
