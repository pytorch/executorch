import torch
import torch.nn as nn
from torchvision import models
from torch.utils.benchmark import Timer
import timm
import traceback
from models import model_dict


def run_torch(model_name, train=False, multithreaded=True, repeat=100):
    model = model_dict[model_name](train)
    loss_fn = nn.CrossEntropyLoss()
    ph_input = torch.randn(1, 3, 224, 224)
    ph_label = torch.randint(0, 10, (1,))

    def train_step(model, loss_fn, input, label):
        model.zero_grad()
        output = model(input)
        loss = loss_fn(output, label)
        loss.backward()
        return loss

    def eval_step(model, input):
        model.eval()
        output = model(input)
        return output

    threads = 48 if multithreaded else 1
    torch.set_num_threads(threads)

    if train:
        step = train_step
        model.train()
        stmt = 'step(model, loss_fn, ph_input, ph_label)'
        _globals = {
            'step': step,
            'model': model,
            'loss_fn': loss_fn,
            'ph_input': ph_input,
            'ph_label': ph_label
        }
    else:
        step = eval_step
        model.eval()
        stmt = 'step(model, ph_input)'
        _globals = {
            'step': step,
            'model': model,
            'ph_input': ph_input
        }
    try:
        t_torch = Timer(
            stmt=stmt,
            globals=_globals,
            num_threads=threads
        ).timeit(repeat)
        print(f"{model_name} - Torch: {t_torch.mean}")
    except Exception as e:
        print(f"{model_name} - Torch: Error")
        traceback.print_exc(file=open(f"./torch_experiment_error.txt", "a"))

    try:
        compiled_model = torch.compile(model, backend='inductor', fullgraph=True)
        train_step(compiled_model, loss_fn, ph_input, ph_label)
        t_inductor = Timer(
            stmt=stmt,
            globals=_globals,
            num_threads=threads
        ).timeit(repeat)
        print(f"{model_name} - Inductor: {t_inductor.mean}")
    except Exception as e:
        print(f"{model_name} - Inductor: Error")
        traceback.print_exc(file=open(f"./torch_experiment_error.txt", "a"))

if __name__ == "__main__":
    run_torch("mobilevit_s", train=True, multithreaded=False)
    # print()
    # run_torch("mobilevit_s", train=False, multithreaded=True)
    # print()
    # run_torch("vit_b_16", train=False, multithreaded=True)
    # run_torch("mobilenet_v2", train=False, multithreaded=True)
    # run_torch("mobilevit_s", train=False, multithreaded=True)
    # print()
    # run_torch("vit_b_16", train=False, multithreaded=False)
    # run_torch("mobilevit_s", train=False, multithreaded=False)