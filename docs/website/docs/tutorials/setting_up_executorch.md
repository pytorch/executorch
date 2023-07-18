# Setting up Executorch

This is a tutorial for building and installing Executorch from the GitHub repository.

## AOT Setup

This will install an `executorch` pip package to your conda environment and allow you to export your PyTorch model to a flatbuffer file using ExecuTorch.

**Step 1: Set up a conda environment**
```bash
conda create -yn executorch python=3.10.0
conda activate executorch

conda install -c conda-forge flatbuffers

# Install the nightly builds
pip install --pre torch -i https://download.pytorch.org/whl/nightly/cpu
```

**Step 2: Set up Executorch**. This will install an  `executorch` pip package to your conda environment.
```bash
mkdir -p ~/src/
cd ~/src/

# Do one of these, depending on how your auth is set up
git clone https://github.com/pytorch/executorch.git
git clone git@github.com:pytorch/executorch.git

./executorch/install.sh

# cd into a directory that doesn't contain a `./executorch/exir` directory, since
# otherwise python will try using it for `import executorch.exir...` instead of using the
# installed pip package.
cd ~/
```

**Step 3: Try it out**

Via python script:
```
python ~/src/executorch/examples/export/export_example.py -m "add"
```

Or via python interpreter:
```
(executorch) ~/  $ python
>>> import executorch.exir as exir
>>> from executorch.exir.tests.models import Mul
>>> m = Mul()
>>> print(exir.capture(m, m.get_random_inputs()).to_edge())
>>> exir.capture(m, m.get_random_inputs()).to_edge().to_executorch().buffer
```
