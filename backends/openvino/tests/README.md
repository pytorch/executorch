# Unit Tests for OpenVINO Backend

## Directory Structure

Below is the layout of the `backends/openvino/tests` directory, which includes the necessary files for the example applications:

```
backends/openvino/tests
├── ops                                 # Directory with base op test script and individual op tests.
    ├── base_openvino_op_test.py        # Script which contains the base class for all op tests.
    └── test_<op_name>.py               # Individual op tests scripts.
├── models                              # Directory with model test scripts.
    └── test_classification.py          # Test script for classification models.
├── quantizer                           # Directory with quantizer test scripts.
    └── test_llm_compression.py         # Test script for llm compression using NNCF algorithms.
├── README.md                           # Documentation for unit tests (this file)
└── test_runner.py                      # Script to execute unit tests.
```

## Executing Unit Tests

### Prerequisites

Before you begin, refer to instructions provided in [OpenVINO Backend for ExecuTorch](../README.md) to install OpenVINO and ExecuTorch Python package with the OpenVINO backend into your Python environment.

### Usage

`test_runner.py` allows to run op or model tests for openvino backend.

### **Arguments**
- **`--test_type`** (optional):  
  Type of the tests to run.  
  Supported values:
  - `ops` (default)
  - `models`
  - `quantizer`

- **`--pattern`** (optional):  
  Pattern to match test files. Provide complete file name to run individual tests. The default value is `test_*.py`
  Examples:
  - `test_convolution.py` (Assuming `--test_type` parameter is provided as `ops`, this will run only convolution tests)
  - `test_add*.py` (Assuming `--test_type` parameter is provided as `ops`, this will run add and addmm op tests)

- **`--device`** (optional):  
  Target device to compile and run tests. Default is `CPU`.
  Examples: `CPU`, `GPU`


## **Examples**

### Execute Tests for All Ops on CPU
```bash
python test_runner.py --device CPU --test_type ops
```

### Execute Convolution Op Tests on CPU
```bash
python test_runner.py --device CPU --test_type ops --pattern test_convolution.py
```

### Execute Tests for all Models on GPU
```bash
python test_runner.py --device GPU --test_type models
