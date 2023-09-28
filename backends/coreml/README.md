## CoreML Backend


For setting up the **CoreML** backend please follow the instructions described in `backend/coreml/setup.md`. 

## AOT 

For delegating the Program to the **CoreML** backend, the client must be responsible for calling `to_backend` with the **CoreMLBackend** tag. 

```
from executorch.backends.coreml.compiler import CoreMLBackend

# This will delegate the whole program to the CoreML backend. 
lowered_module = to_backend("CoreMLBackend", edge.exported_program, [])

```

Currently, the **CoreML** backend would delegate the whole module to **CoreML**. If a specific op is not supported by the **CoreML** backend then the `to_backend` call would throw an exception. We will be adding a **Partitioner** for **CoreML** soon to resolve the issue. 

The `preprocess` code is located at `backends/coreml/coreml_preprocess.py`. The implementation uses `coremltools` to convert the **ExportedProgram** to **MLPackage** which is flattened and returned as bytes to **ExecuTorch**.

### Compute Units
 
The `CoreML` runtime by default loads the model with compute units set to `MLComputeUnitsAll`. This can be overridden by providing `compute_units` at the preprocessing stage.

**MLComputeUnitsCPUOnly**

```
# cpu is equivalent to MLComputeUnitsCPUOnly.
lowered_module = to_backend("CoreMLBackend", edge.exported_program, [CompileSpec("compute_units", bytes("cpu", 'utf-8'))])

```

**MLComputeUnitsCPUAndGPU**

```
# gpu is equivalent to MLComputeUnitsCPUAndGPU.
lowered_module = to_backend("CoreMLBackend", edge.exported_program, [CompileSpec("compute_units", bytes("gpu", 'utf-8'))])

```


**MLComputeUnitsCPUAndNeuralEngine**

```
# ane is equivalent to MLComputeUnitsCPUAndNeuralEngine.
lowered_module = to_backend("CoreMLBackend", edge.exported_program, [CompileSpec("compute_units", bytes("ane", 'utf-8'))])

```

**MLComputeUnitsAll**

```
# all is equivalent to MLComputeUnitsAll.
lowered_module = to_backend("CoreMLBackend", edge.exported_program, [CompileSpec("compute_units", bytes("all", 'utf-8'))])

```


## Runtime 

Directory Structure.
```

executorch
├── backends                        
    ├── coreml                   #  CoreML backend implementation.                 
        ├── runtime              #  CoreML runtime implementation.  
            ├── inmemoryfs       #  In-Memory filesystem implementation. 
            ├── kvstore          #  Persistent key-value store implementation.  
            ├── delegate         #  Delegate implementation.  
            ├── libraries        #  Linked libraries. 
            ├── runner           #  Runner app implementation.  
            ├── test             #  Test files and models. 

```

## Integration

There are two steps required to integrate the **CoreML** backend.
- Exporting the Program: The client must call `to_backend` with the `CoreMLBackend` tag. 

```
from executorch.backends.coreml.compiler import CoreMLBackend

# This will delegate the whole program to the CoreML backend. 
lowered_module = to_backend("CoreMLBackend", edge.exported_program, [])

```
- Running the delegated program: The client application must link with the `coremldelegate` library. Please follow the instructions described in the `backends/coreml/setup.md` to link the `coremldelegate` library. Once linked, there are no additional steps required. **ExecuTorch** would automatically call the **CoreML** delegate to execute the **CoreML** delegated part of the exported **Program**.

  
