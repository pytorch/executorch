file(REMOVE_RECURSE
  "../include/executorch/extension/flat_tensor/serialize/flat_tensor_generated.h"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/flat_tensor_schema.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
