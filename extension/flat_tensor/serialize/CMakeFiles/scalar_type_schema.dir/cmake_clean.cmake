file(REMOVE_RECURSE
  "../include/executorch/extension/flat_tensor/serialize/scalar_type_generated.h"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/scalar_type_schema.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
