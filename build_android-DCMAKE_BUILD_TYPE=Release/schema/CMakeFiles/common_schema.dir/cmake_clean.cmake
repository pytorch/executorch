file(REMOVE_RECURSE
  "include/executorch/scalar_type_generated.h"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/common_schema.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
