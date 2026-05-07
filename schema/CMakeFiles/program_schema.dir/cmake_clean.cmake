file(REMOVE_RECURSE
  "include/executorch/schema/program_generated.h"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/program_schema.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
