//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#pragma once

// Export the contents of the MPSGraphPackage directly as a list of bytes.
// Any changes to this structure will break previous exported models.
struct ExirMPSGraphPackage {
  int64_t manifest_plist_offset;
  int64_t model_0_offset;
  int64_t total_bytes;
  uint8_t data[];
};
