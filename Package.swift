// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "latest"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "915b99d0b770c39566a082bbb24560e7a508dbedd56576a1253bde270228edb4",
    "sha256" + debug: "d0f7f6ba210c827106ee576a6d50397dcdfdb4f75f6ffb466476746aa6ecc0a5",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0dd04bf24b96f3cdfe5056c7ce9d567f1ef57220257f5d794c6573f94c23d949",
    "sha256" + debug: "23b132678d003baaba0158435aea06ce3401876747ba7a922a2850e9c0efff2c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0e7a81c0564f93df188890196c50984315db243e0e5478011d8bf586a587a84a",
    "sha256" + debug: "2dfe691468533cafa1cccbb72e875b220f0c9bdecd3f3fa454004683c381ec85",
  ],
  "executorch": [
    "sha256": "9f420604503d6aaccf1e0668ab2e6d75a3b34f2dc73cfecbe8fc4026f4e0306a",
    "sha256" + debug: "7a5ae52534a37c177cec193f1bf59361dde20fc3a7ebd95f1fb5ee60b66032d1",
  ],
  "kernels_custom": [
    "sha256": "7510b5bb742fe31338f67516405ed766d803ef9d1d5e8def789693224a5cd764",
    "sha256" + debug: "01534b4a36ac398b7167ab7a4a3ac9ffa80103292717f13c27e86180aacf8784",
  ],
  "kernels_optimized": [
    "sha256": "a38c5a4ff51b673b7a74b1e0a90679c0674cf0f8f0ca46302ea3683194a66ed5",
    "sha256" + debug: "bf59eddeef637fefa38ead5e277841854e0f8f4ec5d0e113f95534727f982232",
  ],
  "kernels_portable": [
    "sha256": "d9c8507e58923bcba5ce4dade686c5dd02bff192e3583bf818bfca94a8b4e626",
    "sha256" + debug: "989ab5ff6b651eb828424bb7a64cf43a5fad969c31f838588cfbd6d585773fc3",
  ],
  "kernels_quantized": [
    "sha256": "e4e6fd8d6fe26108563fe7303904f05150b0aaec66cb855b56ed01829eef6cf5",
    "sha256" + debug: "99c62bf9d9a3244d050da3fded9b1d207c997feeca29aa0fa24930796b136e7b",
  ],
].reduce(into: [String: [String: Any]]()) {
  $0[$1.key] = $1.value
  $0[$1.key + debug] = $1.value
}
.reduce(into: [String: [String: Any]]()) {
  var newValue = $1.value
  if $1.key.hasSuffix(debug) {
    $1.value.forEach { key, value in
      if key.hasSuffix(debug) {
        newValue[String(key.dropLast(debug.count))] = value
      }
    }
  }
  $0[$1.key] = newValue.filter { key, _ in !key.hasSuffix(debug) }
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v15),
  ],
  products: deliverables.keys.map { key in
    .library(name: key, targets: ["\(key)_dependencies"])
  }.sorted { $0.name < $1.name },
  targets: deliverables.flatMap { key, value -> [Target] in
    [
      .binaryTarget(
        name: key,
        url: "\(url)\(key)-\(version).zip",
        checksum: value["sha256"] as? String ?? ""
      ),
      .target(
        name: "\(key)_dependencies",
        dependencies: [.target(name: key)],
        path: ".swift/\(key)",
        linkerSettings:
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
