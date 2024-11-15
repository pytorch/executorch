// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241115"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "c6fbdcc1ead4a09ea56314881f96a690a4d9a740bd15ad792f60c1125d883324",
    "sha256" + debug: "362a627e5d06f87c9cf0ec81ae31fd48786916e0789e4a47bfc0f7621993a744",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "c7a26fcfabec4f4d33b0e5f54efb1cd26782a7195f56934a469c9247e92ea15c",
    "sha256" + debug: "18e2dde948b8bf02fd9030de0bfeb35920c5f8baa64e1409a79117d1dc537c4b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ddcd575137f8027338830dca8fd6e823eba5608a8dd206b0482648b95743a428",
    "sha256" + debug: "8ea4261d1b320ddf33eb38d638525f27ed119deadfb3adc8ed4337f6f101af32",
  ],
  "executorch": [
    "sha256": "1109c8984efa05aa85e644aa14b30acdd30bec8c8f388cc04e1a8684f6ed130f",
    "sha256" + debug: "a2a4369b656355593d3ad0ad5a45231fd38d6ad127ae019dd9319fc984ad0578",
  ],
  "kernels_custom": [
    "sha256": "78e2622cbbfe27f38567dc7863a83a0450b6fa8f79025370c522ae63325c2b0d",
    "sha256" + debug: "a64ad7a469fadd0398ec3c7581507112ad5f04db797cf9ab6aff1ee637220dd8",
  ],
  "kernels_optimized": [
    "sha256": "23b744d0d63d503ba9a6699a6580e92510819d5b311edfbccf509ddec3b33a1c",
    "sha256" + debug: "07bb103fa0d72630984daa0893d613f1dde81647936af62bfc370392c79bae73",
  ],
  "kernels_portable": [
    "sha256": "01f8088b3c486873ec0eee7b29fc635b9f115c29075211e3b2b30e294942dc90",
    "sha256" + debug: "dd36f4f551f3e6f47f42f19e6d1885c3c7857e16bc2c8da882b017cb1164c944",
  ],
  "kernels_quantized": [
    "sha256": "8e64ee841d581c2691f35abd1b8dc56c8e31f11c0932f8a7cfbb7e03ff083a50",
    "sha256" + debug: "fc0390785a4aab7b345ea022bf7a90ed5eec15b403104154d86a46c0798d5380",
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
    .iOS(.v17),
    .macOS(.v10_15),
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
