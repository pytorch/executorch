// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250117"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "f60fd8337faf42dfabed916b5378c6a13332d68fee1ab4aa99a3491998cc8586",
    "sha256" + debug: "04c9829f0daa674e8da0c0581405167006fa865639f208e48d8bef69e0063aa9",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "feccabed78c4230560249db8e2b16f43d127fb7ba059f1c2bf64c58e09d755ab",
    "sha256" + debug: "e72b53be195ab46516deeb4dd109ce0620169acf7e242acced42db812d9456b6",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "9827a2a3eb8facc4e2da45e7e71e7d6a42376f257215c50b01ad71e2b1bd7f4c",
    "sha256" + debug: "16e3e56de679f2dfe4c421df6edd31fa8aa4f44d7cb70ed7f46a026d75c3cdb9",
  ],
  "executorch": [
    "sha256": "d11a7443bdb13262796ca9880a5d081e022258f0295523ab036c82cc2a12afce",
    "sha256" + debug: "fedea1e4f648de7f421ab51ec6cb35a582cc92a47fe08c181f271b71686b38af",
  ],
  "kernels_custom": [
    "sha256": "6ad7e90e96560b2aa0c8dada589de25e49b41f90ae4c1c5eeb31e7922b2d74c9",
    "sha256" + debug: "83815e305dd98fa69b0efa27d123182ee5d9f162bc950bb56a34dda71a2dd10b",
  ],
  "kernels_optimized": [
    "sha256": "10c17bea6940c4ec18d9ec542e7f554f3f5211180d65838c9202027e51aa5c7d",
    "sha256" + debug: "e7e88fcef9340d3bde2c31b6f192f318272f05eafc787c2f3926233a51d60703",
  ],
  "kernels_portable": [
    "sha256": "8426f70dc136e4eecf85a10786963682b5ffb1b199b51c3c9805ab9d10e6c428",
    "sha256" + debug: "3e4c94388b8880501ec895902e3dcd8756fa281bd2cdf23ed88cff3cc67b8196",
  ],
  "kernels_quantized": [
    "sha256": "e47b4b6310e88a835d4c0a1da75c527b9039b751d9c62f092cd27cd5273b705b",
    "sha256" + debug: "bd47743c61f1a15cb82922531c30a7e37558ce2ad399611d19aecf8ef0170e07",
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
