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
    "sha256": "4ccadc1f34f9983835454caf15686f1146feea95208a36117d4fb6bbdf643857",
    "sha256" + debug: "750b192f8b7c6ebed83f18c7bc061f9d55bc8522c4808612c29611306d581865",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "c50ef9383daf75c6f23e3c5abc24dc47fb4ac5b24049337cb345fe3614d67cf4",
    "sha256" + debug: "ad35037adf527a8948d2005eb4ebc0f37c481107a228baa267d2ef373f0295c0",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "39cb67809a2ad77d2c95bf6feb3b197826b34f241dab4b9ce60b26c04f2b57c8",
    "sha256" + debug: "ffcce2bb43278127ce8074ee835b744e0de263a389c1ffe3b9f902b8a6fecc8d",
  ],
  "executorch": [
    "sha256": "973adbafd01d780640f222840bed244c866303a1f52bba56e7f5bd1477037db3",
    "sha256" + debug: "cedaa5ff3a51f7971fd8f7a587921e6be8bdb85f26d2c1d2b834e0c6e1d7ed1a",
  ],
  "kernels_custom": [
    "sha256": "1b4975234e6b97349150fe26cc55b6d4732763b60f09a2cad7d03af4356ad846",
    "sha256" + debug: "d05f53f63ff24cf06a0101d8a7639e9af3193280917474bb5d7d8c7c27b5040d",
  ],
  "kernels_optimized": [
    "sha256": "41f053ba68bd9a1c28cde2956090a7b79d45adf8251388b39b4e80ca21549373",
    "sha256" + debug: "b5fe78c12333cdad365e2f60c66ef4d6a166a3fcde500d5a7b1e57d40f864987",
  ],
  "kernels_portable": [
    "sha256": "fb82fb2a275cda5504e71828a5f9af787dd08284f1ab43c3fc56e0478c679425",
    "sha256" + debug: "390f4dfe92e90034ee2bb8cebc4c8059f8cbcc42e31cb7af66d3a9810c08e19f",
  ],
  "kernels_quantized": [
    "sha256": "f7fbd63876a2ceb6e9684088db82c83e8f49ef99b4bb1049c39477167e91f7ef",
    "sha256" + debug: "aed83775b9d59907e3b6f08543d871fed25be2d680250ff79c1647e93033111f",
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
