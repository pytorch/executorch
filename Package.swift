// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241220"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "7fd743a84e0144fad6653c77b5077102ac6a9b4d482ae048cfa3ae2d816fda6e",
    "sha256" + debug: "41aad93b826bf978d102fbe35a8407f2a8e9edc115a483992dad6e59cb8ae99b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3c60583ff0909cef20991f228c24231d8f597aeef48003b752396434057bc8df",
    "sha256" + debug: "3f24010d182b7db6f962255345788811f91612c9880586f7848041ad363624c6",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e6be4660641a3e9b38a727755f752d8e653b8ac97c5621196c2ce5a0d00e99e5",
    "sha256" + debug: "c1807f8396cffffc23732894533634f2e3977b6baf9f2b185871aa4035042e80",
  ],
  "executorch": [
    "sha256": "3c9d1313659b323028ab5e54b9cc280bdfe9a505156de5005486d77bd98811a6",
    "sha256" + debug: "a57e918bcb7d3c064c776809f4576e8b87461b805c85a4572d18e72a7da6dc10",
  ],
  "kernels_custom": [
    "sha256": "5b69572fb22f4319a3273735b423c68e8e44179e99d00c536890a429d97c5623",
    "sha256" + debug: "2c83d91a41c8c008298c7233e07cb5b7e9b3f350d9d95da6145c8c6932323645",
  ],
  "kernels_optimized": [
    "sha256": "69866b323579e1a563595adbcb303e47ab755e75e743456cb711add2b68d0c8e",
    "sha256" + debug: "47afd8742e3589be2e95191441ae050b29a11a5d0f336727c89dcc9eed93f1de",
  ],
  "kernels_portable": [
    "sha256": "8249285ac2337017ceeb29b8c336d552dfb7374ec350eba4530bbe66e04e3940",
    "sha256" + debug: "bde4d06044f90419fed1609b8fcea1bc6d6a2d1728a2c47b78032a2574bfd9b0",
  ],
  "kernels_quantized": [
    "sha256": "ea9480d66bcf1dce1adda746031fe2874399129c010a89c80bab728012ca247b",
    "sha256" + debug: "0776e5db9514aadd3a61c11ca480aa1a50efe1b25f6449d3c3aa15433f6c4465",
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
