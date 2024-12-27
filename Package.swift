// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241227"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "01a72bba137819d00b94cd98ff8c07a21486e5d50bab439f47bf45cab9b85995",
    "sha256" + debug: "06b70aead87292e4ae21b56b8066729cc75d9998eb8d01fe3cce7380ae30be75",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "11c6999c9eb6822859266d3ded668b59ad78f56b694713d0ed01b45cf46d5bba",
    "sha256" + debug: "3e90d9118622d092369b929c041ebf56dd1c49e85c6fd93c98bbba070f94dc08",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a11bac24f79a845ae35ace90632c2a6ebeafbbf495301e051574c2714cf58383",
    "sha256" + debug: "42a8f4a073422792eeb7d982620a7a4b19fa3eae8981cde7f16a32964526b1de",
  ],
  "executorch": [
    "sha256": "7cb81abbd12141c04ff15bb6cf4a18ef38f693cdacc32b8d22f684e4d8cbe6af",
    "sha256" + debug: "e068cdb956a1815ee4856b1b01eecae1c073445b0421ebdab53da953987eab00",
  ],
  "kernels_custom": [
    "sha256": "6f0209f777e4eb0553c62f6f543cfe7185459c9e11bfc94563e867b47ba8b308",
    "sha256" + debug: "7d9bd9c5e5ac9bf9ad0f1ad82ebac4bd6bae9fde569fc9033ac75c0638b5704c",
  ],
  "kernels_optimized": [
    "sha256": "a0a62ed5d6a563c987037b0134fc9ea9ad24748245a524ba2d9d33edc19bd2a1",
    "sha256" + debug: "a4eabd86546b9e1ad9e5a85835767846ebb172035bc14a31c90549448ce79b83",
  ],
  "kernels_portable": [
    "sha256": "ece199f8420a2e1216631e8205c02da85853f063e914a6480651a9ab194aece3",
    "sha256" + debug: "297b65f1d95cedf859e09722ce51808ace79009e113873b3186ee010185feabb",
  ],
  "kernels_quantized": [
    "sha256": "272b9fbb92f22ccbf82405844dc5fdbab92bab234aa44afdfe7a8b0edc3034dc",
    "sha256" + debug: "82047eb3391f754f18b9aa68a373869ba656f1c4d00d80ad6a94f8e23946e32d",
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
