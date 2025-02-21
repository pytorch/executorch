// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250221"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "262f1d9757cec5595442294c273fee9e47271062bbfa3f2c25c150b86df3f73c",
    "sha256" + debug: "d4faf2fb2367e1e032f2080f65766c4121876e12f6bdaa7fbc87d2bd62e4ad01",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b40d52e18c8d958f29c09e175327157b4c256bf9503310cba2c4513a2dc1ab7f",
    "sha256" + debug: "f138f744a22972c85677d619e91f764b56fbf8aad122ee5cbd99367b96ee87e5",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d48394ef5b26c2af1c89c2375ea4c094bdd3ed80f324452b60f33006a48496fd",
    "sha256" + debug: "312a32e8ffc1b69ea4d49dad67a55595d77c1b605a5043793d4938d3b7483758",
  ],
  "executorch": [
    "sha256": "96c0c9742795b1ea1ad8850dacc79f3db41c6945bb620bd8df3c5ab3b0138373",
    "sha256" + debug: "9a9900693a148ab67c7133d3a63223c8c7a2e36d734c6ec621a0103dc0ecfae8",
  ],
  "kernels_custom": [
    "sha256": "5d2cbe83a3ae7fca775eb1e5c8302f005c8ce63fb3b91177d3ddc01ee63cd4d4",
    "sha256" + debug: "37ddce5d39ff170166e48cd2da8e31ff8f00f93aae237dd83cf971aaf53afae0",
  ],
  "kernels_optimized": [
    "sha256": "b9da16718e91ef1f6bd1c1fb6c3f85052b99e370f4db18826c372ba6524cd2ae",
    "sha256" + debug: "58491936eb5b8a46ca86f4d3e113798bcb44b5c9ebc40ae2e4457e83c97908b0",
  ],
  "kernels_portable": [
    "sha256": "7ad949d8b0c8cf380c2927237cab634c443a1cc335ce435128406b2a30ac8021",
    "sha256" + debug: "b059f95205ffdfbb4294cd0b4b8ebebf313fb5c716e7b0bea0830607f5e39ce0",
  ],
  "kernels_quantized": [
    "sha256": "cedfd93b654002b081694a1f616f102ae161e5d9800ad0d6440c3822bced43e1",
    "sha256" + debug: "720bb28df867f7da011aad97a38cdf91a0012ee25b5b2e0b426b1e986c1d080d",
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
