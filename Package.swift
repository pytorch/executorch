// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250204"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "f82f107cd9dfdf2b69cb9761ae1c69ad11f407eee36cf6ea261a4dedfc51b7f2",
    "sha256" + debug: "404202be57e435d74df29b463616454449dc0b1b673aac5eb7459abd1f169477",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "fe56aa1f1c90288b55fb9f695b378fb6e4322c6f387bf406e98048224a20b6e1",
    "sha256" + debug: "ccd98823a57902e13aa33f370b17bd4c8d564e6408c79796f28bcade493d0dfb",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "89ec5be3befbe385b4327ecddc6fe3d93f28f09e471949547b4b16d17018eeee",
    "sha256" + debug: "1933ad4ef23a6f3debccfab8d54d8bc80be783d53387d1650327ca8db8d02e9c",
  ],
  "executorch": [
    "sha256": "a584755e13d887dd17c54091ceada8501595b8d1ce8147fcadd337e1d37cc703",
    "sha256" + debug: "d10e2e803f882e6abce0c151d742a14a07c0f564a13578d04f419b197ed9055d",
  ],
  "kernels_custom": [
    "sha256": "2fa03ac11ba2c012afbaae74ddeb4cf0ca79d71a3240d4a4519bf3fe4b40b69c",
    "sha256" + debug: "7dd06022991f5d721e2ef631042eff787d9a5582a5d65c2e940560c70520535c",
  ],
  "kernels_optimized": [
    "sha256": "580319cf9b441c21d559e15435f430b236ef0b6f7aef8eae608fc2ba694b5b3d",
    "sha256" + debug: "18641c53ac50a8b818ae3f67a392c545f7d62c097943f9e25128ff1f0a1603c1",
  ],
  "kernels_portable": [
    "sha256": "6a604f4ea2f88890e4fc7541e984febe2ad8f21065e6c6f8f0ef5ca3ce18cc86",
    "sha256" + debug: "892915bbb98c1f28b2116bf1ac0408f4517a7b283aef5c8e7699593d11700593",
  ],
  "kernels_quantized": [
    "sha256": "33f1fdb6cbce3be1b708e576cc3adaf1fc227d70108d2c73faeaac2095e304a9",
    "sha256" + debug: "7a1b2225aab9b5b27129eb19967bd044c0a98809f6ced34233ab15d8d0586932",
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
