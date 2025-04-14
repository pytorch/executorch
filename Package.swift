// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250414"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "ca3a104cae0c287b8f5d13886cb9a1e4c22dbed45e1486f6c5f16f565094e3a3",
    "sha256" + debug: "6ca2e8d935b2aa9c70d5b3e2c4b275b2c8603dbe531024fec88e4a231809c2f5",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "41b74a86c19c32e56d75a715b3ff93c5c2695e534ca48ddea40b9972ed67f9fa",
    "sha256" + debug: "4f982dfa48e9e253cf124ef7aee805df03cfd002b30ecae46f2969fa4ee68a56",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ef466453c145a1e2729db97c3440b9dd631f4eab95fea6a90081a864b475e9d8",
    "sha256" + debug: "f36f486fddc22f98db08f3c819a50e626b947c4849baf8f6f9c026b06a6305e9",
  ],
  "executorch": [
    "sha256": "89663b6f955179fb61fe1c383b705ab17924f40c78f42e2ae006a97f32df3890",
    "sha256" + debug: "e6141f2c8a943c352a81b50e40d80571ddff92d50b4a9fd9862133e45001cd4e",
  ],
  "kernels_custom": [
    "sha256": "fd931f1c5965ed1fb51fb623535214c8bb073f7690386e770aa1ce40d7ab72bf",
    "sha256" + debug: "bee23446e19b797ce715d56d6a5acef57ed5c4af9615c59da997e4d89308d29b",
  ],
  "kernels_optimized": [
    "sha256": "c8ac4c72225612dddb740ae8d170552b8c1340f8e6892d31ca98d533ba11e189",
    "sha256" + debug: "08f8488a8f5c59da0779dd3e6398f63df4401b1360d29699169de290335a377b",
  ],
  "kernels_portable": [
    "sha256": "91f1a5c50698d4fbc8c19c6c566c261462b79ea2590287ab13e1ac7b41c1481f",
    "sha256" + debug: "1f8c0958db08f933bd4586b14dbc278c15dd19849795e427d27912880029bafa",
  ],
  "kernels_quantized": [
    "sha256": "5866bd7b81fcb419ceed99e84d2b5750902f1ad184d4b6666eef297f5427e1ac",
    "sha256" + debug: "57202c34f1d869e10ab7280760cfc058631501c618f4672c4d5507b26d56d72a",
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
        path: ".Package.swift/\(key)",
        linkerSettings:
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
