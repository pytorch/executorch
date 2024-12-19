// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241219"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "626dc99cd5bee05d32f6cc2da97439a080cfb1e46f8cf4df187e12099861f1b1",
    "sha256" + debug: "25c9d1bcf10992fd1bb1b03de2814d48e5a06b3cae54fa8fc73af5376dda9f0c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "df20e1c3b7725bf956a81fa77eaca35a5571488374fb3a57abd306b2a5972e72",
    "sha256" + debug: "91255481e7a9af36baa7abe04d23ae1f93b04d4c725876396ffc8d6b838e053c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1712373e1c7ec59503630d9a309dda3cc3ea911d49698c10c05611817b9505ef",
    "sha256" + debug: "4ffd26416ae3d0574a0612b6457d8fb438e5dd046ebe8abc90c6a2a10bbe4cea",
  ],
  "executorch": [
    "sha256": "6343534a0be9420bfd24119411edb53d60d98017c8dfd026478e9a572c385059",
    "sha256" + debug: "63035895198070b66ec13f8361c9a0c14559fb8e4d4a629222192eb4a1ecf771",
  ],
  "kernels_custom": [
    "sha256": "b991a4d1fd91f5c8bf620b9eb82b43553250c7b4610e93a6eb79ca113378ae39",
    "sha256" + debug: "7bf69520bd6d46225a23839d439c110bac1e6c86509b77ccdc9ddc7ae0a7ae65",
  ],
  "kernels_optimized": [
    "sha256": "4a8afdfb8cceb7768bff121732cb78b332b832249f52e7df576cd668eccee9af",
    "sha256" + debug: "8a83118eb10addf2e53f89a15663378b6899250ca425ef40d0ddf89f67b27062",
  ],
  "kernels_portable": [
    "sha256": "008149f368b42fef2f7d457aa8c52d8e2b612264110df31976e248f23450e2a3",
    "sha256" + debug: "16d4f466fa498404fad05344e9a30355720085b2bf4b416fc4df398243858566",
  ],
  "kernels_quantized": [
    "sha256": "e104323ec13084706d87908aa9795e61146f1291ba102361e7318b6aeb65e5da",
    "sha256" + debug: "48991baa05d9635dc0a0950e0513dbb67b6776894c4a4a70f6bf7ac4a1c35289",
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
