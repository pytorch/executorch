// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250223"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "4f0f99679e8cb00259c4816c0e62cbce14644c523b4e11b23a4381e8f6db3d70",
    "sha256" + debug: "10e7359d69b775cec5392b7a1baa565198f89c5ba1499b52c6a1050e9a241095",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "c50ccca6f16bbae0d13d887c56ad7c50a8c5c20fec7f60291c88e52fb6ef4172",
    "sha256" + debug: "cf2912bc501af2f787a697402ba7e81ddbedecfdc13a98140fed81920c3cbf54",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1e2703c9a307cb7fb9d8c7950f66070741f3eed56ed2d32cf51e1bcc7acf9b27",
    "sha256" + debug: "d99568805c5e03394e7d0febbff5ea65d9f4c4425452076ce6545653304c27de",
  ],
  "executorch": [
    "sha256": "c4f52945437d7f393431d0eeed14db10552a93437c4ff6c37c30111bf0acad4a",
    "sha256" + debug: "3414b6c9db23b4d40d34f03f1b80693b15ad500f88523759998ee83d4157be9b",
  ],
  "kernels_custom": [
    "sha256": "6eea37d11b2ac5e60dfabb9fd0d827c0f1d06bacfdf3526f432112374e8ac942",
    "sha256" + debug: "32b1249bc83c87c777b1d849c81156525c6e932af4036e7793a5e8326531b00d",
  ],
  "kernels_optimized": [
    "sha256": "636f5523ae678ac2de6695cf27479ffc5e41a6c7dfda478989b3917e3f189910",
    "sha256" + debug: "c0819301b701ae983d890a3faa6c6bc94aa86320a88f3f376888137580cdbc62",
  ],
  "kernels_portable": [
    "sha256": "91155d06d9cadb1c3799bb57b1783a7049f06d4d387d298adb3a818114a93b43",
    "sha256" + debug: "5013dcdcf583cef1ad916eec67b1cd5d7ba96f3648c9f858689536dfe66b384f",
  ],
  "kernels_quantized": [
    "sha256": "2cba14fc13fcbb2398efc6b84293eea1cb2dcf91f9eea488fa14bcadaf7ab511",
    "sha256" + debug: "b153d340539860243435f71d6912175410ca9c3363318de5751123043564a7f5",
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
