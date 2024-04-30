// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.1.0"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "coreml_backend": [
    "sha256": "6d3759397a6d1294a1ed8e6c8cd35a90516ce65e358b62442fad981f3ac97b7e",
    "sha256" + debug: "fa544b1bf38e9eda9ef0212bca43859be433088f460beadaeaab7d655d9b7044",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "custom_backend": [
    "sha256": "63ecf4af8bd1d13e715aeebd29dc8713b2a5b81187c977771ed95cb5df0c8eb8",
    "sha256" + debug: "5bbac9f5e108d014664f2e3ab46188987de3c8fff414721ea586f33d67d862b5",
  ],
  "executorch": [
    "sha256": "3375eb7df570406a96f51a2df3f42104c47e42f26c011e36f60f199307c44c28",
    "sha256" + debug: "8c8bc839ca1fc1591746386cbeede6af2843c9d11d83c035bcb885310277eb29",
  ],
  "mps_backend": [
    "sha256": "7469f2a17f2636a165e40d5355069170b8911fda51a65f6686a25d5863c4d7af",
    "sha256" + debug: "af54887a94919f1c8a75364f86c5b7ccfd15dc1cf941148451fc8b59fc16d726",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "optimized_backend": [
    "sha256": "44f8e8cb5434f2276189ab6f463b96196622bae7162e28d969c009147139f360",
    "sha256" + debug: "932d3979ec25fbb4c11a777e5aac9a01376fc43e1bb79cb26a61cfc38f0643d3",
  ],
  "portable_backend": [
    "sha256": "529adcf4fb42d376a93bf902855be524d28a9478acab09245003f9d6942cb6b6",
    "sha256" + debug: "9c0abb8e83cb25e2b429aa275ed34a4004a36cadfdd82fe2654a65d69a555656",
  ],
  "quantized_backend": [
    "sha256": "9cc4ddb16c86807f139ff969fd3b08242851dcfde78431d436884ff6dd80ca0b",
    "sha256" + debug: "2362a6f1c7ffa2c2730ffa48a167f11a480cb430f7d6a99fe98b53fe389bdc4f",
  ],
  "xnnpack_backend": [
    "sha256": "6349c94f0aa225d7421cb3ba4a169c9a8abef27e3a50f3bf0e08d24ce8695ab5",
    "sha256" + debug: "5778635bbc8c8f2808f1629fbe3d56ed9db97990353d796f42998daffe754662",
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