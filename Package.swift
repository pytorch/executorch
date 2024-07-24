// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.3.0"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "2039098bdb348697e45821148e6fbf7b5c578144aa652251c47f1d358779a213",
    "sha256" + debug: "39334bcb506d76448627274bbb54f09398ba0dcc3fd47975b65049079a17f32e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6dbf545dd114aac457fa58176e0dcc3541576ab6809ecfa4d428b171c7b0c1b7",
    "sha256" + debug: "4eb47de7c58dd7d75ca9f6c924b5c506b8a17a6aff23d04173f806b476140faf",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "61bc7552d8152cad772c478bdaf3a083aa49d8f18ae8c4bbf56838b10ff7875d",
    "sha256" + debug: "e8395d10317e9df7f1737a71e39728286ecbfcd7830a8f275c8c6b236e2631da",
  ],
  "executorch": [
    "sha256": "c84ac159f80e5311012b00e38e23d9421c68d0e8c0b6ef71aa6ae41e79b11f87",
    "sha256" + debug: "e673753bc1cf5de4100d1fb4cf8358bc18fbf78c78e6e71e06c2f6432616554a",
  ],
  "kernels_custom": [
    "sha256": "0c3230db4261f53017541dc055d63f19e0793f78d149620f21af5ca812e589ff",
    "sha256" + debug: "668f9f2eeb7f05e0d74550befd4f5ab1de3bd97c2e860898e4c06f4f598d81c3",
  ],
  "kernels_optimized": [
    "sha256": "95925c74c954ba5b1488fa6324ce9391d1423747a9a6dac4a10886e780465683",
    "sha256" + debug: "2fe72c4faf2f39b65ab05ba11e9ad651b0408d5b6e1381f70e54ebfdb62aaa7e",
  ],
  "kernels_portable": [
    "sha256": "bc51846b9c1062d992b0300b9e98e82902374ea043fd7cfe608024f361a6a5c3",
    "sha256" + debug: "1662753e306eaef4a383c07bd4b2591c95d6dd13609f1fd1df451a9cdff2d536",
  ],
  "kernels_quantized": [
    "sha256": "beaa93416e8db577e376d56e4e4a4273a8278affec54e4d654d81545ce7c2b76",
    "sha256" + debug: "028e9324d172298edfbb71ec63b5343af192a5106663df75850ec0aa2b2fd811",
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
