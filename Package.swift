// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241228"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "c1af67ced5429c7cdc9e9159b2a60c275b79147a074cc69958cc7cd15fd9428a",
    "sha256" + debug: "85a6c1e26ea8655414604c7e8d46371a45353bd8f2b6f0521fafc5c24358ca09",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0501b76ea8ef1073b185f7e1920c7e310184cce10b94fec82181d8dc202e3347",
    "sha256" + debug: "ce867e82ec34163e984da035b9c43fa1a0d6d6239a72aa0d6ed43a497f572332",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7fceda2f7cb9b86b4dff9c388559f251287f6bb3686e45592b8161e9b81a05d7",
    "sha256" + debug: "a94477a8492bafc8b695173adf8ec020f1fd32dd4c6fe5e3e4f0878d757e8468",
  ],
  "executorch": [
    "sha256": "d921f87e6316e1fc5d640091d05df2b8a355c4e10c679e35e15890077a5aefc6",
    "sha256" + debug: "3bf60a8730120c365f4b25d6289b92bc19b2485666a4734a4cd90c1faf8dd9cd",
  ],
  "kernels_custom": [
    "sha256": "dfae59701260d75b6820b03df013317cb3dc77ed27ad86561b1aa87d319605b4",
    "sha256" + debug: "26c347a27d33d58202655b9ba88a49abd7332b9da5066388f51e559d6c7df2e9",
  ],
  "kernels_optimized": [
    "sha256": "b8e0174cd661807a63de421df80c438322d3c4e998ebe526b0ff6094550bdfd2",
    "sha256" + debug: "9833a24bbccf82a55e49a8aaea28409d63a8e5ed2f493827cd4b92be8bff5f02",
  ],
  "kernels_portable": [
    "sha256": "4067b49aa16193f2aab52c8a64306a3630171a3554638b1766b2d110f79ebede",
    "sha256" + debug: "8557b8dad6395999aa4c7ff922231db79a5f0bd2313425e7fe4a11f04a367c6f",
  ],
  "kernels_quantized": [
    "sha256": "9bebb7a8690d5b586564530aa0115b4b60f8d41c4cde3d5ed0da77071ecc4ba0",
    "sha256" + debug: "a9194902ccb7a57ca57413b06409954c3d8808a73e6c05f864c5c1c9a8b57e5e",
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
