// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250306"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "93ae0990ce41e07983bb3177f208fde951ec439d51eb9e684541ff926a34bad9",
    "sha256" + debug: "b99276da2b35520a8ef2618efada69b886c1d3a7aef65270074027ca8effbac9",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8d4eb02f35fd049508131b4de9e6f30fa0aa6e3eb046e6ab9f57f361c9085514",
    "sha256" + debug: "c53fc33841e042c33960b197defa026426faef995e1a9a0f7d132559a2bc0116",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "29736956411c1913c3648afdcd1304592ce386a0e6cc0bc51a596ee9268cfa91",
    "sha256" + debug: "180315f74fe0e143eafa384d5e5c79b9074d124a35eebb169715c436c821f5c3",
  ],
  "executorch": [
    "sha256": "5357233e53187c57505e5fa27b252cd0d222084b124c7d55e702183e1dd360de",
    "sha256" + debug: "85e29e690daaf8df84145966ba667ece79c3da3798738b0aff6494ce62975c8c",
  ],
  "kernels_custom": [
    "sha256": "f2d4f60ba1587e5e6e25cf9f91abcce90057ec1be758b9c463e93ab6d179e16a",
    "sha256" + debug: "9d43c1acdbd3bd930476b7cd78e30c2e285b2304ab936551a22410e7aba35b31",
  ],
  "kernels_optimized": [
    "sha256": "f9ce313c7c5bc203773feffc51e5c850d21f0b9465cb59899483b18415713599",
    "sha256" + debug: "db547e24c5bbc1aca250f1f7dd5aa3f5603d8990d53c103f3b22fa4b26bb6a0c",
  ],
  "kernels_portable": [
    "sha256": "4f14ca0bbb7d8d79ae53f5974add90323bad664186988a11e709434488d149d9",
    "sha256" + debug: "586a8726dc76b9736ea473ef35cd39bbf535f7079e422c91caddfd89d66be3b3",
  ],
  "kernels_quantized": [
    "sha256": "71f1ddf4c7dfea782523698208a35feb80dd592f88e60a6ac51ece9fa0805003",
    "sha256" + debug: "a6a2e5863c331a0f34a46d91120b6713c4daa2281e92bd38c5b45e7baa6995ac",
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
