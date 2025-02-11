// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250211"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "794b43f3ced250a07f9e73d7a4b1403be5091858d507a625ac4f29c02d9f5fc4",
    "sha256" + debug: "ca2ba01669afaf1789206a5439fbc56e9f260da657c4414345e8604422fb6d6c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4d69cee0939a3026bc44b6d52690e99cc3cf1da70c9e9108ba2977d3b21d4747",
    "sha256" + debug: "52df77900c0d11281aaa2211c61f2febb5f9720b1684d82a764a9ac0e27511b6",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "fae4d169a94f4163d6d16e2a4d8dfef43566a60b1539395e364b794d2b4cc6ff",
    "sha256" + debug: "53ee8dc6485af315271f3a77b5acf68322eef9dcc2d58b403912a1587cf542a4",
  ],
  "executorch": [
    "sha256": "677cf8e5f3aadf0ee99746d68ecf12fe4b2c63c98feeba4bd160f242d599d00d",
    "sha256" + debug: "8900d47a6dc6c54d18b8ad0606464dc59f5fed8ad0328706933e41e9ccbe3348",
  ],
  "kernels_custom": [
    "sha256": "970668661831a652e4a0d7e2a3030515e076c11c522a6c3f375aaa2b3683b478",
    "sha256" + debug: "aa6dca5050bec0df3885f2b171608ccf92259f5b0cdf145452c1652dd91f64b7",
  ],
  "kernels_optimized": [
    "sha256": "1c68edd104007995ea7b86e7e074b7c3cb84a46ae4c1dbf6e9405ae5d6fe7053",
    "sha256" + debug: "3137195d6708608d4607800bbe63152be9889141b877293d88c8e820581acaa7",
  ],
  "kernels_portable": [
    "sha256": "85e54e69b03f95d247ea100e39fef72823e79389aaaecf0460cc848107a1f7e6",
    "sha256" + debug: "6264a72ba27983f6248c028eb2daf0f8dbe7cc19b1bcaaf7315de25772be8dc0",
  ],
  "kernels_quantized": [
    "sha256": "e24d1259d1c6721d9f9c2374ac4bc905636405e38a3f7ae010290e8b0e75fa0d",
    "sha256" + debug: "e3ca93ec93ee8a62bee6105b1d08e01c21ea2e12ce729517fcf5ed8412365acf",
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
