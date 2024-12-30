// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241230"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "925b4254228f830b8920ac1b0014afb2c00b78864b3692453296a7f44ab49d52",
    "sha256" + debug: "c36095771a8ef95f3bcf8548509b773ecefe7ce64ec39cccbb814ae7945ee324",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "15db28df48b6ada9d58bde3269d07fab8eebb67db89933fb6819b6a213616785",
    "sha256" + debug: "f3374d1942af23de282988b781b5a0b249f6c4e5f027020d5190441a8f29e79b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0fa99b45f1c1953674e73daf6a43567e9e14dea5d378722c209b06ddaf55e398",
    "sha256" + debug: "33d4c6ad988116aa7983a387cd738c656ebdc15397acba8f3eeb71a634ef3f84",
  ],
  "executorch": [
    "sha256": "10fa8f85200d0dd7cd1be9a906fd119701aca4fe978a1355862dae1bb2c4d671",
    "sha256" + debug: "01721f58a768ca797d3a69c685d9fa14cc9e35dd438d08e86dc3fbcb076d087b",
  ],
  "kernels_custom": [
    "sha256": "5313422067e4c0f743ba96f47ce8c736dde09876e6eb85d777300928663f71f5",
    "sha256" + debug: "00721c43fa0f1835ee148ca59984ee762ae128d7da60aabfb172ba48c61438c1",
  ],
  "kernels_optimized": [
    "sha256": "a50118f7062aef904e0a158367d7f5241e466fb2173385909ccab2bc754ef12b",
    "sha256" + debug: "32ba1d5a1e1496ca115661b9984e6604abeba57f2eebcd4c3116c37a47f548f6",
  ],
  "kernels_portable": [
    "sha256": "f6294624278f30a0f2ae120e29929706237a7bef4162fac8428bc476c57c45b8",
    "sha256" + debug: "1446151ad4f85e18698a2881f2463bd91015023d92a1167c3b3aef3ae5fd9a85",
  ],
  "kernels_quantized": [
    "sha256": "76da576d558fe22c3a8de2f4c595c1fdb2cf96073072ceb93e658fb840730f8a",
    "sha256" + debug: "027073fe5da7d484b3e369dff4e4e3464b61efd3455bad2c877177fe6ae67c1e",
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
