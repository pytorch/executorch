// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.6.0"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "8e3b513b59a35d2624b5570b3a5f5b8e251b108a70321513e9ff809f2cd88866",
    "sha256" + debug: "e313fbf380b6ee0bb4dbe4f8343703884c84e32d61693a63db016da4c38983d8",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7b8c931ccd627239fbe7959ed8c24ef039081725b5344bc3eece0e403181f4dc",
    "sha256" + debug: "d6a26557d67a3232ef561bb69a3b369664b803fca010e2a3825bdbc87ccb7105",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c66c30450060153550857fd1c0d2360f88274ab04acf6f088cee4c86756c85f8",
    "sha256" + debug: "f973895d257360af86baaebfb872b55b22110e2c591463b42749ca27860b8f90",
  ],
  "executorch": [
    "sha256": "6215c9e6e4e228eed8934571d006cab255f1a844fe19d9e7bc3df06dc3ce6c7c",
    "sha256" + debug: "c4aa0abfcd45ded7a6d794a3a3343a283d8bf217dd9b4d93abb320e7668a44f9",
  ],
  "kernels_custom": [
    "sha256": "2bf2b773876a329ebea3e54672b39966d142b361965068c3609a9ac8c09f4b22",
    "sha256" + debug: "f8a4232564c3b1c47b922edf1bb8844746b03123dddf68d5410fb152ab4a2d91",
  ],
  "kernels_optimized": [
    "sha256": "131642d3bdc19cc994284092f44f5d2759da3c6505b05b3b6923b468e2f3daed",
    "sha256" + debug: "0708b211569d88e37cf5a9fdfd301574937b2789f2b8149dca1c76be54c15767",
  ],
  "kernels_portable": [
    "sha256": "d9119defdb25c4b0c5ff593e36dbc25ba77d33b6f90c7ca1faa89d15a6cd14e3",
    "sha256" + debug: "622884ee565ce5720b6ab503c1a986c44794fc84602e5a006e3f6914a18dfc91",
  ],
  "kernels_quantized": [
    "sha256": "e6d31c7999466248e5e55efa309cede0bec8f398a07a3d7f84c5e922a08a08bf",
    "sha256" + debug: "1f2afe89e4abefaf9126e087aceab9cd0416a444efd8addd4e862b5fdfa31c2d",
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
