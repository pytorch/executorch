// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241225"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "b8cbfee88c122bb2040ef55ee6b2356d827567aa1d982abdd43b68dacecdaaec",
    "sha256" + debug: "dccc1414894fd94c636006f127b2583f00a4f7e1c7698504a66e7c3a6d7623cd",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "09991efaced9b1bd9245f0fb917e588bae5e2640be1c43df0c6c1857cb509222",
    "sha256" + debug: "e68ba6c7157da0eaea361ce7647baedae979a52898d2c161aacb96c8fc9b48a6",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "60d5f379c639bb5e30a4aa168e833dd0e7b1fefa38702cabcd38ddc8a3ebb746",
    "sha256" + debug: "e7dc77fd798a2b9340c5c354c36b3e91f29d1acceea0b8f02a12cd6b84e336ed",
  ],
  "executorch": [
    "sha256": "d3c425549eb6659c91d6b8ef63cf0d636dadd096df27035ac97ec4ab0a599efc",
    "sha256" + debug: "b1e6a21c22983a4b9c10510f67b8d9f691e38839689824fb3cd8976ef57dba70",
  ],
  "kernels_custom": [
    "sha256": "dadd8924e12aaaa9c8f8a553885e1c6b2ca4bb77316fe88bf90c3d48b8564b12",
    "sha256" + debug: "acec97ac869f1e23afa39db106222a9c46f504098b27500e0123a2fb2d0ad4bd",
  ],
  "kernels_optimized": [
    "sha256": "71a128858323a85b23cf417accf35229258416aea9607372cb05f00f7d7f5b49",
    "sha256" + debug: "0350961af54c7aaaa65fb9cff2014704826b0106a8821fc599add77aba1b1ffc",
  ],
  "kernels_portable": [
    "sha256": "fd8f693eac20bedf3c64dbc4a1a15bc94a53fae60d557e8529c7dd79c6864090",
    "sha256" + debug: "e1340c02d31c51093304c34ace45bc589fb68120b093cd02d1d95c1ffaafecc1",
  ],
  "kernels_quantized": [
    "sha256": "b957367d35085b770e3f5c3538ef45f645d6ba24cfa041a66259296d6adeb21b",
    "sha256" + debug: "e7657a05822d91ae8032684a30f3ab4798bb1b919101695be0f2b82c3b2c98fe",
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
