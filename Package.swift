// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "latest"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "fb695081f33cfd5b30726fbd69bf88f871d6a3681def67f44eddb67f8d18de7f",
    "sha256" + debug: "b56dbfe24faf70f78bc5b06e4829e729ae932d6ee004b5ef25c71fe34a5b5f13",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "79755e7ac367d4e2ad5c4a2f7ea651c67eaed9fdd3d980801609ab2325a275d4",
    "sha256" + debug: "717a3714fd2d76d16f6ff362556946714fe7b6dc8c4bd1be8614152324a317e8",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "34478146d22b7f3666a496a4a23bd523f52558ca3da259b7865d50ea709864ec",
    "sha256" + debug: "51f6c0a08417233274cd53bd18253b63ee2d3f4f26205ffa82afd069610f6d2d",
  ],
  "executorch": [
    "sha256": "2e9ca59757be19f2e190e71f8db9eac6fa552732c8f5b9815f213387021b5a92",
    "sha256" + debug: "85c2cad98c4d35849fbbafdfae8838786a60ee501a07341895d8317b1c511b61",
  ],
  "kernels_custom": [
    "sha256": "ae9cda49a6ebed30d4cee267291f1923d556c585f5477b24aacded752cf6e589",
    "sha256" + debug: "142a92c8163522cfad942f6b0be12a6b446ee512ef1aae1fbba62d70efae44a6",
  ],
  "kernels_optimized": [
    "sha256": "83abe352ff0254b96b73a75481a4381e3cf504423af9af3e81110c2823899799",
    "sha256" + debug: "18184c3365c35642572ed2d976422f4b4775c5cfa4952869f4dc8a0f557c8365",
  ],
  "kernels_portable": [
    "sha256": "057d80f12544a45b33b8e518d55682ffe3504fd44de4a77bb01207d98844f3c8",
    "sha256" + debug: "a890e0c607652fcb34bfc0e0a2f265e5a1007d32203e044a2d122fb763cfb732",
  ],
  "kernels_quantized": [
    "sha256": "707bf51c6de2a19722fa82d417b2c00c81251bbee06edb2e94f22d1e52bb7685",
    "sha256" + debug: "e078180f52aa13c6ed0959cab646bdae8336db4c7f24b46caea2b68efc5b36cc",
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
