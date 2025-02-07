// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250207"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "53cd91441b7114af5c492a189e155bdc9d16ddcca2cd6737bf45557dc5b40afe",
    "sha256" + debug: "e34d08bd71fd51ee32cc6706b049f87dd59a2945599f93c06eb9fe3eff60c3b6",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7ef65e11cd320268518894b19dfb1d1696b94728396828f27dc0691eeadbdd3a",
    "sha256" + debug: "e8ac322f30e79f51acef817b912d8e6e470093f5371a791c49495b5274db067b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "23263d98be32aabc43326b3f750e03cde5f801462d3e0b5694a4f3ecf8f88cc3",
    "sha256" + debug: "14b768acfdfd8549e309c1f497b8c671a5e90115c0b9f450a6c19a961b7f2b25",
  ],
  "executorch": [
    "sha256": "c10d0d66cec9289b3dfbc433f8aebd421f22331ec99c0d095f815f3ff0021648",
    "sha256" + debug: "0131ab04a145bb84297b15d10a027d06aeac72b0158e8db797eef21a7ddaa463",
  ],
  "kernels_custom": [
    "sha256": "97bd9f13180f59213309af4ae932613d61237f107532a0f71e01a6f700d3c4f2",
    "sha256" + debug: "45a9e4051f3f397e8564ae37fa891dd0608bc73d73f16689fb8d7b0418a38a0c",
  ],
  "kernels_optimized": [
    "sha256": "e2ecc558b8dc0281621baad494fea9a3b81f8ad57094d780e9570653b6a12de0",
    "sha256" + debug: "f2325f7be02cb4ce10d55026035b8abf6b16a36b800b0b44b6ac0bed58a27e96",
  ],
  "kernels_portable": [
    "sha256": "047e56603996265b7caabf3e7cb48877757069d5dc7151cc83e31ac933c61ece",
    "sha256" + debug: "d2bfd8472f322542eb360e58fffa311e65db144a820d45ecfa9aa78ce33cf658",
  ],
  "kernels_quantized": [
    "sha256": "25cfdb06561082a970e1331d8f9bfd2def8b66e58f825525d21d6e6a0c629fba",
    "sha256" + debug: "d4b64a9891417b877f142e479eb13f129c017fce9d28fd14465ffee09494f304",
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
