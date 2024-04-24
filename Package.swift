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
  "coreml_backend": [
    "sha256": "f966e06f3c6d62c97437c199824590d63f820e53f06da8373a12fb6e1692e145",
    "sha256" + debug: "bd8a94239d67bf954f7bd231cbd4bce0b44cabcdad28f45aa0e53313c2c0abd7",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "custom_backend": [
    "sha256": "db4913c87c5d955167fc4e5caa4084be42b168fed86edca2940e1863d43a85c2",
    "sha256" + debug: "bd14d647e145c0a7999756d319b44d910257527cc466f08507093eb18304a5d9",
  ],
  "executorch": [
    "sha256": "06359a41d0acea630c58c0906f938d4aa239c7a96182801052e5a60045f93604",
    "sha256" + debug: "ca536d460fac2a8dcf193bac50a2c3a2d231296235a02af0c965a860c03f50f3",
  ],
  "mps_backend": [
    "sha256": "e783a9f9fb4632b2bb4bc8d330ba1e59229f9072f62e608e1a79060c815890f9",
    "sha256" + debug: "c6b0b83d281bc53dccc4dffb49617470179b5a051f25d156bdeaee68bf8e4257",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "optimized_backend": [
    "sha256": "4c3c8e6c77c5430662374bbbc615e64d982568ad2dea7b3cb03311ec856ebe9a",
    "sha256" + debug: "f41e47036eedad2e20f60b3a9e7366d5d2b6d982ffa0088fea8baf16db54f2ef",
  ],
  "portable_backend": [
    "sha256": "a67463a50ef2e1921a7375eb4d6e4fd9344cb28d8e18a15219520f74253b6219",
    "sha256" + debug: "292902ae0ded56193d94172360b3f37fdc9dd402b0064dda03a46d2572a2d480",
  ],
  "quantized_backend": [
    "sha256": "dc47b8bf5ac68a96e4ef1f809cdf15d3a1987ad73052efc2881d96adcaf8f4a0",
    "sha256" + debug: "22e6f8039168f481a6a724797c817a976932478cdff7466ff131b8c962707131",
  ],
  "xnnpack_backend": [
    "sha256": "30a1942e9867578cfdc8d877bad1e8170ea82c6eb36d343ecf7303bf47de381c",
    "sha256" + debug: "98b2d0fd582f9822c8c2eb9162022c46984dfccf15da71b0518a906516fbf1b7",
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
