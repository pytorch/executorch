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
    "sha256": "6383ded41b4e631b6b2a3ede67acedaed7106a5216a2ef9240e75ca97f3127ac",
    "sha256" + debug: "9c8100e23fd75df12c344b0cb22b8acfc84fcb697a1814a6778bcd1bce4d47da",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "21a229fcde9f831dfabb32be8d2cc4a626cc9b5ba68a7f2b34a6e0d13c608d21",
    "sha256" + debug: "0160975fa22220cf46c9d36a880280394dae085188a259188ad70adb4df79a84",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "17af715e8f5bb5dfc244d17b68f9ae450719058f72ebbad9fff91eed52a74918",
    "sha256" + debug: "3827e3e1cb678171c3471e0ef59fa245506aaa892962c38993a69337ad27314c",
  ],
  "executorch": [
    "sha256": "b7db803ae6c4c904a713aaba5e3afce728e70d1107477560abca5ef89efe9304",
    "sha256" + debug: "45c0ce0d75417bfe0932a409d96c12e95541873018ee0b856e8d920649cfe2df",
  ],
  "kernels_custom": [
    "sha256": "1b5f278dfbc549dfc91ceb713061185a59e824bafec3a8c7727921af46472a4d",
    "sha256" + debug: "04d6a5ea3c5fbaab24186f39f8a258cc431a3114776e091345b9b91e89fd6653",
  ],
  "kernels_optimized": [
    "sha256": "15d5900d8bc17bf6cead8d1471c85f8e0bcf921ce389245b990e4eca15bbe350",
    "sha256" + debug: "43ef322b7bb42f917ba3133952fa358fe8638039a25e9528533ff59b2010873f",
  ],
  "kernels_portable": [
    "sha256": "6b3e714d28916206d7800c63eeffc20fb38a95a9e36bfb9174b58fbb975ac75b",
    "sha256" + debug: "b6343dc7b729e8b05925c3584a6078e98b706c47c66a45e2ffb0f722419814b5",
  ],
  "kernels_quantized": [
    "sha256": "3bb26944442413d193e5ccbf444f876193be57f2d672af928d2fb001c4ed2876",
    "sha256" + debug: "e4922ac6b050e98f3f67f959b67b0a5333597180e70928ad8908170107190662",
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
