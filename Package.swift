// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.1.0"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "coreml_backend": [
    "sha256": "5bfa35cb5143b4af6840e0e5dd2d40bce93dff331b8eb5798a46274239391a5d",
    "sha256" + debug: "1422019da9000f8ff7be597de9e0e3b2482f99cdaa75c2d179835778647be1a6",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "custom_backend": [
    "sha256": "2201a61eaf7e06e1937cb73a469fb36cabc219496ba004b85feb2cc7c10f300d",
    "sha256" + debug: "3eb6eb97bf0641d2305b0f50ff05a8862d7d65e2491cf4aa05ef1d108649f07c",
  ],
  "executorch": [
    "sha256": "2b55cbcff845ab9eaf16a21e520546b2975ef8c55b9e3fbbcc0c375334e40c6f",
    "sha256" + debug: "12933cedff6cf21c9d21668779f8d8af8049646fe7d290787b12227ff7abe4a7",
  ],
  "mps_backend": [
    "sha256": "510d708361b6ea0692ce5aeb638725d6275824b37bbe744aa876fda24cc2bbbf",
    "sha256" + debug: "6a67ba0bf8033f17bd66acb222446df51cd1304e24a4fb2c6d97e15a30fb24f0",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "optimized_backend": [
    "sha256": "50aaa54901a7cee1059e71cc623f054610406d65ba8fd6edb10b45861be67237",
    "sha256" + debug: "3f43f465727c8705432f4bb69260cc9501c519e5da006fc19ee2ab2ea260d1f0",
  ],
  "portable_backend": [
    "sha256": "964238e92828665aa598c05b2264faab91fb13ce0f42633cc7d5653300af3e9b",
    "sha256" + debug: "d6d85304a4b40f13c9b893e8c264ebdb15307cacf8997494b3818a52e4914b28",
  ],
  "quantized_backend": [
    "sha256": "37d31a319f92e26bab2b7ec5e783a8b14457dee0a4638dcdca1d9e17539ee3fb",
    "sha256" + debug: "6b45f66f60f6106a41e191418c970bf7b0605df73b9815a06441a5f0809b54e6",
  ],
  "xnnpack_backend": [
    "sha256": "03d506243c392e872519ae1335a025ef202319c1db339a753f9d7d74cba226f0",
    "sha256" + debug: "3341e89abc99552a6a5bad360003baed194a83e865338bc07afe9e4f171ea169",
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
