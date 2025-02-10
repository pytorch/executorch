// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250210"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "b9231d8ebaddc2cb0e9eb911d16d3aa2cfb57531b2e3a0287f028d3920c653e8",
    "sha256" + debug: "e2efa1d3cf6bafc499572a73dde967dd29bffff72a42e2539f84d5bbd8a196ab",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "94f8b827ec8caa2bacdb3f26ec056863bf92f98f450dad0e91ee046b87032a94",
    "sha256" + debug: "d42fad77f0818a83c63fdc846a9077cca55bdd52f706f37da482ee115d45cb50",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "62cb78fa04844b47af511d89ee7b9de79c0f5983e692d16d466d622005725ac0",
    "sha256" + debug: "111793dcf11ab3c192008d0ee88f5aa9bcb899ea97587abb4432b7312d8c7843",
  ],
  "executorch": [
    "sha256": "eca0507955470226535ab6e266c4c73023fc81ea54bd75ffc6bb8bb5704cfc66",
    "sha256" + debug: "787ecb72dd2dad9c0f3f16ee546cd5618836673a34d67e64a90392ee67de1ffd",
  ],
  "kernels_custom": [
    "sha256": "837d869731fc00e8a452e3a4f75d6bd8754df0f86fcda15f6370725a0f212cdc",
    "sha256" + debug: "2e3a1062c27c8a0421d3e59d16d19de6b5a8d4e2550fd76688b36fcf6941eca2",
  ],
  "kernels_optimized": [
    "sha256": "d2868ad8bfbb214e082d6b14090c75a86efcd0ac35a507885dc5707c651d3045",
    "sha256" + debug: "88dd00c9b80ef654b7cb91e70f5617e458c6b6a705c23ec3b94ac3b8149acc0c",
  ],
  "kernels_portable": [
    "sha256": "49b00eb5a3310ffa6c8a8ae87649ca0d68dc76a69aaa94fde8443e33feca8181",
    "sha256" + debug: "a895f36fcbb8654981b169710b832da52ad277d5e949688a408259b9a75d0cb2",
  ],
  "kernels_quantized": [
    "sha256": "a60b5571e79d7c91dd2bc7706b6afe1fd4cc9b05d8062063191d3678ed69a566",
    "sha256" + debug: "a5ab1d150e38f0aee40d029795dc9daf8040e37ce0e98e539357a16640589c71",
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
