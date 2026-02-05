// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260205"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug_suffix = "_debug"
let dependencies_suffix = "_with_dependencies"

func deliverables(_ dict: [String: [String: Any]]) -> [String: [String: Any]] {
  dict
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      result[key] = value
      result[key + debug_suffix] = value
    }
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      var newValue = value
      if key.hasSuffix(debug_suffix) {
        for (k, v) in value where k.hasSuffix(debug_suffix) {
          let trimmed = String(k.dropLast(debug_suffix.count))
          newValue[trimmed] = v
        }
      }
      result[key] = newValue.filter { !$0.key.hasSuffix(debug_suffix) }
    }
}

let products = deliverables([
  "backend_coreml": [
    "sha256": "01f0f1e019a49e8bf977aa0bf03cf19c931c0e09802f479a22fa33540add88e3",
    "sha256" + debug_suffix: "c9d925f0aa0348250fb68dd2faea01bc97e2a260daea5fa47c447dc43a358481",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "63738fc6940ceb0441478c2146785ed15c8e79e149657532dd04bc4536e825ca",
    "sha256" + debug_suffix: "1e8dde4a2b7eee44b5eb1b9909e27779cf851283aeb36479d8d1d5cb47674947",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "23e606cf4cd60f558c398597ef0917ad41e4d46c2ad292076e8768a1d2c882f7",
    "sha256" + debug_suffix: "18a05e2b4a7b736d119360172efb5537e657fa4dce7f4072abc937143a360e3a",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "3da1eb4defbc76c96b4f59f4c526c97afe2501d77cd85b930c66d0f21e88db2e",
    "sha256" + debug_suffix: "acc9f5f67b1b1f3856a7d8069dc8d207065d55fcad6b6b8c6c16ee67e9082658",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "51b09fc7cf0f990db8bd15a751d7fa2cd2b86b449f0531ef6302aedbcf432605",
    "sha256" + debug_suffix: "f5939e9cc751b624ea557bfa5de81ad3e8497c3a7b5157322028826f1991ff00",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "5c9b6fa6a1ab32c2e7c8f27c96aaefc465e8f992d643fdd876832c2ff6a5e804",
    "sha256" + debug_suffix: "e87fbc3f86c3522453119ecc17d0ed3c7a5e6bad31b3bfc0a11fe5b19dff825d",
  ],
  "kernels_optimized": [
    "sha256": "c7f8ff65b9833922d3c54dac082ebd3df06a8315bdc2298cb3ad80aa24b367c3",
    "sha256" + debug_suffix: "ea3770277a5fb682b09f9057061bfe7e2882844cbc6f079becc80700c394d17f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ef1d54c85545514cd3a232663d3eff3e966110449faa10fafd114aedfa0e28a2",
    "sha256" + debug_suffix: "1e5c2ed0650a41ac686a2fa6bf4df3083831393e949143c36ae4d6b0b901240e",
  ],
  "kernels_torchao": [
    "sha256": "402de2bffe67c532ae4e83c92ed34a70880a35a3dde8a7adf1b7a16f8e09f8ee",
    "sha256" + debug_suffix: "de155646def8a3ba0fa3f581aa045eb156576926457f711299fb4ad270fd70ef",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "552b8f57afc762ba7372eb374180a12f4c9d9e6a3b6c19c2d27693c1e7094ebe",
    "sha256" + debug_suffix: "23b1ed82213c9e1e8071e9d0d2c12d602fcbc6687b9558b50bc8c8b19ac19c10",
  ],
])

let packageProducts: [Product] = products.keys.map { key -> Product in
  .library(name: key, targets: ["\(key)\(dependencies_suffix)"])
}.sorted { $0.name < $1.name }

var packageTargets: [Target] = []

for (key, value) in targets {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
}

for (key, value) in products {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
  let target: Target = .target(
    name: "\(key)\(dependencies_suffix)",
    dependencies: ([key] + (value["targets"] as? [String] ?? []).map {
      key.hasSuffix(debug_suffix) ? $0 + debug_suffix : $0
    }).map { .target(name: $0) },
    path: ".Package.swift/\(key)",
    linkerSettings:
      (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
      (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
  )
  packageTargets.append(target)
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v12),
  ],
  products: packageProducts,
  targets: packageTargets
)
