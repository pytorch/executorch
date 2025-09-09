// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250908"
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
    "sha256": "d8ccb38a907b889b5148cad78b7f64e750172475fb68512dc147d4045b18255c",
    "sha256" + debug_suffix: "ec70a8b5dc29e058138e4b41a1b28d9014a60ebf436784810462af8b6797be9a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "23b5903098c6f0d432f3bc46f18edf5db9819bde8e85266d7087e8c2f4883c96",
    "sha256" + debug_suffix: "a56d77f16ac9255e168cc4d70db2572cc9ba6478ed4cdf9d248631529c50093e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "840c91a6c7569370f26ef841ad587089b277790556216a42c92211785180ac05",
    "sha256" + debug_suffix: "1c76c827b66838e0793d652cbca64cc376070b6210a4ad84ed0f754a363f8b25",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "a473ad8ea5ef132e1e38a28f14526d3ef192c95858f11689a704f4e249579177",
    "sha256" + debug_suffix: "7af0f0f8f182b64ca2929f673ea034987af22af6fa41e21a7c3621d8a03dc55a",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "0ed094f9e6e79fad89cad856900f3d886235b17253ea7447741fb808f95e0df9",
    "sha256" + debug_suffix: "0db58ec53938d6d43505eec568c3eb0c24e0621ce90c54e7021e1027a57daaf3",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "fbb0f89eb787d72df8c07ab532eba6d06882cdcbd2bdbfa429970412cac66b21",
    "sha256" + debug_suffix: "44709537c453395b66bd3928740a07ac561098e032b158c3d7936965477ee495",
  ],
  "kernels_optimized": [
    "sha256": "b0d08d78d5003ecbb3f37156d4d151d5089b69e6b3c7353190e17f0b0c5fc27c",
    "sha256" + debug_suffix: "6c90ba36d151a61b32a97eecc46bd2c225b089db066f907c12bbabdf83608e35",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "7b9a9af833302c8e4afcb7e42307a5926649dad905306913ffa7cdc23702a1fd",
    "sha256" + debug_suffix: "561b00d28728e4b96efccff944cf85b6433d436e5cb8a178ad35cea603ae71e5",
  ],
  "kernels_torchao": [
    "sha256": "9f7eb8683086ff2c885753660609e08bbfc94ad6ca5d073e4c9a7b37b41dac3c",
    "sha256" + debug_suffix: "5c0df9854606e70369f22d182564f46d0addca9b93605293ede10eabee1d1262",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "77c33bc4bbea30fddd98f1585a2b1f2507a8f02373bf4524606b17f4baa89bb5",
    "sha256" + debug_suffix: "eb9c784569092888fcabedcdd631397bde0dbabafe2ef4e4bb78979c6b63f973",
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
