// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260604"
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
    "sha256": "dd754e1ed03588eb18187d2680b9372029b884544aecdf0b3520e49164f2899f",
    "sha256" + debug_suffix: "2149f469e5b36ee69e19e84d034c9483b7c33d2355ef22c28df092f1acd0775a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "fabfa05e1fa2b97fa66d1417a5068093026b13e064370adf50d3b966d2dfc5e7",
    "sha256" + debug_suffix: "4bc850b203860acf71d80f2f7fbfb8b2390eee7226262950b8298754f0f81d7c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "983ff5187a3fc025cf056ca9c5f19cdef8cf89b12fa4ba622ada1f938339ff15",
    "sha256" + debug_suffix: "b2dfe3fc73260df70c39b74e7df98700180eb2f8f607ee7371b63c98cc693b39",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "c5a7ecdcdd5094add1a4d01d8b53a6205a743c1e0ad7a559ed9c7c005dee9dd1",
    "sha256" + debug_suffix: "770eb0f9cc0e68bdb7d287f713ac1eb8cded8e7873e21e3b29e41ebf531b90a9",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "6997682ec10df62046532b039ba06b5841e0ea532d0510c06ea3fc9c4c885473",
    "sha256" + debug_suffix: "31fd22bf1806f7ff99df95932de908047c553c16ed15cfde28d0b4f4206e8d41",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "13bca55d70a066090fcce225b08c260f4fd04828bfad34d5bbeb0b3b4fe36945",
    "sha256" + debug_suffix: "21bbfe33949ee9a17c752ccc74eb64c3f150f3c28778860b1e9954055b85f703",
  ],
  "kernels_optimized": [
    "sha256": "07e77f8f986df4ec41996fed1f3af4c0e679cb5cf799f9b12466f151b2a51df1",
    "sha256" + debug_suffix: "1dabe6adf126fd44ee71f77f772e14941a8a90797c2e5c4e6e982548a7ad67e2",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a9afa91f71dad46f388379451908cd0ea945172cf2b2a02484b9d293a697dfb6",
    "sha256" + debug_suffix: "1b4a35b7c6bcae8a7343f27cf05794f24cd4237bac081619c7837c2080c887dc",
  ],
  "kernels_torchao": [
    "sha256": "2ff55565dc9a48d3802b340f9e0bf1961d1a34e2ef85c49d9e554fd30ced4bc0",
    "sha256" + debug_suffix: "b4e7dab15d46dd531748d36b2fa3218a2559a708f79f62b16d0759287be377f0",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "f7262f3da923b7cef29167090c5384a013655da58d8da6fe7bda290b925ddecf",
    "sha256" + debug_suffix: "7ae270ef07ff22b965307d386b0c7d4bd84570df404623f4e43d39ed54e0eaa0",
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
