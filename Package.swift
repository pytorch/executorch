// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260509"
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
    "sha256": "4d19a809fbffa5dad89567fb023437ee8b386db14b8ba7ee56b492e52a6c2990",
    "sha256" + debug_suffix: "113e9f9bb0c92a97d9ebdf3cecd91acd321d7b95ae81f666edb84731421d4400",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b92fd26cfbe4421bf49ed99435e2bd7c842c4292c692ac82f5868c281f15d1e9",
    "sha256" + debug_suffix: "c9da29772399aec09839eb9aaaf1336121d3f5b75fe904c2fe1ba83ca38d4a49",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2cbcaf1610986031b6bb9018363733e04602ec06ffa52b456929ca1a64664483",
    "sha256" + debug_suffix: "efbcbaeb2d95383a9fb1984fac092d86a855d969fc770525e8d5641876abe46c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "84680c68e3aaa1c8bd08d2b0b6d93e8945544499451608c9f6f7581aeac2686f",
    "sha256" + debug_suffix: "fc82cffe58ec34eb9b9124c20abc7597a00f940a0f236cc06817c8c3981cf8bf",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "7d8090e741fbe1dfbc99aa86389f01669db40ff7eed43715085acd9bc8c9b85f",
    "sha256" + debug_suffix: "580edc3f9e3de8f1ca19fbb4303e88b2cd8eaad4ad03a12b4c452184842b0158",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8a029cf5374ceedd15d1cb52b42d246c274636c545a384aaa4758274cf42d409",
    "sha256" + debug_suffix: "6a7c4c85f5496ea0e3af7db876437c8ec032592807fe4125dee417c398bc7f2e",
  ],
  "kernels_optimized": [
    "sha256": "9bc3379647c18431199ef6734861d0cdf2cd2371b4bfefe8af3efaa02e3ab16f",
    "sha256" + debug_suffix: "adb31c3e7c63429d4974a51f57faed1459afcd799c9b689d1144b850647fb28a",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "5f855fbe931c856fdc34445be6b171fad9ef34e6fffd9a868a6ee352fc3b6af9",
    "sha256" + debug_suffix: "e9e3ed08f2b712e1242f4882dc8851a347cc7e81f8b8c39c97379a99740a8169",
  ],
  "kernels_torchao": [
    "sha256": "945dcfb52769ea25659959e529349dcd75c5f569b92dda6f513fae2167986a84",
    "sha256" + debug_suffix: "fa868125a3fec8db18b0b0d06c0d98fec1a4b6c1cf36c8bcdf5625d7bf237648",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "6d7636220aa1753d58aaf5e699e1bc17ee6ceea039656b2e8c17077091a9a868",
    "sha256" + debug_suffix: "b9fe84319a92a21604fc759f0a8902ec335212084a076d22d3127061aa0f0968",
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
