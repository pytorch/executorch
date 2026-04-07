// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260407"
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
    "sha256": "7bcfece4c676ab237287d23e34c39ce641c1e730d807eceb19bc685b77464917",
    "sha256" + debug_suffix: "fa34177489284121690e4e10b11be3019b47d90d5eb2d02dea7299e7277cda3a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5d1fc40758bf21e864b32adc824fab0aff8d860c507280e8e543889e64905c04",
    "sha256" + debug_suffix: "7d81ec2b056f48cc6e8c8e4e1ba533932ca9981c3bd384cec1dec24569b8fd64",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "de87e5a82eb2e25d63d6c07e6829098ccbf961d3d507d98d5d4695e304d413e2",
    "sha256" + debug_suffix: "5052df63f8f55fe342f16afd6dd88885e59223cb5f20ba36eb9757842d0d789b",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9224f2c6a9906800cc634454ea969f505a38fc0c56a5b94ee54b5f3e2b6c32b5",
    "sha256" + debug_suffix: "b3e8cfdcdeb4dad7b4e90c78a06da124e332d4b2063f6a299a6ef9e689abe1b6",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "0a04266c4231ae5587aaaa53f5a0b6b25b75659d601cecfd3285383c89791054",
    "sha256" + debug_suffix: "80d98271d3e1e559a65e4c36e604f674003ace0037db129d9bd04c9ed20d7e02",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "77e9d47ed0d0efb7ea24d3944b2ef75054b347741c094fc2d916f7913c4048ef",
    "sha256" + debug_suffix: "f108126846463f645737806a51e04eb08fda414aba969e484005bc41192cf42e",
  ],
  "kernels_optimized": [
    "sha256": "809ebdad68f8fd5a846555dfb9cf21f996c7d56d7ba06feef2db4783c247efeb",
    "sha256" + debug_suffix: "b140289695f3ba148b9c8b7392b9fe0744ab228681c23e1d70307e8ff2b283e5",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9fef7c281e0c079d2b5a0fd9b6ddc6c426f6cbf5fc06cee6aebdcd5c54992443",
    "sha256" + debug_suffix: "625e004d9d3ee7b20be288ab37529a74ab564ced3620bd51c2d4cb55a47bc839",
  ],
  "kernels_torchao": [
    "sha256": "96398b3f7ce8fc19ca7b0c431899a7a411e86632face28d0d961ee18bb6a3b34",
    "sha256" + debug_suffix: "f5b96d680c1d6eee36e714e6e4af396f19ee4ff4cb3b99d0b09463a60c23cece",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "70302740b890ea8bca137c29e7efe695185912b7fc6a968e074415e79202ea2d",
    "sha256" + debug_suffix: "032f8f7be8c2cb0923d3d24037c5951f143ac9e0748c81634dcef22c78999fdb",
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
