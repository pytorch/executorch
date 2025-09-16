// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250916"
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
    "sha256": "f57b126c2ed42b3d864a294e45cefe5df4d18a06f1a6fe34690732357d576b99",
    "sha256" + debug_suffix: "4d483f827e39c7ef4bbea146a18bccf89d9a994d1a877282669a2ae767da9418",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "472022eff249d6f20e9a6f895def6a3957b312c822fa24b98a3a4205dc8f05be",
    "sha256" + debug_suffix: "1c9b7a2483d071b03ba3769dbf7cc9e782bfd597528941cf98c566516cd22320",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8bd87cba59afa9388e00924567f81da1cd6ca5bc236fbc0ab1848c5cef00edd6",
    "sha256" + debug_suffix: "2d4f1e260e55441c82c39fb96aaa21b8b1d7bb47cb5f7340dada964eddedf7e5",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0e1fef84610561fb632448decd63bed462ff1543fd5281df84dc53f1ec444fc1",
    "sha256" + debug_suffix: "d4879b6fd5c62e4facd0319cdbe506d5d1d7cf229885294c5f6ffbae3bef3bdc",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "68e8a8ad80d74238770ba3d8d6f0135f59da993bee0d57d937cd23654ed1600d",
    "sha256" + debug_suffix: "c8902fcb7522883da852d8433103220102bb123680e907c63bf21ca2b7cabf84",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "9eaa73c45079c0948eb825ce13d7a316a538b6f90c9458427f84e84faf262845",
    "sha256" + debug_suffix: "088e972e7d3075ae5f6d2c9ea999559bd0e68f070ac4d3ec98eff714e3a350a4",
  ],
  "kernels_optimized": [
    "sha256": "a4d40d3b1396c404fe2f03458e5ffb2d9394fae676198aaa45e561360570d7ca",
    "sha256" + debug_suffix: "16e55c5fabab581f3515f753cab4633fa4ead6691930fb6fd52eb49b9d64869e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c748fd99c287aebc4eb69bdbf9f7978712f1c347192bc6fa1f0cced34dfcbe96",
    "sha256" + debug_suffix: "979ab72aae32303a30754d2ee7cd37645ab69568947b001ceef9248bb69ce8ed",
  ],
  "kernels_torchao": [
    "sha256": "b2328a98e6f02c0b26ccc09dd543af25f494a11ed225e7ba5ba38ce4a8d7b6b2",
    "sha256" + debug_suffix: "0c456990b5ec1407ef524c3ba57875e72cd25fd8ee3b40b26654979bde59006d",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "7731072ce1193dacd1f4fc97252d951cb82b94e7a3d37280823ccc955c95f9ad",
    "sha256" + debug_suffix: "4e5707dbfbc9ce314f345cf084f36ef764a44650cf282f362b836dc20edf8296",
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
