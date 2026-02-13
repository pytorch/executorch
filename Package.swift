// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260213"
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
    "sha256": "a185fc672f2117e77cae4b1c0802f4e965681752809f3c56737e6e41a72af062",
    "sha256" + debug_suffix: "acabf342f9f9b63a47e32098eeaa05facaba2f0ffc6e0a0fa703b00203f1b75c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2b2c26471f72bcd107a49c0c6e2c2fa00060e68cc047d5d6536657a49f4c3fe5",
    "sha256" + debug_suffix: "94584ddedb530f8316a3bd35f0f0ac82fbb3f6a159d94aab85446fcb3e5bf12a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6ab12b219704374f3ce1b7e93f64003025d9d979b377cd93a54927799dc1d96f",
    "sha256" + debug_suffix: "895947ed930495492523a9903bb96d5970d363c4bee8876c22ffe40f7093d64e",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "75a5871671c0dfb0e32f70a6ae20fd1377a24a5ec321c8affc691ddaf297e276",
    "sha256" + debug_suffix: "871e77a00d16ce46f8243621f2818d8740d3b6d5ba823a7781f5743176671bc1",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "d2b8c65508810954429f215abad6c67e4459d5fa6f477f8f4cc301849a7620fc",
    "sha256" + debug_suffix: "07ed22ed41a116b343b4671ca96787bc68a0ff084c454894b277729ebb63b0f7",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c30d2e402f1d909124f6da5532d121a285348b4ba3c59766b06b20da713e1288",
    "sha256" + debug_suffix: "892b1ec0bf3f2b1d74248994ff5b1141d8277b5e1baca515b346c6c1c40c0bcc",
  ],
  "kernels_optimized": [
    "sha256": "decd9472f5de14f88fdaad1d86089d571f924c34953cadd36a19cb69523ab4d0",
    "sha256" + debug_suffix: "1639e54ca0bd576b489c8f3a1b52caf3187f732162e58020d560530576ac8437",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "76fcf7dc4a8ca14eade1f7b7076e1f775ef49555ee0580866e7219920e3a1320",
    "sha256" + debug_suffix: "c16fed845d6b5632a589a286b3e3cebdef7b7ecf0991423d8e2d00a3be2fb648",
  ],
  "kernels_torchao": [
    "sha256": "7996f93685b088d8b4cb09149d7d6a5217db6503937c123e580aed90d577856b",
    "sha256" + debug_suffix: "e30790f2630b2c2b1dc62390679b651251457b5ebf332575b2c97e7366955732",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "00147e325227cb6835ed076ba8edf3f1f2ff07e6d3f211b4f0061f22749665b5",
    "sha256" + debug_suffix: "81fbf05f4f9d7d2ec667a33f59cf02531c363ce8164ee9c503e6bd4da7e4b4b8",
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
