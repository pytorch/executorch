// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250901"
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
    "sha256": "07a31a015464b341b71303455a51c361ded89ab291e7b69d8a8774887d2cb42d",
    "sha256" + debug_suffix: "3705aa58af8b39ecdafa3b5f2ea760ce3332b6032a26ca64f7af4eff0e239252",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "570be1c97c233515f751896518cfc5e999e0aa4db13fa783d86a3da1b6e92651",
    "sha256" + debug_suffix: "9ac52cb034e3346638a4715d874149871a500fa0ebc1ddf61579cd8898e011fa",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "949bd3c478eb3c6060dbc83182a38951183cbbec3a6335071d8900f90f463fa3",
    "sha256" + debug_suffix: "69a7d71cca0a79cfaefa7d728118172cce1ee7f4c93b1edde1a984f88a2e633b",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "336471902fccf6384249eded7b375f09c398789531bba04507202bad47cb87ca",
    "sha256" + debug_suffix: "bf51256c9e2aced783c5b15183f8a08b5181972db0a925a1f0acc10c90893817",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "b458482c608cde2fa872d1ddbb226b40bca00e6124153e9802575cbd4065d9c0",
    "sha256" + debug_suffix: "8611b400ff1ec5361ec9cb4878dd925026c746459878e898a1d58075e6981760",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f58a556d6666e6814019b97bbee820822796f46b21d310c702d609b93b7a1937",
    "sha256" + debug_suffix: "2460f2fdc98189f0f7511711ae1afa36c44e2097a9f51e6010c919fc30b90686",
  ],
  "kernels_optimized": [
    "sha256": "04f7ccd34256817b9487ea1bd5a18d667bbb4fb37ab919fb57bf8b6e3085860c",
    "sha256" + debug_suffix: "920201799f3ffc76614b51c06125240648d6938cd0a9eed2263e25b25b116cd5",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "02074cfcee603afd54fa3b1bd195a29203459cc7d8135a2e9c2109dd61266251",
    "sha256" + debug_suffix: "e625063e19e2c107f26bfd247ece8132c4e283363d454c3167729eda4ff38264",
  ],
  "kernels_torchao": [
    "sha256": "d2a037acd18dd8c8052fb456569355613343a47ebf976e694816ffadc8dea437",
    "sha256" + debug_suffix: "4ec6b5a7a12c625c383e9827ee76ace710eff23dbe248db6b087dce22c388f30",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "dd3638508f7ab0fffc6c01b14b6018ba3cedfd80d1c56b75fcdcd58bda763401",
    "sha256" + debug_suffix: "46f9fa3e1d20d819c2ac621541263a348bc9947c9d97a9578f091b076c251a4f",
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
