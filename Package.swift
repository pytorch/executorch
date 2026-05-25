// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260525"
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
    "sha256": "ab112caeee524ad19c84bb010657049171516c6bb241f7ab07055fe10f556931",
    "sha256" + debug_suffix: "b6a5748cfe10052fed22a3f9bf15ca11261c8f1e305afe90cbb4edc9c08feadd",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6298a469c7c8ed20f41fbbf92ab85d28679214dbb7a83f2f4c4db629440a25a2",
    "sha256" + debug_suffix: "b93313c4fa76abb97ebcfc22dce94d46fdf53857203210b9bbc73268c59478ae",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3847295833232042c1d45f7ef8d8a474663af79fd9511d1cf397ad92fd014f7d",
    "sha256" + debug_suffix: "12ff81969b1773dbf8db36c8f22d04606ad4148e77f5829400c022db20807cea",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0c943e6e5a5377d88e436b1c4542f0877741be16d28fe5e6d5af36391686cb5a",
    "sha256" + debug_suffix: "520f87a4c5de7451f8de84974cdb5be7bf6f1b6b0820038076d6632f622ce633",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "88cf4bb73add8f448d792d012ccc90269a13572fcc7586f3f8a2b69aa8b51ad8",
    "sha256" + debug_suffix: "d350b6c7f17e2edc57632678fa0084548f3b26b60445031bdd8b8dccffb58d8a",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "5d05f018c13b127bba15d5e189f6b04785453fbe92dcd464a893b59be5998310",
    "sha256" + debug_suffix: "295cd9f86fd04d8f6dd48cb9067986ccda94d1e95e8aa96dcfed5ca25814dc82",
  ],
  "kernels_optimized": [
    "sha256": "779fad3fbaac5c983bbd6e1777c33e2ae68fce072cf6664aa096535cb9b02798",
    "sha256" + debug_suffix: "445c5ac770bc585d57da83760a0b81995b01a17c45c41966c1953adbb541393f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d1ecaa37a140b1ff76bccae52e868e153ab529826cc8f6e527541f85fdb427d3",
    "sha256" + debug_suffix: "aa1300b9b903ee74cf240e0af336730e4a5574990b5a032309cb66c2a7e1d5f4",
  ],
  "kernels_torchao": [
    "sha256": "2bbb52ebf97fc2c6bd7a9990e425e1f71a0c465f33a9a37e4e8e325617aa1cbc",
    "sha256" + debug_suffix: "3b7962c8d4f9684700f0c72d6b74044f4f70849bea96d823f3fd328aa0d20e4f",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "4f092e5c98562162bed9386c2ac0c217db3f65e7be89f340677b9491713e4659",
    "sha256" + debug_suffix: "e82838f6da235142c39f51a417d5d06a783490f591c852bdef3840fbde477026",
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
