// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250930"
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
    "sha256": "20d67f1f4679a3733340e8319c46634886a13b6d3466c0ede633899e2b73ba73",
    "sha256" + debug_suffix: "96108c6afc35ed08c0a4d037d1a30f95e4dc2c7a0b667536cf617ad9f154c53e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ac62f7c8748517ca812c1eb008837429a427d4da55bf67c23cab46bd8ec9d7be",
    "sha256" + debug_suffix: "27f1d0f74043ac495a27fdfb6e92e67da3a32b826fcd4bfa7b24be5de063c805",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f253da4d49f100e08f010f46782b4425c56bcfac96e94353f59c1708bb29ba66",
    "sha256" + debug_suffix: "288b314ada0bf866217952029bdfdd8cd49f7227a6de1f7eabbca6e6829b6a03",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "ae52828a61c6176a3057f95a4019a4b0086b2f3af9b915531ba976fc90a361e4",
    "sha256" + debug_suffix: "a46e76db63f0cbdd2693408be23efae228792eb561954bc286e4a0c5d2b94645",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a04529054bb47067bdc13ee364f963f43fb4375abf5af12a1e1276d5ac8f85e5",
    "sha256" + debug_suffix: "cf52343908958bd03b5da7bf9b92ba6ec269324b50c36c8edb28662675b05479",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "9b880477cb076955451c083e3bb0a15dd27a4cbfff37f6e37c3f9d735a21ab86",
    "sha256" + debug_suffix: "3aa3da61b65842618e6bd8ffb456b74678181dbcd8b1d2cd9a425fd42b4d0202",
  ],
  "kernels_optimized": [
    "sha256": "061b5809cba35881cf06124ceee7e70d6bda6bbcac9d5d9a2e6ee1f2897926f5",
    "sha256" + debug_suffix: "7e9f14bbfe5e732d403acacad4e2b54f4b5d0f4168d2a6024db015b84d72ab2d",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "de99578fea7bd65bbb9556a9a7a4b07e1576219fc17db537e78a50078f7bbd5e",
    "sha256" + debug_suffix: "a6bbd3010b79120eca94ab6424967444aa95ec68119cf509f199ed05a243ab68",
  ],
  "kernels_torchao": [
    "sha256": "ac45388611bcbdcc207823eeb06ae562fd172d815c4e394af2d02b051bf2a47e",
    "sha256" + debug_suffix: "f59712f514430a1547ef4b55b83589fb22296ff401a6af520774383acc6ed108",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "933fc912ef9ec11ceb4c81979e825a2711ce50c1b41af240dac8a7146f596aff",
    "sha256" + debug_suffix: "ebf86c25fbe191c4d6f9cddda81660ca196c92536a22225c059f6ed329fb22dc",
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
