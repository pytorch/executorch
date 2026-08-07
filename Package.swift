// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0"
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
    "sha256": "33425eb7c1e2a0eeb70a6f47dbe10efd2827557031f0389546dc310adb1c16b2",
    "sha256" + debug_suffix: "1398e71758dc15a7097e068fb7805ad74da31647e79522f04f9f046d9344d91d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e53a4ace14460ae68371346dd7b93efe5343518fcb09c5d03a780a222b3c1dc3",
    "sha256" + debug_suffix: "16439b377460b1ce407569c9e9dcf2f4412e134de939e6413a03d270d3a0c108",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8931a9ddfc5916e9424ba578a1484075f4765dde9fee4eced4ccb522ddbf5abb",
    "sha256" + debug_suffix: "e08316c25114b47485429bd614594be2017cb41b1e32c462498628a4fb3fc1d3",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "6b289f42055af779f226015bd944df4c28bedd039a2e6e1d6ca5a9c391138f10",
    "sha256" + debug_suffix: "1b8aa630f055ba6915fba3b7acc4970568f8da75af646473d648ce9cf064b93b",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "2f7f6c4849fb759b85c909f839976616d3eea92ca71001eb2f3fd985431d0921",
    "sha256" + debug_suffix: "8ef2b15b4bb5c72bc1e261671bd975bf2279f9ebecbba5b2f0e4c78fa3a85df1",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "7ac1e389b19bb09ce7482a0410c01b7494f034893e421d75e948cd58c5b77b9e",
    "sha256" + debug_suffix: "7f8a96a232e48fb50091640051e6e0d00521b39b53cfda50cba514944b70511d",
  ],
  "kernels_optimized": [
    "sha256": "6b8a48fe371c523fde12dc528e013e069272c54dda1b82cd3108945bdd5eeda9",
    "sha256" + debug_suffix: "cf04d82942b545f4fc35daf387209dd30cd6ad302242bb0a57efecd38594952f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "dff8280aa24160d66db56d3bc2ecdfc4b17bc6e7fa35f14733ead28d7ffb954c",
    "sha256" + debug_suffix: "59b656bc6fca851424c33114c98a250646824165b8b0bb0296a61f71041f1cc4",
  ],
  "kernels_torchao": [
    "sha256": "0bcfd66732bef3293949157ec72097eb2dd87dd19875969a36b62f7d2627113a",
    "sha256" + debug_suffix: "70797307d08defcabef725b6a58d9b13da726eafcf03ef27e82bada54279ac01",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "051bafc8849f1091b7b372e770470e926a7fa590ca1ec9b6986e4f36de1225c3",
    "sha256" + debug_suffix: "67a6fe5ccf187f4e6f229597501c584af3ed3f8298294ad1341b0554e0b7d521",
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
