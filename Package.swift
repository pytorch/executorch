// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260607"
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
    "sha256": "dde9880f0e66a7aa561488978f1865f320eae1bb050d5c53ff6b950eb4463b84",
    "sha256" + debug_suffix: "309f30739fccd93c0612af57c2ffb22ba989f64d4d63b587a5c0b805697944e2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a350b88ffbcde98bde4c91096a8f0752efeb0c3482898a1ee03776df154fd927",
    "sha256" + debug_suffix: "35ba9268d97216836fed45d8ba4203c85f11758e10e1bb2cd1f58916f6e5d3a0",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "9966476b9e4865191e8e65b320fd2788c90fc88eac77889cc2d384af1379a608",
    "sha256" + debug_suffix: "505ef4bc613de319fb06feab1e8a936e89235e3d968cc10c274de653157159e5",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5139a48af88a79fb18335f7ef4894bc679add5cb745503a16059cbfcfedacc0f",
    "sha256" + debug_suffix: "e4438879ac4eff6ea0f1f14823ca65ce6d272eacbefb4de8ca8158919be57740",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a5e83b17f0c6cdb7d24cd98ee748d9ee3570e5181ba5b035f2b0eedde5960e1d",
    "sha256" + debug_suffix: "9f83b1cd961d94b972bb27d40a06c0e2b054ef720efebcd18c38da879eded477",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "cc14989628b700aa26be2f9d9f06ec09fea028e2ddd50916dcb7b7cbd65ec139",
    "sha256" + debug_suffix: "05d86ee46e612ced5c62a271a256c3db0fa3a08147610f229f4a44f726c3f942",
  ],
  "kernels_optimized": [
    "sha256": "c6a592618e8375c10186ad158d6073e8113514ef932ab524707d72ec68b0a78e",
    "sha256" + debug_suffix: "32276f8191c921084b868385a72b1788f774ceadc509d96560e353252350a824",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d06f66213b3d805d1749ab7ab31aa3fa061ff1933ce76422f84452e1ddf7f283",
    "sha256" + debug_suffix: "7efb2ac84c7e7837b3dc082be145ee501aa6bcc4e88de621e5cbc85eb7292cfa",
  ],
  "kernels_torchao": [
    "sha256": "a20189af295d6a5acf266c78daf43e8a2f68bc72ee01c7afe9f53ecf08e72f8f",
    "sha256" + debug_suffix: "53ae36b996d04d191db916305de15ee34769a95593a7d7c45a9c54b2b5bb377a",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "5d0d86952b8ea570797268f4c4e3c6223107c1144d24db9280ac96cd5824d960",
    "sha256" + debug_suffix: "351767fc0d43862af41ddac6e709052e30a664b26b04f03a32234401d4a9bdf1",
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
