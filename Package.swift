// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250922"
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
    "sha256": "03712f4a88a94e4daba14bd0a18900522f81d1f3b7dc77b34a1b523d7ee158da",
    "sha256" + debug_suffix: "5b41a6cc481730e905ea1a522653e780e1c55f6b000a6434113fa7a492974a9f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "fa0924ca80c0b2d078f20d178f7d7a5931480290370c4e121a9b8d0a3bab52a3",
    "sha256" + debug_suffix: "4d917d818b4c1ad5f98987e594799ef0d40e1e1ff54a78b441048a4ab83351c4",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1bef53088cbfb1f54e1fb6fbbb1c522803f537ae5348de081e5ddf62911a95af",
    "sha256" + debug_suffix: "57670930ede3e1334313e4efe94d26ea1ec7851cf380340d468432cd9ceec975",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "c72c3de1e925fd1fc52f27575d7acada3f8f4b0b9b0e2f99aba3c944f89ff749",
    "sha256" + debug_suffix: "9eca3162009486850889f73ce9d301b48395c703b8f3dc0dc1a38ac201ac3ec7",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "2a779a1a38fb21e9cb3e879740733dc12dc87fa2fe41116cc678b9c77450b61b",
    "sha256" + debug_suffix: "9aa6330e8327e72d67c187e708b9d5fdfc1efb70bfb662021e06a9a14a1fc7ae",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "85df2e8ba9a9eafc82ef5c7bf62cbef593d8b19b711b91150561dda2ee31da91",
    "sha256" + debug_suffix: "af017cea7cc7716f57769c37dfac15f2628484a4f618f69cfbe700cb0657486d",
  ],
  "kernels_optimized": [
    "sha256": "d6e212e9bb2342198e1d9cec0a6b1f2b012a5d8d3bc69ff6a35866680080d3da",
    "sha256" + debug_suffix: "17a96eed8b621d8912e43f2c71ef3ce5ade866ab463a86817ec20a3c5359fb12",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "754ebd482b5f8776853c152da2db13942fd7ee3dc9eff042837db3bda8ff0a4d",
    "sha256" + debug_suffix: "ce528289daa81b51eaf86f1719d40ca1c82f061971af702932a366332deba32d",
  ],
  "kernels_torchao": [
    "sha256": "c79856428e7721f4db9f8a9cd66b2413f25ef192a257df777d716774488efda2",
    "sha256" + debug_suffix: "4e6312b7fea284b727ddab6668dc35ac35ca6848c083ad650d1c7c7f90ff636b",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "ada7d8476fde69232413a30c313e8674b151fe1f0155fe7eb231e0cd67d4ee44",
    "sha256" + debug_suffix: "6e89703f3ff1bbfda18f72a10ae5ad6fb2e41b5a6a78cd70fd1417609a748abc",
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
