// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251227"
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
    "sha256": "0db8edda14a960ebd0c09f1a6497eb35d125b4e6fd3fb20149056a872cf97cce",
    "sha256" + debug_suffix: "bbf4b9b03e341e1bfc283ff4bee6c78cc6659f8a094f43bd05ca5c213f160239",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "bb04b12b24aebb5e2285328259f074d1c24c0383caa7faa2d173aa7eb0de2a2d",
    "sha256" + debug_suffix: "bb88e34c3800368dddda47a6054c9594966186ea8e2f5ce85b84318a45c48cc3",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0c42aab86df1b9788e448486771d3be3395335262c0be113754326bd5f2b14fa",
    "sha256" + debug_suffix: "daf40a39ce7733e73ee158bd59bd1b07a72b6008f6f9901da9668d86888a99be",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "860b6039fbc4309a2b84fa37f29f88f392de9ac03775214cf1d07fa3f0ed12f1",
    "sha256" + debug_suffix: "550753edd47f5299c65bbd003e6938252bc1a784283164aa29f9bc9301bed054",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "62fa589cc8fc9e5e83445e0b53321330673f0b70ac3dc7e609500edfcaaf3794",
    "sha256" + debug_suffix: "668ed7b1ae33e9eb207a98b36350daa5f0732b138dd65618c9b71c64148dc14f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "92c592812c3e218f25620435c5c9653b0b22384b17ff8242d83f47ddcc13b5e7",
    "sha256" + debug_suffix: "f64652ab3481ea6c2ddac3fb56c4613d1822a239ccadf0244be5e1321b4cdef4",
  ],
  "kernels_optimized": [
    "sha256": "2f9f8374a106b688f940cd0d5d1fd90b510ebd8cb9e5cb8daa1a36270806ca9b",
    "sha256" + debug_suffix: "1506d9ed8296ec954f57ee3b5edc22059619754122ea4a44e71ccca6efe3da3b",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "74c7ae934522c41e999c6d2262f4d1d39da68e94d3bcfa740779ebc32afe2c6d",
    "sha256" + debug_suffix: "b5db9eadda9c20dac4067071933f28bd4782b30df1145d2c710c316c3da3c61f",
  ],
  "kernels_torchao": [
    "sha256": "e5b09796bb68c20fad3c73f2bf054c3c6ebaa787027fa4cd688a59f9007b7d47",
    "sha256" + debug_suffix: "6a348ff97587d0265be91a3d347e8ea2a61a5522f045cc37a2d55911f3928141",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "159e396e50260accc0fb6db75bca336c87b320e82c4c2abf1941976526c5cae9",
    "sha256" + debug_suffix: "6a70257ca95365784ecf3aae89afeb9170ab12c3f430dccfd335454b851f5ebf",
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
