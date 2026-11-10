// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261110"
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
    "sha256": "0ea026fc949b68ccfc1d461bdcb4258e91a7e446655555c9f40ec8f50acc19d8",
    "sha256" + debug_suffix: "8efd47715fbdd2df0ce4cd4d070adf62f5ff4b313e6060e9dc73a0cc17661a5d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "6df43fcc46abb0e2dd6ed283f9b7baaec94781689475a7486c032e800b59d40a",
    "sha256" + debug_suffix: "e935a35243b36b7c80f2d7cbbc8f68030ce0dadd711f9180d35db66fdad0d383",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "65e49179d7c34015d3a7c70e22fed5153997460bd913fd7c2fd025ad3e16d451",
    "sha256" + debug_suffix: "26d2b8d5067d69573eefdeb367ae581f227be29e064450f411a193b833fde618",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "86a859189a25bb11a5b21bbd94b963c7245884e12066abfe6d42951f65a43446",
    "sha256" + debug_suffix: "e8573d606d22324c98cc6d352d47e566826ee80317c00d1b87e6b9606324c1d4",
    "frameworks": [
      "Accelerate",
      "CoreGraphics",
      "CoreImage",
      "CoreVideo",
      "Foundation",
    ],
    "libraries": [
      "c++",
    ],
  ],
  "executorch_dump": [
    "sha256": "6aa4d55640966961e2dc374994c607ebf180f5aa190aeb59ed26438be173f883",
    "sha256" + debug_suffix: "544265de4d7cdc11766642d0f62b83cdbea7701945efb216a348343415ea64ea",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "9cefb22ec28fb38e09d887422c5e167e2722e429cc4b857b59945dfdd82ce8d3",
    "sha256" + debug_suffix: "924734889aac96561fa41d58b5c07e579ee904f4cf5501644edc62e356ffef62",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "ab57225d99272e86288dd015626f3858da7d62adebfd793e93cc893f7c7c7aa0",
    "sha256" + debug_suffix: "fd927f14b91f2ae86d522166d8377daafc7e218fd9b39da81e7f2a877dc742ab",
  ],
  "kernels_optimized": [
    "sha256": "41ad7cf4ebcdc16540c80df822db9500338b3e9fbb47cc8de4e9b2439f151a95",
    "sha256" + debug_suffix: "a340747185fb2b28015bb3501da5a49b1a82a40b24d31f07fac85cdb4ef545a8",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "fe979230fa7b2a724ced80020a07a945ceb42b5e4ada3abc865f8f2ab2fbaaa6",
    "sha256" + debug_suffix: "e8d41874226f3aff5009d0349daa26ad44b9407382388dbcc52816e6dbe0922c",
  ],
  "kernels_torchao": [
    "sha256": "395d94877277e917e0ab7744f04f035678b48a279098ceb66be3f6d254c0dd7e",
    "sha256" + debug_suffix: "918d5e564296fe500a8ef0a0126d526e65a6e100347184b16fdbe6092ca3301d",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "ea65c131af0af993e7a737a4a8a8b799e1e5426eba17f7d73e8beda671eb20ce",
    "sha256" + debug_suffix: "063b14a932f50e73d8b5a42aa01889f05cbf81420e66b1da67dab70491e1c9e5",
  ],
  "threadpool": [
    "sha256": "90d65438064feb13811a0fe6c3286fbec41f75f0acebd0b619262c85c917ae95",
    "sha256" + debug_suffix: "24c87bab23902adb8c6e4017799762faf27febdba523a25f113cb60320f6862e",
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

// The MLX Metal kernel libraries, one per platform slice, shipped as a single
// resource bundle both MLX products share. Kept out of the generic loop above so
// there is one bundle (executorch_backend_mlx_resources.bundle) rather than a
// separate debug copy, and so the release and debug delegates resolve the same
// name. Each slice's MLX binary asks for its own mlx-<slice>.metallib.
//
// The release job commits all three files before publishing, so they are declared
// unconditionally. A missing one is only reported at package-resolution time and
// does not fail the build, so the release job has to assert they arrived.
let mlxMetallibSlices = ["mlx-ios", "mlx-ios-simulator", "mlx-macos"]
if products.keys.contains("backend_mlx") {
  packageTargets.append(.target(
    name: "backend_mlx_resources",
    path: ".Package.swift/backend_mlx_resources",
    resources: mlxMetallibSlices.map { .copy("\($0).metallib") }
  ))
  for suffix in ["", debug_suffix] {
    if let index = packageTargets.firstIndex(where: {
      $0.name == "backend_mlx\(suffix)\(dependencies_suffix)"
    }) {
      packageTargets[index].dependencies.append(.target(name: "backend_mlx_resources"))
    }
  }
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v14),
  ],
  products: packageProducts,
  targets: packageTargets
)
