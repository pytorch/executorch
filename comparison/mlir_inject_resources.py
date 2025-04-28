import re

def collect_required_resources(content):
    """ 필요한 dense_resource 이름과 타입(사이즈) 추출 """
    pattern = r'dense_resource<([\w\\\.]+)>\s*:\s*tensor<(\d+)x(f\d+)>'
    matches = re.findall(pattern, content)
    resources = {}
    for name, dim, dtype in matches:
        print("- Found: ", name, dim, dtype)
        dim = int(dim)
        if dtype == "f32":
            size = dim * 4  # float32 = 4 bytes
        elif dtype == "f16":
            size = dim * 2  # float16 = 2 bytes
        elif dtype == "i32":
            size = dim * 4  # int32 = 4 bytes
        else:
            raise ValueError(f"Unsupported dtype {dtype}")
        resources[name] = size
    return resources

def generate_dummy_blob(size):
    """ 크기에 맞는 00 blob 생성 (hex string) """
    return '00' * size

def generate_dialect_resource_block(resource_sizes):
    """ dialect_resources 블록 생성 """
    lines = ["{-#", "  dialect_resources: {", "    builtin: {"]
    for key, size in resource_sizes.items():
        blob = generate_dummy_blob(size)
        lines.append(f'      {key}: "0x04000000{blob}",')
    if lines[-1].endswith(','):
        lines[-1] = lines[-1][:-1]  # 마지막 콤마 제거
    lines += ["    }", "  }", "#-}"]
    return "\n".join(lines)

def patch_file(target_file, output_file):
    with open(target_file, 'r') as f:
        content = f.read()
    resource_sizes = collect_required_resources(content)
    if len(list(resource_sizes.values())) > 0:
        dialect_block = generate_dialect_resource_block(resource_sizes)

        with open(output_file, 'a') as f:
            f.write("\n\n" + dialect_block + "\n")

        print(f"Patched file saved to {output_file}")