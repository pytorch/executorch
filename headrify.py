import binascii
bytes_per_line = 32
hex_digits_per_line = bytes_per_line * 2

# copied from
# https://git.mlplatform.org/ml/ethos-u/ml-embedded-evaluation-kit.git/tree/scripts/py/gen_model_cpp.py

magic_attr = '__attribute__((section(".sram.data"), aligned(16))) char'
# magic_attr = '__attribute__((section("network_model_sec"), aligned(16))) char'
# magic_attr = '__attribute__((section("input_data_sec"), aligned(16))) char'
filename="./add.pte"
filename="add_u55_fp32.pte"
with open(filename, "rb") as fr, open(f"{filename}.h", "w") as fw:
    data = fr.read()
    hexstream = binascii.hexlify(data).decode('utf-8')

    hexstring = magic_attr + ' add_pte[] = {'

    for i in range(0, len(hexstream), 2):
        if 0 == (i % hex_digits_per_line):
            hexstring += "\n"
        hexstring += '0x' + hexstream[i:i+2] + ", "

    hexstring += '};\n'
    fw.write(hexstring)
    print(f"Wrote {len(hexstring)} bytes, original {len(data)}")

