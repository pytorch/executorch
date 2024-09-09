import csv
import os
import sys
import subprocess

max_log_n = 30


def is_distinct(l):
    return len(set(l)) == len(l)


def float_avx_0(register, aux_registers, ident=''):
    if not is_distinct(aux_registers):
        raise Exception('auxiliary registers must be distinct')
    if register in aux_registers:
        raise Exception(
            'the main register can\'t be one of the auxiliary ones')
    if len(aux_registers) < 4:
        raise Exception('float_avx_0 needs at least four auxiliary registers')
    res = ident + '"vpermilps $160, %%%%%s, %%%%%s\\n"\n' % (register,
                                                             aux_registers[0])
    res += ident + '"vpermilps $245, %%%%%s, %%%%%s\\n"\n' % (register,
                                                              aux_registers[1])
    res += ident + '"vxorps %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        aux_registers[2], aux_registers[2], aux_registers[2])
    res += ident + '"vsubps %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        aux_registers[1], aux_registers[2], aux_registers[3])
    res += ident + '"vaddsubps %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        aux_registers[3], aux_registers[0], register)
    return res


def float_avx_1(register, aux_registers, ident=''):
    if not is_distinct(aux_registers):
        raise Exception('auxiliary registers must be distinct')
    if register in aux_registers:
        raise Exception(
            'the main register can\'t be one of the auxiliary ones')
    if len(aux_registers) < 5:
        raise Exception('float_avx_1 needs at least five auxiliary registers')
    res = ident + '"vpermilps $68, %%%%%s, %%%%%s\\n"\n' % (register,
                                                            aux_registers[0])
    res += ident + '"vpermilps $238, %%%%%s, %%%%%s\\n"\n' % (register,
                                                              aux_registers[1])
    res += ident + '"vxorps %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        aux_registers[2], aux_registers[2], aux_registers[2])
    res += ident + '"vsubps %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        aux_registers[1], aux_registers[2], aux_registers[3])
    res += ident + '"vblendps $204, %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        aux_registers[3], aux_registers[1], aux_registers[4])
    res += ident + '"vaddps %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        aux_registers[0], aux_registers[4], register)
    return res


def float_avx_2(register, aux_registers, ident=''):
    if not is_distinct(aux_registers):
        raise Exception('auxiliary registers must be distinct')
    if register in aux_registers:
        raise Exception(
            'the main register can\'t be one of the auxiliary ones')
    if len(aux_registers) < 4:
        raise Exception('float_avx_2 needs at least four auxiliary registers')
    res = ident + '"vxorps %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        aux_registers[0], aux_registers[0], aux_registers[0])
    res += ident + '"vsubps %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        register, aux_registers[0], aux_registers[1])
    res += ident + '"vperm2f128 $0, %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        register, register, aux_registers[2])
    res += ident + '"vperm2f128 $49, %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        aux_registers[1], register, aux_registers[3])
    res += ident + '"vaddps %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        aux_registers[2], aux_registers[3], register)
    return res


def float_avx_3_etc(from_register_0,
                    from_register_1,
                    to_register_0,
                    to_register_1,
                    ident=''):
    if not is_distinct(
        [from_register_0, from_register_1, to_register_0, to_register_1]):
        raise Exception('four registers must be distinct')
    res = ident + '"vaddps %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        from_register_1, from_register_0, to_register_0)
    res += ident + '"vsubps %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        from_register_1, from_register_0, to_register_1)
    return res


def double_avx_0(register, aux_registers, ident=''):
    if not is_distinct(aux_registers):
        raise Exception('auxiliary registers must be distinct')
    if register in aux_registers:
        raise Exception(
            'the main register can\'t be one of the auxiliary ones')
    if len(aux_registers) < 4:
        raise Exception('double_avx_0 needs at least four auxiliary registers')
    res = ident + '"vpermilpd $0, %%%%%s, %%%%%s\\n"\n' % (register,
                                                           aux_registers[0])
    res += ident + '"vpermilpd $15, %%%%%s, %%%%%s\\n"\n' % (register,
                                                             aux_registers[1])
    res += ident + '"vxorpd %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        aux_registers[2], aux_registers[2], aux_registers[2])
    res += ident + '"vsubpd %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        aux_registers[1], aux_registers[2], aux_registers[3])
    res += ident + '"vaddsubpd %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        aux_registers[3], aux_registers[0], register)
    return res


def double_avx_1(register, aux_registers, ident=''):
    if not is_distinct(aux_registers):
        raise Exception('auxiliary registers must be distinct')
    if register in aux_registers:
        raise Exception(
            'the main register can\'t be one of the auxiliary ones')
    if len(aux_registers) < 4:
        raise Exception('double_avx_1 needs at least four auxiliary registers')
    res = ident + '"vperm2f128 $0, %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        register, register, aux_registers[0])
    res += ident + '"vxorpd %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        aux_registers[1], aux_registers[1], aux_registers[1])
    res += ident + '"vsubpd %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        register, aux_registers[1], aux_registers[2])
    res += ident + '"vperm2f128 $49, %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        aux_registers[2], register, aux_registers[3])
    res += ident + '"vaddpd %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        aux_registers[3], aux_registers[0], register)
    return res


def double_avx_2_etc(from_register_0,
                     from_register_1,
                     to_register_0,
                     to_register_1,
                     ident=''):
    if not is_distinct(
        [from_register_0, from_register_1, to_register_0, to_register_1]):
        raise Exception('four registers must be distinct')
    res = ident + '"vaddpd %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        from_register_1, from_register_0, to_register_0)
    res += ident + '"vsubpd %%%%%s, %%%%%s, %%%%%s\\n"\n' % (
        from_register_1, from_register_0, to_register_1)
    return res


def float_sse_0(register, aux_registers, ident=''):
    if not is_distinct(aux_registers):
        raise Exception('auxiliary registers must be distinct')
    if register in aux_registers:
        raise Exception(
            'the main register can\'t be one of the auxiliary ones')
    if len(aux_registers) < 2:
        raise Exception('float_sse_0 needs at least two auxiliary registers')
    res = ident + '"movaps %%%%%s, %%%%%s\\n"\n' % (register, aux_registers[0])
    res += ident + '"shufps $160, %%%%%s, %%%%%s\\n"\n' % (aux_registers[0],
                                                           aux_registers[0])
    res += ident + '"shufps $245, %%%%%s, %%%%%s\\n"\n' % (register, register)
    res += ident + '"xorps %%%%%s, %%%%%s\\n"\n' % (aux_registers[1],
                                                    aux_registers[1])
    res += ident + '"subps %%%%%s, %%%%%s\\n"\n' % (register, aux_registers[1])
    res += ident + '"addsubps %%%%%s, %%%%%s\\n"\n' % (aux_registers[1],
                                                       aux_registers[0])
    res += ident + '"movaps %%%%%s, %%%%%s\\n"\n' % (aux_registers[0],
                                                     register)
    return res


def float_sse_1(register, aux_registers, ident=''):
    if not is_distinct(aux_registers):
        raise Exception('auxiliary registers must be distinct')
    if register in aux_registers:
        raise Exception(
            'the main register can\'t be one of the auxiliary ones')
    if len(aux_registers) < 4:
        raise Exception('float_sse_1 needs at least four auxiliary registers')
    res = ident + '"movaps %%%%%s, %%%%%s\\n"\n' % (register, aux_registers[0])
    res += ident + '"shufps $68, %%%%%s, %%%%%s\\n"\n' % (aux_registers[0],
                                                          aux_registers[0])
    res += ident + '"xorps %%%%%s, %%%%%s\\n"\n' % (aux_registers[1],
                                                    aux_registers[1])
    res += ident + '"movaps %%%%%s, %%%%%s\\n"\n' % (register,
                                                     aux_registers[2])
    res += ident + '"shufps $14, %%%%%s, %%%%%s\\n"\n' % (aux_registers[1],
                                                          aux_registers[2])
    res += ident + '"movaps %%%%%s, %%%%%s\\n"\n' % (register,
                                                     aux_registers[3])
    res += ident + '"shufps $224, %%%%%s, %%%%%s\\n"\n' % (aux_registers[3],
                                                           aux_registers[1])
    res += ident + '"addps %%%%%s, %%%%%s\\n"\n' % (aux_registers[0],
                                                    aux_registers[2])
    res += ident + '"subps %%%%%s, %%%%%s\\n"\n' % (aux_registers[1],
                                                    aux_registers[2])
    res += ident + '"movaps %%%%%s, %%%%%s\\n"\n' % (aux_registers[2],
                                                     register)
    return res


def float_sse_2_etc(from_register_0,
                    from_register_1,
                    to_register_0,
                    to_register_1,
                    ident=''):
    if not is_distinct(
        [from_register_0, from_register_1, to_register_0, to_register_1]):
        raise Exception('four registers must be distinct')
    res = ident + '"movaps %%%%%s, %%%%%s\\n"\n' % (from_register_0,
                                                    to_register_0)
    res += ident + '"movaps %%%%%s, %%%%%s\\n"\n' % (from_register_0,
                                                     to_register_1)
    res += ident + '"addps %%%%%s, %%%%%s\\n"\n' % (from_register_1,
                                                    to_register_0)
    res += ident + '"subps %%%%%s, %%%%%s\\n"\n' % (from_register_1,
                                                    to_register_1)
    return res


def double_sse_0(register, aux_registers, ident=''):
    if not is_distinct(aux_registers):
        raise Exception('auxiliary registers must be distinct')
    if register in aux_registers:
        raise Exception(
            'the main register can\'t be one of the auxiliary ones')
    if len(aux_registers) < 2:
        raise Exception('double_sse_0 needs at least two auxiliary registers')
    res = ident + '"movapd %%%%%s, %%%%%s\\n"\n' % (register, aux_registers[0])
    res += ident + '"haddpd %%%%%s, %%%%%s\\n"\n' % (aux_registers[0],
                                                     aux_registers[0])
    res += ident + '"movapd %%%%%s, %%%%%s\\n"\n' % (register,
                                                     aux_registers[1])
    res += ident + '"hsubpd %%%%%s, %%%%%s\\n"\n' % (aux_registers[1],
                                                     aux_registers[1])
    res += ident + '"blendpd $1, %%%%%s, %%%%%s\\n"\n' % (aux_registers[0],
                                                          aux_registers[1])
    res += ident + '"movapd %%%%%s, %%%%%s\\n"\n' % (aux_registers[1],
                                                     register)
    return res


def double_sse_1_etc(from_register_0,
                     from_register_1,
                     to_register_0,
                     to_register_1,
                     ident=''):
    if not is_distinct(
        [from_register_0, from_register_1, to_register_0, to_register_1]):
        raise Exception('four registers must be distinct')
    res = ident + '"movapd %%%%%s, %%%%%s\\n"\n' % (from_register_0,
                                                    to_register_0)
    res += ident + '"movapd %%%%%s, %%%%%s\\n"\n' % (from_register_0,
                                                     to_register_1)
    res += ident + '"addpd %%%%%s, %%%%%s\\n"\n' % (from_register_1,
                                                    to_register_0)
    res += ident + '"subpd %%%%%s, %%%%%s\\n"\n' % (from_register_1,
                                                    to_register_1)
    return res


def plain_step(type_name, buf_name, log_n, it, ident=''):
    if log_n <= 0:
        raise Exception("log_n must be positive")
    if it < 0:
        raise Exception("it must be non-negative")
    if it >= log_n:
        raise Exception("it must be smaller than log_n")
    n = 1 << log_n
    res = ident + "for (int j = 0; j < %d; j += %d) {\n" % (n, 1 << (it + 1))
    res += ident + "  for (int k = 0; k < %d; ++k) {\n" % (1 << it)
    res += ident + "    %s u = %s[j + k];\n" % (type_name, buf_name)
    res += ident + "    %s v = %s[j + k + %d];\n" % (type_name, buf_name,
                                                     1 << it)
    res += ident + "    %s[j + k] = u + v;\n" % buf_name
    res += ident + "    %s[j + k + %d] = u - v;\n" % (buf_name, 1 << it)
    res += ident + "  }\n"
    res += ident + "}\n"
    return res


def composite_step(buf_name,
                   log_n,
                   from_it,
                   to_it,
                   log_w,
                   registers,
                   move_instruction,
                   special_steps,
                   main_step,
                   ident=''):
    if log_n < log_w:
        raise Exception('need at least %d elements' % (1 << log_w))
    num_registers = len(registers)
    if num_registers % 2 == 1:
        raise Exception('odd number of registers: %d' % num_registers)
    num_nontrivial_levels = 0
    if to_it > log_w:
        first_nontrivial = max(from_it, log_w)
        num_nontrivial_levels = to_it - first_nontrivial
        if 1 << num_nontrivial_levels > num_registers / 2:
            raise Exception('not enough registers')
    n = 1 << log_n
    input_registers = []
    output_registers = []
    for i in range(num_registers):
        if i < num_registers / 2:
            input_registers.append(registers[i])
        else:
            output_registers.append(registers[i])
    clobber = ', '.join(['"%%%s"' % x for x in registers])
    if num_nontrivial_levels == 0:
        res = ident + 'for (int j = 0; j < %d; j += %d) {\n' % (n, 1 << log_w)
        res += ident + '  __asm__ volatile (\n'
        res += ident + '    "%s (%%0), %%%%%s\\n"\n' % (move_instruction,
                                                        input_registers[0])
        for it in range(from_it, to_it):
            res += special_steps[it](input_registers[0], output_registers,
                                     ident + '    ')
        res += ident + '    "%s %%%%%s, (%%0)\\n"\n' % (move_instruction,
                                                        input_registers[0])
        res += ident + '    :: "r"(%s + j) : %s, "memory"\n' % (buf_name,
                                                                clobber)
        res += ident + '  );\n'
        res += ident + '}\n'
        return res
    res = ident + 'for (int j = 0; j < %d; j += %d) {\n' % (n, 1 << to_it)
    res += ident + '  for (int k = 0; k < %d; k += %d) {\n' % (
        1 << (to_it - num_nontrivial_levels), 1 << log_w)
    subcube = []
    for l in range(1 << num_nontrivial_levels):
        subcube.append('j + k + ' + str(l * (1 <<
                                             (to_it - num_nontrivial_levels))))
    res += ident + '    __asm__ volatile (\n'
    for l in range(1 << num_nontrivial_levels):
        res += ident + '      "%s (%%%d), %%%%%s\\n"\n' % (move_instruction, l,
                                                           input_registers[l])
    for it in range(from_it, log_w):
        for ii in range(1 << num_nontrivial_levels):
            res += special_steps[it](input_registers[ii], output_registers,
                                     ident + '      ')
    for it in range(num_nontrivial_levels):
        for ii in range(0, 1 << num_nontrivial_levels, 1 << (it + 1)):
            for jj in range(1 << it):
                res += main_step(input_registers[ii + jj],
                                 input_registers[ii + jj + (1 << it)],
                                 output_registers[ii + jj],
                                 output_registers[ii + jj +
                                                  (1 << it)], ident + '      ')
        tmp = input_registers
        input_registers = output_registers
        output_registers = tmp
    for l in range(1 << num_nontrivial_levels):
        res += ident + '      "%s %%%%%s, (%%%d)\\n"\n' % (
            move_instruction, input_registers[l], l)
    res += ident + '      :: %s : %s, "memory"\n' % (
        ', '.join(['"r"(%s + %s)' % (buf_name, x) for x in subcube]), clobber)
    res += ident + '    );\n'
    res += ident + '  }\n'
    res += ident + '}\n'
    return res


def float_avx_composite_step(buf_name, log_n, from_it, to_it, ident=''):
    return composite_step(buf_name, log_n, from_it, to_it, 3,
                          ['ymm%d' % x for x in range(16)], 'vmovups',
                          [float_avx_0, float_avx_1,
                           float_avx_2], float_avx_3_etc, ident)


def double_avx_composite_step(buf_name, log_n, from_it, to_it, ident=''):
    return composite_step(buf_name, log_n, from_it, to_it, 2, [
        'ymm%d' % x for x in range(16)
    ], 'vmovupd', [double_avx_0, double_avx_1], double_avx_2_etc, ident)


def float_sse_composite_step(buf_name, log_n, from_it, to_it, ident=''):
    return composite_step(buf_name, log_n, from_it, to_it, 2,
                          ['xmm%d' % x for x in range(16)], 'movups',
                          [float_sse_0, float_sse_1], float_sse_2_etc, ident)


def double_sse_composite_step(buf_name, log_n, from_it, to_it, ident=''):
    return composite_step(buf_name, log_n, from_it, to_it, 1,
                          ['xmm%d' % x for x in range(16)], 'movupd',
                          [double_sse_0], double_sse_1_etc, ident)


def plain_unmerged(type_name, log_n):
    signature = "static inline void helper_%s_%d(%s *buf)" % (type_name, log_n,
                                                       type_name)
    res = '%s;\n' % signature
    res += '%s {\n' % signature
    for i in range(log_n):
        res += plain_step(type_name, 'buf', log_n, i, '  ')
    res += "}\n"
    return res


def greedy_merged(type_name, log_n, composite_step):
    try:
        composite_step('buf', log_n, 0, 0)
    except Exception:
        raise Exception('log_n is too small: %d' % log_n)
    signature = 'static inline void helper_%s_%d(%s *buf)' % (type_name, log_n,
                                                       type_name)
    res = '%s;\n' % signature
    res += '%s {\n' % signature
    cur_it = 0
    while cur_it < log_n:
        cur_to_it = log_n
        while True:
            try:
                composite_step('buf', log_n, cur_it, cur_to_it)
                break
            except Exception:
                cur_to_it -= 1
                continue
        res += composite_step('buf', log_n, cur_it, cur_to_it, '  ')
        cur_it = cur_to_it
    res += '}\n'
    return res


def greedy_merged_recursive(type_name, log_n, threshold_step, composite_step):
    if threshold_step > log_n:
        raise Exception('threshold_step must be at most log_n')
    try:
        composite_step('buf', threshold_step, 0, 0)
    except Exception:
        raise Exception('threshold_step is too small: %d' % threshold_step)
    signature = 'void helper_%s_%d_recursive(%s *buf, int depth)' % (type_name,
                                                                     log_n,
                                                                     type_name)
    res = '%s;\n' % signature
    res += '%s {\n' % signature
    res += '  if (depth == %d) {\n' % threshold_step
    cur_it = 0
    while cur_it < threshold_step:
        cur_to_it = threshold_step
        while True:
            try:
                composite_step('buf', threshold_step, cur_it, cur_to_it)
                break
            except Exception:
                cur_to_it -= 1
                continue
        res += composite_step('buf', threshold_step, cur_it, cur_to_it, '    ')
        cur_it = cur_to_it
    res += '    return;\n'
    res += '  }\n'
    cur_it = threshold_step
    while cur_it < log_n:
        cur_to_it = log_n
        while True:
            try:
                composite_step('buf', cur_to_it, cur_it, cur_to_it)
                break
            except Exception:
                cur_to_it -= 1
                continue
        res += '  if (depth == %d) {\n' % cur_to_it
        for i in range(1 << (cur_to_it - cur_it)):
            res += '    helper_%s_%d_recursive(buf + %d, %d);\n' % (
                type_name, log_n, i * (1 << cur_it), cur_it)
        res += composite_step('buf', cur_to_it, cur_it, cur_to_it, '    ')
        res += '    return;\n'
        res += '  }\n'
        cur_it = cur_to_it
    res += '}\n'
    signature = 'void helper_%s_%d(%s *buf)' % (type_name, log_n, type_name)
    res += '%s;\n' % signature
    res += '%s {\n' % signature
    res += '  helper_%s_%d_recursive(buf, %d);\n' % (type_name, log_n, log_n)
    res += '}\n'
    return res


def extract_time(data):
    cpu_time = float(data['cpu_time'])
    time_unit = data['time_unit']
    if time_unit != 'ns':
        raise Exception('nanoseconds expected')
    return cpu_time / 1e9


def get_mean_stddev():
    with open('measurements/output.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        first = True
        for row in reader:
            if first:
                header = row
                first = False
            else:
                data = {}
                for (x, y) in zip(header, row):
                    data[x] = y
                if data['name'] == 'benchmark_fht_mean':
                    mean = extract_time(data)
                elif data['name'] == 'benchmark_fht_stddev':
                    stddev = extract_time(data)
    return mean


def measure_time(code, log_n, type_name, method_name, num_it=3):
    if num_it % 2 == 0:
        raise Exception('even number of runs: %d' % num_it)
    with open('measurements/to_run.h', 'w') as output:
        output.write(code)
        output.write('const int log_n = %d;\n' % log_n)
        signature = 'void run(%s *buf)' % type_name
        output.write('%s;\n' % signature)
        output.write('%s {\n' % signature)
        output.write('  %s(buf);\n' % method_name)
        output.write('}\n')
    with open('/dev/null', 'wb') as devnull:
        code = subprocess.call(
            'cd measurements && make run_%s' % type_name,
            shell=True,
            stdout=devnull)
        if code != 0:
            raise Exception('bad exit code')
        code = subprocess.call(
            './measurements/run_%s --benchmark_repetitions=%d --benchmark_format=csv > ./measurements/output.csv'
            % (type_name, num_it),
            shell=True,
            stderr=devnull)
        if code != 0:
            raise Exception('bad exit code')
    return get_mean_stddev()


if __name__ == '__main__':
    final_code = '#include "fht.h"\n'
    hall_of_fame = []
    for (type_name,
         composite_step_generator) in [('float', float_avx_composite_step),
                                       ('double', double_avx_composite_step)]:
        for log_n in range(1, max_log_n + 1):
            sys.stdout.write('log_n = %d\n' % log_n)
            times = []
            try:
                (res, desc) = (greedy_merged(type_name, log_n,
                                             composite_step_generator),
                               'greedy_merged')
            except Exception:
                (res, desc) = (plain_unmerged(type_name, log_n),
                               'plain_unmerged')
            time = measure_time(res, log_n, type_name,
                                'helper_%s_%d' % (type_name, log_n))
            times.append((time, res, desc))
            sys.stdout.write('log_n = %d; iterative; time = %.10e\n' % (log_n,
                                                                        time))
            for threshold_step in range(1, log_n + 1):
                try:
                    res = greedy_merged_recursive(type_name, log_n,
                                                  threshold_step,
                                                  composite_step_generator)
                    time = measure_time(res, log_n, type_name,
                                        'helper_%s_%d' % (type_name, log_n))
                    times.append(
                        (time, res,
                         'greedy_merged_recursive %d' % threshold_step))
                    sys.stdout.write(
                        'log_n = %d; threshold_step = %d; time = %.10e\n' %
                        (log_n, threshold_step, time))
                except Exception:
                    sys.stdout.write('FAIL: %d\n' % threshold_step)
            (best_time, best_code, best_desc) = min(times)
            hall_of_fame.append((type_name, log_n, best_time, best_desc))
            final_code += best_code
            sys.stdout.write('log_n = %d; best_time = %.10e; %s\n' %
                             (log_n, best_time, best_desc))
        final_code += 'int fht_%s(%s *buf, int log_n) {\n' % (type_name,
                                                              type_name)
        final_code += '  if (log_n == 0) {\n'
        final_code += '    return 0;\n'
        final_code += '  }\n'
        for i in range(1, max_log_n + 1):
            final_code += '  if (log_n == %d) {\n' % i
            final_code += '    helper_%s_%d(buf);\n' % (type_name, i)
            final_code += '    return 0;\n'
            final_code += '  }\n'
        final_code += '  return 1;\n'
        final_code += '}\n'
    with open('fht_avx.c', 'w') as output:
        output.write(final_code)
    sys.stdout.write('hall of fame\n')
    with open('hall_of_fame_avx.txt', 'w') as hof:
        for (type_name, log_n, best_time, best_desc) in hall_of_fame:
            s = 'type_name = %s; log_n = %d; best_time = %.10e; best_desc = %s\n' % (
                type_name, log_n, best_time, best_desc)
            sys.stdout.write(s)
            hof.write(s)
