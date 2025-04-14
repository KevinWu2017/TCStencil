from multiprocessing import Pool
import itertools, subprocess, argparse
from param import FLAG_DICT, COMMON_SOURCE, BASELINE_SOURCE, TENSOR_SOURCE, BINARY_DIR

import os

# parser = argparse.ArgumentParser()
# parser.add_argument('--arch', type=str, default='A100')
# args = parser.parse_args()

BASE_FLAGS = ''
gpu_name = ''
try:
    # 调用 nvidia-smi 获取 GPU 名称和计算能力
    output = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        universal_newlines=True
    ).strip()
    gpu_name, gpu_cc = [s.strip() for s in output.split(',')]
    gpu_name = gpu_name.replace(' ', '_')
    print(f"GPU Name: {gpu_name}")
    print(f"GPU Compute Capability: {gpu_cc}")
    # 根据计算能力设置 CUDA_COMPUTE_CAPABILITY
    if gpu_cc == "8.0":
        BASE_FLAGS = '-O3 -arch sm_80 -DRUN_TIMES=1'
    elif gpu_cc == "8.6":
        BASE_FLAGS = '-O3 -arch sm_70 -DRUN_TIMES=1'
    elif gpu_cc == "8.9":
        BASE_FLAGS = '-O3 -arch sm_89 -DRUN_TIMES=1'
    elif gpu_cc == "9.0":
        BASE_FLAGS = '-O3 -arch sm_86 -DRUN_TIMES=1'
    else:
        raise RuntimeError(f"Unsupported GPU compute capability: {gpu_cc}")
except Exception as e:
    raise RuntimeError(f"Failed to detect GPU compute capability: {e}")

# 创建输出目录
OUTPUT_DIR = f'./data/{gpu_name}/layout16/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")
else:
    print(f"Directory already exists: {OUTPUT_DIR}")

# BASE_FLAGS = '-O3 -arch sm_80 -DRUN_TIMES=1' if args.arch == 'A100' else '-O3 -arch sm_70 -DRUN_TIMES=1'


def compile_command_gen(cuda_compute, stencil_size, mesh_size, tile_size):
    cmd = ' '.join(['nvcc', BASE_FLAGS, f'-DMESH_SIZE={mesh_size},TILE_SIZE={tile_size}'])
    output_file_name = f'{cuda_compute}_{stencil_size}_{mesh_size}_{tile_size}'
    if cuda_compute == 'baseline':
        cmd = ' '.join([cmd, '-o', BINARY_DIR + output_file_name,
                        COMMON_SOURCE[stencil_size],
                        BASELINE_SOURCE[stencil_size],
                        ])
    else:
        cmd = ' '.join([cmd, '-o', BINARY_DIR + output_file_name,
                        COMMON_SOURCE[stencil_size],
                        TENSOR_SOURCE[stencil_size],
                        ])
    

    return cmd, output_file_name

def exec_cmd(cmd, wait=False):
    p = subprocess.Popen(cmd.split(' '))
    if wait:
        p.wait()

def exec_cmd_list_async(cmd_list):
    n_processors = os.cpu_count()
    pool = Pool(n_processors)
    for cmd in cmd_list:
        pool.apply_async(exec_cmd, args=(cmd, True))
    pool.close()
    pool.join()


def run_file_serial(file_list):
    example_cmd = 'bash script/run_file_with_ncu.sh {file_name} {output_file} {BIN_DIR}'

    for index, file_name in enumerate(file_list):
        print('{index}/{total} {file_name}'.format(index=index,
                                                   total=len(file_list), file_name=file_name))
        cmd = example_cmd.format(
            file_name=file_name, output_file='./data/{}/layout16/ncu_result.txt'.format(gpu_name),
            BIN_DIR=BINARY_DIR)
        proc = subprocess.Popen(cmd.split(' '))
        proc.wait()


if __name__ == '__main__':
    compile_cmd_list = []
    exec_files_list = []
    for cuda_compute, stencil_size, mesh_size, tile_size in itertools.product(FLAG_DICT['cuda_compute'], FLAG_DICT['stencil_size'], 
                                                                            FLAG_DICT['mesh_size'], FLAG_DICT['tile_size']):
        if (mesh_size // 16) % tile_size != 0:
            continue
        cmd, output_file_name = compile_command_gen(cuda_compute, stencil_size, mesh_size, tile_size)
        compile_cmd_list.append(cmd)
        exec_files_list.append(output_file_name)

    exec_cmd_list_async(compile_cmd_list)
    run_file_serial(exec_files_list)
