import os

# Handle case when PROFILE_TREND environment variable is not set
profile_trend = os.getenv('PROFILE_TREND', '0')
try:
    profile_trend = int(profile_trend)
except ValueError:
    profile_trend = 0

if profile_trend == 1:
    FLAG_DICT = {
        'stencil_size': ['box2d2r', 'box2d1r'],
        'cuda_compute': ['tensor'],
        'mesh_size': [160, 320] + [(16 * 10 * i * 4) for i in range(1, 17)],
        'tile_size': list(range(1, 16))
    }
elif profile_trend == 2:
    FLAG_DICT = {
        'stencil_size': ['box2d2r'],
        'cuda_compute': ['tensor'],
        'mesh_size': [2 ** i * 640 for i in range(0, 5)],
        'tile_size': list(range(1, 16))
    }
else:
    FLAG_DICT = {
        'stencil_size': ['box2d2r', 'star2d2r', 'box2d1r', 'star2d1r'],
        # 'cuda_compute': ['baseline', 'tensor'],
        'cuda_compute': ['tensor'],
        'mesh_size': [(16 * 10 * i) for i in {64}],
        'tile_size': list(range(1, 16))
    }


TENSOR_DIR = './layout16/'
BASELINE_DIR = './baseline/'
COMMON_DIR = './common/'
BINARY_DIR = './bin_ncu/'

COMMON_SOURCE = {
    'box2d2r': f"{COMMON_DIR}main_2r.cu {COMMON_DIR}gold_box2d2r.cpp",
    'star2d2r': f"{COMMON_DIR}main_2r.cu {COMMON_DIR}gold_star2d2r.cpp",
    'box2d1r': f"{COMMON_DIR}main_1r.cu {COMMON_DIR}gold_box2d1r.cpp",
    'star2d1r': f"{COMMON_DIR}main_1r.cu {COMMON_DIR}gold_star2d1r.cpp",
}

BASELINE_SOURCE = {
    'box2d2r': f"{BASELINE_DIR}host_box2d2r.cu {BASELINE_DIR}kernel_box2d2r.cu",
    'star2d2r': f"{BASELINE_DIR}host_star2d2r.cu {BASELINE_DIR}kernel_star2d2r.cu",
    'box2d1r': f"{BASELINE_DIR}host_box2d1r.cu {BASELINE_DIR}kernel_box2d1r.cu",
    'star2d1r': f"{BASELINE_DIR}host_star2d1r.cu {BASELINE_DIR}kernel_star2d1r.cu",
}

TENSOR_SOURCE = {
    'box2d2r': f"{TENSOR_DIR}host_box2d2r.cu {TENSOR_DIR}kernel_box2d2r.cu",
    'star2d2r': f"{TENSOR_DIR}host_star2d2r.cu {TENSOR_DIR}kernel_star2d2r.cu",
    'box2d1r': f"{TENSOR_DIR}host_box2d1r.cu {TENSOR_DIR}kernel_box2d1r.cu",
    'star2d1r': f"{TENSOR_DIR}host_star2d1r.cu {TENSOR_DIR}kernel_star2d1r.cu",
}