from pathlib import Path
import hashlib

MODELS = {
    'yolox_tiny': {'name': 'YOLOX-Tiny · 速度优先', 'file': 'yolox_tiny.onnx', 'size': 416,
                   'sha256': '427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7'},
    'yolox_s': {'name': 'YOLOX-S · 均衡', 'file': 'yolox_s.onnx', 'size': 640,
                'sha256': 'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'},
    'yolox_m': {'name': 'YOLOX-M · 精度优先', 'file': 'yolox_m.onnx', 'size': 640,
                'sha256': '21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f'},
    'yolox_l': {'name': 'YOLOX-L · 最高精度', 'file': 'yolox_l.onnx', 'size': 640,
                'sha256': '7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411'},
}

PLATE_FILES = {
    'y5fu_640x_sim.onnx': '0306de937471b87f56eb3f5620815e7e7058f8ab7428e0734fc64627cc4d716c',
    'rpv3_mdict_160_r3.onnx': '8fb08b5db2adeccf43b05006bbbf409e4659d08d72e46a62631c00ff751eaeb3',
}

# HuggingFace git-lfs SHA-256 of official PaddleOCR PP-OCRv5 rec ONNX (Apache-2.0).
PPOCR_KEYS = 'ppocrv5_keys.txt'
PPOCR_KEYS_SHA256 = '90793e0d61fac33e2efd24a91d9b53af4af85e19a0acea9e534274c37312e179'
ONNXOCR_HF = 'https://huggingface.co/jingsongliu/onnxocr_model/resolve/main/models/license_plate/'
ONNXOCR_MIRROR = 'https://hf-mirror.com/jingsongliu/onnxocr_model/resolve/main/models/license_plate/'
PLATE_MODELS = {
    'hyperlpr3': {'name': 'HyperLPR3 20230229 · 车牌专用', 'kind': 'hyperlpr'},
    'ppocrv5_mobile': {
        'name': 'PP-OCRv5 Mobile · 2025 识别', 'kind': 'ppocr',
        'file': 'ppocrv5_mobile_rec.onnx',
        'sha256': 'da72dc72ca4dc220df0dfde68c1dedc31c58d3e76a25871122e5056227d50092',
        'urls': [
            'https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec_onnx/resolve/main/inference.onnx',
            'https://hf-mirror.com/PaddlePaddle/PP-OCRv5_mobile_rec_onnx/resolve/main/inference.onnx',
        ],
    },
    'ppocrv5_server': {
        'name': 'PP-OCRv5 Server · 2025 高精度识别', 'kind': 'ppocr',
        'file': 'ppocrv5_server_rec.onnx',
        'sha256': 'd9dc333c9c7b042c6dffb8e33d72b6f65c9c1d463d0a3c2f78174fea55e94752',
        'urls': [
            'https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec_onnx/resolve/main/inference.onnx',
            'https://hf-mirror.com/PaddlePaddle/PP-OCRv5_server_rec_onnx/resolve/main/inference.onnx',
        ],
    },
    'onnxocr_plate': {
        'name': 'OnnxOCR 车牌专用 · 2026 检测+识别', 'kind': 'onnxocr',
        'file': 'onnxocr_plate_rec.onnx',
        'sha256': '71ae808d441bb2975bc7abd913a3d325b32344071ed2217d8efedccd12ba8799',
        'urls': [ONNXOCR_HF+'plate_rec.onnx', ONNXOCR_MIRROR+'plate_rec.onnx'],
        'det': 'onnxocr_plate_det.onnx',
        'det_sha256': '1a52f932768925c525014f3ff09769d4e1901763417f57e62c6fb20837843736',
        'det_urls': [ONNXOCR_HF+'car_plate_detect.onnx', ONNXOCR_MIRROR+'car_plate_detect.onnx'],
    },
}


def model_directory(settings):
    return settings.model.parent


def catalog(settings):
    root = model_directory(settings)
    return [{'id': key, **value, 'installed': (root/value['file']).is_file()} for key, value in MODELS.items()]


def plate_installed(root, rec_id):
    spec = PLATE_MODELS[rec_id]
    if spec['kind'] == 'hyperlpr':
        return all((root / name).is_file() for name in PLATE_FILES)
    if spec['kind'] == 'onnxocr':
        return (root / spec['det']).is_file() and (root / spec['file']).is_file()
    det = root / 'y5fu_640x_sim.onnx'
    return det.is_file() and (root / spec['file']).is_file() and (root / PPOCR_KEYS).is_file()


def plate_catalog(settings):
    root = model_directory(settings) / 'plate'
    return [{'id': key, 'name': spec['name'], 'installed': plate_installed(root, key)}
            for key, spec in PLATE_MODELS.items()]


def verify(path: Path, expected):
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise ValueError(f'Model checksum mismatch: {path.name}')


def verify_text(path: Path, expected):
    digest = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    if digest != expected:
        raise ValueError(f'Model checksum mismatch: {path.name}')
