"""Download pinned official weights; optionally restore Android voice assets."""
import argparse
import hashlib
import os
import secrets
import shutil
import urllib.request
import zipfile
from pathlib import Path
from model_catalog import MODELS, PLATE_FILES, PLATE_MODELS, PPOCR_KEYS, PPOCR_KEYS_SHA256, verify, verify_text

ROOT = Path(__file__).resolve().parent
RELEASE = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/'
MIRROR = 'https://huggingface.co/hr16/yolox-onnx/resolve/main/'


def download(target, urls, digest):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        verify(target, digest)
        return target
    urls = [urls] if isinstance(urls, str) else list(urls)
    last = None
    for url in urls:
        partial = target.with_suffix('.part')
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'cam-model-download'})
            with urllib.request.urlopen(req, timeout=600) as response, partial.open('wb') as output:
                shutil.copyfileobj(response, output)
            verify(partial, digest)
            partial.replace(target)
            return target
        except Exception as error:
            last = error
            partial.unlink(missing_ok=True)
    raise last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--android-voice', action='store_true')
    args = parser.parse_args()
    for spec in MODELS.values():
        download(ROOT/'models'/spec['file'],
            [RELEASE+spec['file'], MIRROR+spec['file']], spec['sha256'])
    if not all((ROOT/'models/plate'/name).exists() for name in PLATE_FILES):
        # Upstream SDK publishes this HTTP archive. Both archive and extracted
        # files must match fixed hashes; unverified weights are never installed.
        archive = download(ROOT/'validation/hyperlpr-models.zip',
            'http://hyperlpr.tunm.top/raw/20230229.zip',
            'ce1cb895dc754a1bf6b50f99f4d745c0a8e1bcd5fecd02ca9b66ad7c24dae15e')
        with zipfile.ZipFile(archive) as source:
            for name, digest in PLATE_FILES.items():
                content = source.read('20230229/onnx/'+name)
                if hashlib.sha256(content).hexdigest() != digest:
                    raise ValueError('Plate model checksum mismatch: '+name)
                target = ROOT/'models/plate'/name
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(content)
    for name, digest in PLATE_FILES.items():
        verify(ROOT/'models/plate'/name,digest)
    plate_dir = ROOT/'models/plate'
    if (plate_dir/PPOCR_KEYS).exists():
        verify_text(plate_dir/PPOCR_KEYS, PPOCR_KEYS_SHA256)
    for spec in PLATE_MODELS.values():
        if spec.get('urls'):
            download(plate_dir/spec['file'], spec['urls'], spec['sha256'])
        if spec.get('det_urls'):
            download(plate_dir/spec['det'], spec['det_urls'], spec['det_sha256'])
    notice = plate_dir/'PADDLEOCR-NOTICE.txt'
    if not notice.exists():
        notice.write_text(
            'PP-OCRv5 recognition ONNX: PaddlePaddle / PaddleOCR, Apache-2.0\n'
            'https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec_onnx\n'
            'https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec_onnx\n',
            encoding='utf-8')
    onnxocr_notice = plate_dir/'ONNXOCR-NOTICE.txt'
    if not onnxocr_notice.exists():
        onnxocr_notice.write_text(
            'OnnxOCR license-plate ONNX: jingsongliu/onnxocr_model, Apache-2.0\n'
            'https://huggingface.co/jingsongliu/onnxocr_model\n'
            'https://github.com/jingsongliujing/OnnxOCR\n',
            encoding='utf-8')
    if args.android_voice:
        archive=download(ROOT/'validation/vosk-model-small-cn-0.22.zip',
            'https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip',
            '3af8b0e7e0f835ae9d414ce5df580237a3cfb08d586c9fbbb0f7ff29ad5b14ba')
        asset=(ROOT.parent/'IllegalCapture/app/src/main/assets/model-cn').resolve()
        with zipfile.ZipFile(archive) as source:
            for name in source.namelist():
                parts=Path(name).parts[1:]
                if not parts or name.endswith('/'): continue
                target=asset.joinpath(*parts).resolve()
                if not target.is_relative_to(asset): raise ValueError('Unsafe archive path')
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(source.read(name))
    env=ROOT/'.env'
    if not env.exists():
        env.write_text('TRAFFIC_API_TOKEN='+secrets.token_urlsafe(32)+'\n',encoding='utf-8')
        os.chmod(env,0o600)
    print('All requested models verified. Local token is in backend/.env (keep private).')


if __name__=='__main__': main()
