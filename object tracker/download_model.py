"""
Downloads the YOLOv8n ONNX model.
Tries multiple known public URLs as fallbacks.
"""
import os
import sys
import urllib.request

URLS = [
    "https://huggingface.co/qualcomm/YOLOv8-Nano/resolve/main/YOLOv8-Nano.onnx",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx",
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx",
]

def download(url, dest):
    print(f"  Trying: {url}")
    def progress(block, block_size, total):
        done = block * block_size
        if total > 0:
            pct = min(100, done * 100 / total)
            sys.stdout.write(f"\r  Progress: {pct:.1f}%  ({done:,}/{total:,} bytes)")
        else:
            sys.stdout.write(f"\r  Downloaded {done:,} bytes")
        sys.stdout.flush()
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()

def main():
    target_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(target_dir, "yolov8n.onnx")

    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model already exists at {model_path} ({size_mb:.1f} MB)")
        return

    print("Downloading YOLOv8n ONNX model...")
    for url in URLS:
        try:
            download(url, model_path)
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            if size_mb < 1:
                print(f"  File too small ({size_mb:.2f} MB), likely not the real model. Removing.")
                os.remove(model_path)
                continue
            print(f"Download successful! ({size_mb:.1f} MB)")
            return
        except Exception as e:
            print(f"  Failed: {e}")
            if os.path.exists(model_path):
                os.remove(model_path)

    print("\nAll download URLs failed.")
    print("Please install ultralytics and export manually:")
    print("  pip install ultralytics")
    print("  yolo export model=yolov8n.pt format=onnx opset=12")
    sys.exit(1)

if __name__ == "__main__":
    main()
