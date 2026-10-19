import torch

def get_device():
    # picks GPU if available, otherwise CPU, and always prints which one
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print("using GPU:", name)
        return "cuda"
    else:
        print("using CPU (no GPU found)")
        return "cpu"

if __name__ == "__main__":
    get_device()
