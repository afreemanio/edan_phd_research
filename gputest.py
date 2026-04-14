def test_pytorch():
    print("=== PyTorch GPU Test ===")
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        if cuda_available:
            device = torch.device("cuda")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")

            # simple tensor test
            x = torch.rand(3, 3).to(device)
            y = torch.rand(3, 3).to(device)
            z = x @ y

            print("Tensor device:", z.device)
            print("Matrix multiply result:\n", z)
        else:
            print("Running on CPU only.")

    except Exception as e:
        print("PyTorch test failed:", e)


def test_jax():
    print("\n=== JAX GPU Test ===")
    try:
        import jax
        import jax.numpy as jnp

        print(f"JAX version: {jax.__version__}")

        devices = jax.devices()
        print("Available devices:", devices)

        gpu_devices = [d for d in devices if d.platform in ("gpu", "cuda")]

        if gpu_devices:
            print("Using GPU device:", gpu_devices[0])

            x = jnp.ones((3, 3))
            y = jnp.ones((3, 3))
            z = jnp.dot(x, y)

            print("Result device:", z.device)
            print("Matrix multiply result:\n", z)
        else:
            print("No GPU detected, running on:", devices[0].platform)

    except Exception as e:
        print("JAX test failed:", e)


if __name__ == "__main__":
    test_pytorch()
    test_jax()