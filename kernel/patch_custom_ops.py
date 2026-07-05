from pathlib import Path

target = Path("/opt/venv/lib/python3.12/site-packages/vllm/_custom_ops.py")
text = target.read_text()

marker = "current_platform.import_kernels()\n"
hook = r'''

def _load_nvfp4_cache_writer_extension() -> None:
    try:
        if hasattr(torch.ops._C_cache_ops, "concat_and_cache_nvfp4_mla"):
            return
    except AttributeError:
        pass

    path = "/opt/nvfp4_extra/_C_nvfp4.so"
    try:
        torch.ops.load_library(path)
    except OSError as exc:
        if "libcuda.so.1" in str(exc):
            logger.warning("Skipping NVFP4 cache writer extension: %s", exc)
            return
        raise


_load_nvfp4_cache_writer_extension()
'''

if "_load_nvfp4_cache_writer_extension" not in text:
    if marker not in text:
        raise SystemExit(f"Could not find insertion marker in {target}")
    text = text.replace(marker, marker + hook, 1)
    target.write_text(text)
