"""Detect what this machine can actually run, and recommend a Whisper model for it.

No new dependencies: CUDA presence and supported compute types come from ctranslate2
(already a faster-whisper dependency), VRAM from nvidia-smi when it is on PATH, and RAM
from the OS.

The numbers in MODEL_CATALOGUE are measured on this project's corpus (1,441 calls of
8 kHz telephone audio, ~58 s average) on a 12-core CPU, not vendor claims.
"""
import os, shutil, subprocess

# rough resident memory per model at int8, plus the relative speed we actually observed
MODEL_CATALOGUE = [
    {"id": "tiny",     "label": "Tiny",        "params": "39M",  "ram_gb": 0.4,
     "speed": "fastest", "quality": "rough drafts only",
     "note": "Loses proper nouns and numbers on telephone audio."},
    {"id": "base",     "label": "Base",        "params": "74M",  "ram_gb": 0.6,
     "speed": "very fast", "quality": "low",
     "note": "Usable when CPU is the hard constraint."},
    {"id": "small",    "label": "Small",       "params": "244M", "ram_gb": 1.0,
     "speed": "~0.5x realtime on CPU", "quality": "good",
     "note": "Project default. ~28 s per 58 s call on 12 CPU cores at int8."},
    {"id": "medium",   "label": "Medium",      "params": "769M", "ram_gb": 2.5,
     "speed": "~2x slower than small", "quality": "better",
     "note": "Noticeably better on banking terms and names. Worth it on a GPU."},
    {"id": "large-v3", "label": "Large v3",    "params": "1.55B", "ram_gb": 4.5,
     "speed": "~6x slower than small on CPU", "quality": "best",
     "note": "Practical on a GPU. On CPU the full 1,441-call batch runs for days."},
    {"id": "deepdml/faster-whisper-large-v3-turbo-ct2",
     "label": "Large v3 Turbo", "params": "809M", "ram_gb": 3.0,
     "speed": "2-3x slower than small on CPU", "quality": "mixed on 8 kHz",
     "note": "Tested on this corpus: slower AND less stable than small -- its trimmed "
             "4-layer decoder garbled one caller's turns into a run-on. Prefer large-v3."},
]
MODEL_IDS = [m["id"] for m in MODEL_CATALOGUE]


def _total_ram_gb():
    try:
        if os.name == "nt":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            m = _MemStatus()
            m.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.ullTotalPhys / 1024 ** 3, 1), round(m.ullAvailPhys / 1024 ** 3, 1)
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        return round(total / 1024 ** 3, 1), round(avail / 1024 ** 3, 1)
    except Exception:
        return None, None


def _gpus():
    """GPU name + VRAM via nvidia-smi. Absent driver simply means no CUDA."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8)
        gpus = []
        for line in out.stdout.strip().splitlines():
            name, _, mem = line.partition(",")
            if name.strip():
                gpus.append({"name": name.strip(),
                             "vram_gb": round(int(mem.strip()) / 1024, 1) if mem.strip().isdigit() else None})
        return gpus
    except Exception:
        return []


def detect():
    """What this machine offers. Never raises -- unknown fields come back None."""
    try:
        import ctranslate2
        cuda_count = ctranslate2.get_cuda_device_count()
        cpu_compute = sorted(ctranslate2.get_supported_compute_types("cpu"))
    except Exception:
        cuda_count, cpu_compute = 0, ["int8", "float32"]
    total_ram, avail_ram = _total_ram_gb()
    gpus = _gpus()
    return {
        "cuda_devices": cuda_count,
        "gpus": gpus,
        "vram_gb": gpus[0]["vram_gb"] if gpus and gpus[0].get("vram_gb") else None,
        "cpu_cores": os.cpu_count(),
        "ram_total_gb": total_ram,
        "ram_available_gb": avail_ram,
        "cpu_compute_types": cpu_compute,
    }


def recommend(hw=None):
    """Pick a model/device/compute for this machine, and say why.

    Deliberately conservative on CPU: transcription is the pipeline's cost centre, and a
    model that swaps is slower than a smaller one that fits.
    """
    hw = hw or detect()
    cores = hw.get("cpu_cores") or 4
    avail = hw.get("ram_available_gb")
    vram = hw.get("vram_gb")
    warnings = []

    if hw.get("cuda_devices"):
        device, compute = "cuda", "float16"
        if vram is None or vram >= 10:
            model, why = "large-v3", "A CUDA GPU makes the most accurate model practical."
        elif vram >= 5:
            model, why = "medium", f"{vram} GB VRAM fits medium comfortably; large-v3 would not."
            warnings.append("large-v3 needs roughly 10 GB VRAM at float16.")
        else:
            model, why = "small", f"Only {vram} GB VRAM -- small keeps it on the GPU."
    else:
        device, compute = "cpu", "int8"
        if cores <= 3:
            model, why = "base", f"Only {cores} CPU cores; anything larger will crawl."
        elif avail is not None and avail < 2.5:
            model, why = "small", (f"{cores} cores is plenty, but only {avail} GB RAM is free "
                                   "-- a bigger model would swap and end up slower.")
            warnings.append(f"Just {avail} GB RAM free. Close other apps before a large batch.")
        elif cores >= 8 and (avail is None or avail >= 6):
            model, why = "small", ("Small is the best speed/quality trade-off on CPU. "
                                  "Medium is available if you can spare ~2x the time.")
        else:
            model, why = "small", "Small is the safe default for CPU transcription."
        warnings.append("No CUDA GPU detected, so transcription runs on CPU "
                        "(~28 s per one-minute call at small/int8).")

    return {
        "model": model, "device": device, "compute_type": compute,
        "reason": why, "warnings": warnings,
    }
