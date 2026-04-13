# syntax=docker/dockerfile:1.4
FROM vllm/vllm-openai:cu130-nightly-9dd5ee0117be1b891d1ddc4e5cd7d03071ec11ae

# curl for debugging; python3-tk if you need matplotlib; git because
# editable install often pulls version info from git.
# Added cmake, build-essential, and ninja-build to compile vllm extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  python3-tk \
  git \
  cmake \
  build-essential \
  ninja-build \
&& rm -rf /var/lib/apt/lists/*

# === PyCharm remote-debug shim ==============================================
# PyCharm bind-mounts the host project at /opt/project and prepends it to
# PYTHONPATH, so `import vllm` resolves to /opt/project/vllm (the live source
# tree). That tree has no compiled C extensions (vllm/_C*.so), so imports
# fail with `ModuleNotFoundError: No module named 'vllm._C'`. At Python
# startup a .pth hook symlinks the compiled artifacts from the image's
# installed vllm into /opt/project/vllm so the mounted sources can load them.
RUN /usr/bin/python3 -c "import vllm, os; print(os.path.dirname(os.path.abspath(vllm.__file__)))" > /.vllm_installed_path

RUN <<'SETUP'
set -e
SITE=$(/usr/bin/python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")
cat > "$SITE/vllm_debug_setup.py" <<'PYFILE'
import os
try:
    _installed = open("/.vllm_installed_path").read().strip()
    _mounted = "/opt/project/vllm"
    if (_installed and _installed != _mounted
            and os.path.isdir(_installed) and os.path.isdir(_mounted)):
        for _root, _dirs, _files in os.walk(_installed):
            for _f in _files:
                if not _f.endswith((".so", ".pyd")):
                    continue
                _src = os.path.join(_root, _f)
                _rel = os.path.relpath(_src, _installed)
                _dst = os.path.join(_mounted, _rel)
                if os.path.islink(_dst) and not os.path.exists(_dst):
                    try:
                        os.remove(_dst)
                    except OSError:
                        pass
                if os.path.lexists(_dst):
                    continue
                try:
                    os.makedirs(os.path.dirname(_dst), exist_ok=True)
                    os.symlink(_src, _dst)
                except OSError:
                    pass
except Exception:
    pass
PYFILE
echo "import vllm_debug_setup" > "$SITE/zzz_vllm_debug.pth"
SETUP