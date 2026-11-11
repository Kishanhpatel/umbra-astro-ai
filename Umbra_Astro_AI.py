#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Umbra Astro AI (BETA), Copyright (c) 2026 Kishan Patel. Own code; no code taken from other scripts.
"""Umbra Astro AI: Deblur, Denoise, Starless and Gradient in one window (pick the tool at the top). Same processing code as the four single scripts.
Preview or Apply; Apply replaces the image in Siril (Undo available). One model is kept in memory at a time.
Run in Siril (Scripts menu), or standalone:
  python Umbra_Astro_AI.py --tool deblur|denoise|starless|gradient --image in.fits --out out.fits [--model PATH.onnx] [tool options, see --help]
  python Umbra_Astro_AI.py --ui-test        (opens the window on a fake image, runs Preview once, exits)
  python Umbra_Astro_AI.py --check-updates | --download [MODEL_ID|all]"""
VERSION = '0.1.0'
# First release of the combined script. Models go to the same folders as the single scripts ("Umbra Deblur", "Umbra Denoise", "Umbra Starless", "Umbra Gradient").
# ============================================================================================ shared code (src/common.py)
# Everything the five scripts share: dependency install, ONNX Runtime sessions, updater, preferences, Siril I/O, family UI, CLI.
import sys, os, json, hashlib, argparse, time, math, re, types
from pathlib import Path
_STANDALONE = any(a in ('--image', '--ui-test', '--download', '--check-updates') or a.startswith(('--image=', '--download=')) for a in sys.argv)
if __name__ == '__main__' and not _STANDALONE and '--help' not in sys.argv and '-h' not in sys.argv:
    import sirilpy as _siril_deps
    def _ort_requirement():
        """onnxruntime variant for this machine (all variants import as 'onnxruntime'; only one may be installed). An existing one is kept."""
        import importlib.metadata as _md, platform as _pf, shutil as _sh
        for n in ('onnxruntime-gpu', 'onnxruntime-directml', 'onnxruntime-rocm', 'onnxruntime'):
            try:
                _md.version(n); return n
            except Exception:
                pass
        if _pf.system() == 'Windows': return 'onnxruntime-directml'          # GPU through DirectX 12 (NVIDIA / AMD / Intel); CUDA needs onnxruntime-gpu
        if _pf.system() == 'Linux' and _sh.which('nvidia-smi'): return 'onnxruntime-gpu'   # needs the CUDA 12 + cuDNN 9 libraries; falls back to CPU without them
        return 'onnxruntime'                                                   # macOS: Core ML is built in; Linux without NVIDIA: CPU
    _siril_deps.ensure_installed('numpy', _ort_requirement(), 'PyQt6', 'scipy', 'astropy')
import numpy as np
import urllib.request

SCRIPT_DIR = Path(__file__).resolve().parent
ENGINES = {}      # tool key -> engine namespace (filled by the build)
TOOLS = {}        # tool key -> ToolSpec (filled by the build)
USER_AGENT = 'UmbraAstro'    # generic client name only; nothing about the user or the machine is sent
_REDIRECT_HOSTS = {'github.com', 'objects.githubusercontent.com', 'release-assets.githubusercontent.com', 'raw.githubusercontent.com'}
GITHUB_USER = 'Kishanhpatel'


class ToolSpec:
    """Everything that differs between the tools (the shared code reads only this)."""
    def __init__(self, **kw):
        self.legacy_repos = ()
        self.fallback_folder = None
        self.labels = {}
        self.default_stems = ()
        self.notes = []
        self.linear_only = True
        self.preview_whole = False       # Gradient: preview runs on the whole image
        self.reuse_preview = False       # Gradient: Apply with unchanged settings reuses the preview result
        self.alt = None                  # (button text, info key) second preview layer
        self.progress_unit = 'tile'
        self.__dict__.update(kw)

    @property
    def title(self):
        return f'{self.name} (BETA)'

    # hooks (overridden by the tool modules)
    def build_panel(self, ui, layout, host):
        raise NotImplementedError
    def add_cli(self, ap):
        pass
    def params_from_args(self, a):
        return {}
    def image_params(self, header_text):
        return {}
    def preview_params(self, params):
        return params
    def label(self, params):
        return ''
    def output(self, params, out, info):
        return out
    def post_apply(self, ctx, params, info):
        return ''
    def check_cli(self, a, ap):
        pass
    def post_cli(self, a, dst, info, header, label):
        return ''


def env(spec, name, default=None):
    return os.environ.get(spec.prefix + name, default)


def offline_mode(spec):
    return bool(env(spec, 'OFFLINE') or os.environ.get('UMBRA_OFFLINE'))


# ----------------------------------------------------------------------------------------------- models folder
def _onnx_files(d):
    try:
        return sorted(Path(d).glob('*.onnx')) if Path(d).is_dir() else []
    except OSError:
        return []


def model_dir(spec):
    """(folder, used_fallback). <PREFIX>MODEL_DIR wins; else '<folder>' next to the script; Deblur falls back to the old
    'Umbra Deconvolve' folder when its own is missing or empty and the old one has models."""
    e = env(spec, 'MODEL_DIR')
    if e:
        return Path(e), False
    main = SCRIPT_DIR / spec.folder
    if spec.fallback_folder and not _onnx_files(main) and _onnx_files(SCRIPT_DIR / spec.fallback_folder):
        return SCRIPT_DIR / spec.fallback_folder, True
    return main, False


def download_dir(spec):
    e = env(spec, 'MODEL_DIR')
    return Path(e) if e else SCRIPT_DIR / spec.folder


_MANIFEST_NAMES = {}      # (tool, file stem) -> name from the manifest (after an update check)


def model_label(spec, stem):
    return spec.labels.get(stem) or _MANIFEST_NAMES.get((spec.key, stem)) or stem


def list_models(spec):
    items = [(model_label(spec, f.stem), str(f)) for f in _onnx_files(model_dir(spec)[0])]
    items.sort(key=lambda it: (0 if Path(it[1]).stem in spec.default_stems else 1, it[0]))
    return items


def default_model_path(spec, saved=None):
    e = env(spec, 'MODEL')
    if e:
        return e
    d = model_dir(spec)[0]
    if saved:
        p = d / Path(saved).name
        if p.suffix == '.onnx' and p.is_file():
            return str(p)
    items = list_models(spec)
    return items[0][1] if items else None


# ----------------------------------------------------------------------------------------------- preferences
class Prefs:
    """Umbra_<Tool>_prefs.json: {"model": <file name>, "settings": {...}}; combined: {"tool": key, "tools": {key: {...}}}."""
    def __init__(self, path, combined=False, enabled=True):
        self.path, self.combined, self.enabled = Path(path), combined, enabled

    def _read(self):
        try:
            d = json.loads(self.path.read_text())
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def load(self, key):
        d = self._read()
        if self.combined:
            d = (d.get('tools') or {}).get(key) or {}
        s = d.get('settings')
        return (d.get('model') if isinstance(d.get('model'), str) else None), (s if isinstance(s, dict) else {})

    def load_tool(self):
        t = self._read().get('tool')
        return t if isinstance(t, str) else None

    def save(self, key, model, settings, current=None):
        if not self.enabled:
            return
        try:
            entry = {'model': Path(model).name if model else None, 'settings': settings}
            if self.combined:
                d = self._read()
                tools = d.get('tools') if isinstance(d.get('tools'), dict) else {}
                tools[key] = entry
                out = {'tool': current or key, 'tools': tools}
            else:
                out = entry
            self.path.write_text(json.dumps(out, indent=1))
        except Exception:
            pass


# ----------------------------------------------------------------------------------------------- ONNX Runtime
_IMPORT_ERROR = {}   # first onnxruntime / scipy import failure (remembered so every later attempt shows the real cause)


def log_file(spec):
    return SCRIPT_DIR / f'{spec.script}_error.log'


def write_error_log(spec, text):
    try:
        log_file(spec).write_text(f'{time.ctime()}\npython {sys.executable}\n\n{text}')
    except Exception:
        pass


def import_engine(spec):
    """Import onnxruntime and scipy exactly once; on failure keep the ORIGINAL error (with traceback) for every later attempt."""
    if _IMPORT_ERROR:
        raise RuntimeError(_IMPORT_ERROR['msg'])
    try:   # Siril's working folder may have been deleted; some imports then fail in os.getcwd() -> move to a folder that exists
        os.getcwd()
    except OSError:
        os.chdir(str(SCRIPT_DIR))
    try:
        import onnxruntime  # noqa: F401
        import scipy.ndimage  # noqa: F401
    except BaseException as e:
        import traceback
        tb = traceback.format_exc()
        _IMPORT_ERROR['msg'] = f'{type(e).__name__}: {e} (full details: {log_file(spec)})'
        write_error_log(spec, f'sys.path {sys.path}\n\n{tb}')
        print(tb, flush=True)
        raise RuntimeError(_IMPORT_ERROR['msg']) from e


def make_session_factory(prefix):
    """-> factory(model_path, feed) -> (run_fn, provider_label, note).  run_fn(feed_dict) -> list of outputs (like sess.run(None, feed)).
    GPU providers are tried in order of preference for the platform (CUDA, DirectML, ROCm, Core ML); each is checked against the CPU
    provider on one probe feed (outputs must agree to 1e-3 relative) and the first that works is used. Anything missing, failing or
    disagreeing falls through to the next, and finally to the CPU.  <PREFIX>CPU=1 forces the CPU."""
    def factory(model_path, feed):
        import onnxruntime as ort
        path = str(model_path)
        so = ort.SessionOptions(); so.log_severity_level = 3
        holder = []
        def mk_cpu():
            if not holder:
                holder.append(ort.InferenceSession(path, so, providers=['CPUExecutionProvider']))
            return holder[0]
        note = None
        sess, label = None, 'CPU'
        if not os.environ.get(prefix + 'CPU'):
            avail = ort.get_available_providers()
            cands = [('CUDAExecutionProvider', 'CUDA', {'use_tf32': '0', 'cudnn_conv_algo_search': 'HEURISTIC'}, False),
                     ('DmlExecutionProvider', 'DirectML', {}, True),
                     ('ROCMExecutionProvider', 'ROCm', {}, False),
                     ('CoreMLExecutionProvider', 'Core ML', {'ModelFormat': 'MLProgram', 'MLComputeUnits': 'CPUAndGPU'}, False)]
            notes = []
            for name, lab, opts, dml in cands:
                if name not in avail:
                    continue
                try:
                    so2 = ort.SessionOptions(); so2.log_severity_level = 3
                    if dml:                                    # DirectML requirements
                        so2.enable_mem_pattern = False; so2.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    g = ort.InferenceSession(path, so2, providers=[(name, opts), 'CPUExecutionProvider'])
                    if name not in g.get_providers():
                        raise RuntimeError('provider did not load')
                    a, b = g.run(None, feed), mk_cpu().run(None, feed)
                    err = max(float(np.abs(x - y).max() / max(np.abs(y).max(), 1e-9)) for x, y in zip(a, b))
                    if np.isfinite(err) and err < 1e-3:
                        sess, label = g, lab
                        break
                    notes.append(f'{lab} result differed from the CPU ({err:.1e})')
                except Exception as e:
                    notes.append(f'{lab} unavailable ({str(e).strip().splitlines()[0][:90] if str(e).strip() else type(e).__name__})')
            if sess is None and notes:
                note = '; '.join(notes) + '; using the CPU.'
            elif sess is None:
                note = 'No GPU provider in this onnxruntime build; using the CPU.'
        if sess is None:
            sess = mk_cpu()
        return (lambda f: sess.run(None, f)), label, note
    return factory


_ENGINE_CACHE = {}     # tool key -> (model path, Engine)   (one model in memory per tool; the combined window unloads the others)


def get_engine(spec, model_path):
    c = _ENGINE_CACHE.get(spec.key)
    if c and c[0] == str(model_path):
        return c[1]
    _ENGINE_CACHE.pop(spec.key, None)
    import_engine(spec)
    p = Path(str(model_path or ''))
    if not p.is_file():
        raise FileNotFoundError(f'Model not found: {p}')
    eng = ENGINES[spec.key].Engine(str(p), make_session_factory(spec.prefix))
    _ENGINE_CACHE[spec.key] = (str(p), eng)
    return eng


def unload_engines(keep=None):
    for k in list(_ENGINE_CACHE):
        if k != keep:
            _ENGINE_CACHE.pop(k, None)
    import gc; gc.collect()


def stretched_warning(data):
    """Message if the data looks stretched (median > 0.2 of max), else None."""
    d = np.asarray(data, np.float32)
    fin = d[np.isfinite(d)]
    if fin.size == 0:
        return None
    mx = float(fin.max())
    if mx > 0 and float(np.median(fin)) > 0.2 * mx:
        return 'the data looks stretched (median above 20% of the maximum); this tool expects linear data. Running anyway.'
    return None


# ----------------------------------------------------------------------------------------------- model download / updates
class DownloadError(Exception):
    pass


class UpdSpec:
    """Where a script looks for models: its manifest repo and every repo whose release assets / release page it accepts."""
    def __init__(self, repo, allowed=None):
        self.repo = repo
        self.allowed = list(allowed or [repo])
        self.manifest_url = f'https://raw.githubusercontent.com/{repo}/main/models/models.json'
        self.releases_url = f'https://github.com/{repo}/releases'


def updspec_for(spec):
    return UpdSpec(spec.repo, [spec.repo, *spec.legacy_repos])


def combined_upd(specs):
    """The combined script's own manifest repo; it accepts model / script URLs from the repos of all its tools (no re-upload)."""
    me = f'{GITHUB_USER}/umbra-astro-ai'
    return UpdSpec(me, [me] + [t.repo for t in specs] + [r for t in specs for r in t.legacy_repos])


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        from urllib.parse import urlparse
        u = urlparse(newurl)
        if u.scheme != 'https' or u.hostname not in _REDIRECT_HOSTS:
            raise DownloadError(f'Refused a redirect to {u.hostname or newurl}')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _ssl_context():
    """Default certificate store; if the Python build has none (python.org macOS builds without 'Install Certificates'), use certifi's.
    Certificates are always verified."""
    import ssl
    ctx = ssl.create_default_context()
    if not ctx.cert_store_stats().get('x509_ca') and not ctx.cert_store_stats().get('x509'):
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass
    return ctx


def _open(url, timeout=20):
    """The only network entry point: HTTPS GET with a generic user agent; redirects only to GitHub hosts; certificates verified."""
    opener = urllib.request.build_opener(_SafeRedirect(), urllib.request.HTTPSHandler(context=_ssl_context()))
    return opener.open(urllib.request.Request(url, headers={'User-Agent': USER_AGENT}), timeout=timeout)


def _valid_model_url(url, file, upd):
    from urllib.parse import urlparse
    u = urlparse(str(url))
    return (u.scheme == 'https' and u.hostname == 'github.com' and not u.query and u.path.endswith('/' + file)
            and any(u.path.startswith(f'/{r}/releases/download/') for r in upd.allowed))


def _valid_entry(m, upd):
    try:
        f = m['file']
        return (re.fullmatch(r'[A-Za-z0-9._-]{1,120}\.onnx', f) is not None and '..' not in f and _valid_model_url(m['url'], f, upd)
                and re.fullmatch(r'[0-9a-f]{64}', str(m['sha256']).lower()) is not None and int(m['size_bytes']) > 0 and bool(m.get('name')))
    except Exception:
        return False


def _version_tuple(v):
    try:
        return tuple(int(p) for p in str(v).strip().lstrip('v').split('.'))
    except Exception:
        return None


def parse_manifest(data, upd):
    """Validated manifest dict {'models': [...valid entries...], 'script': {'version','url'} or None}. Bad entries and entries marked
    "legacy" (kept only for old script versions) are dropped."""
    if not isinstance(data, dict) or data.get('schema') != 1 or not isinstance(data.get('models'), list):
        raise DownloadError('The model list has an unexpected format.')
    models = [dict(m, sha256=str(m['sha256']).lower(), size_bytes=int(m['size_bytes'])) for m in data['models']
              if isinstance(m, dict) and not m.get('legacy') and _valid_entry(m, upd)]
    sc = data.get('script')
    if not (isinstance(sc, dict) and _version_tuple(sc.get('version')) and any(str(sc.get('url', '')).startswith(f'https://github.com/{r}/') for r in upd.allowed)):
        sc = None
    return dict(models=models, script=sc)


_MANIFEST_CACHE = {}


def fetch_manifest(upd, url=None, use_cache=False):
    key = url or upd.manifest_url
    c = _MANIFEST_CACHE.get(key)
    if use_cache and c and time.time() - c[0] < 300:
        return c[1]
    try:
        with _open(key, 20) as resp:
            raw = resp.read(1_000_000)
        man = parse_manifest(json.loads(raw.decode('utf-8')), upd)
        _MANIFEST_CACHE[key] = (time.time(), man)
        return man
    except DownloadError:
        raise
    except Exception as e:
        why = ' Certificate problem: on macOS run "Install Certificates.command" for your Python.' if 'CERTIFICATE' in str(e).upper() else ''
        raise DownloadError(f'Could not reach the model list ({type(e).__name__}). Offline? The installed models still work.{why}') from e


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def model_state(entry, folder):
    """'missing' | 'current' | 'update' for a manifest entry against the installed file."""
    p = Path(folder) / entry['file']
    if not p.is_file():
        return 'missing'
    return 'current' if file_sha256(p) == entry['sha256'] else 'update'


def entries_for(manifest, tool=None):
    """Manifest entries for one tool (entries with a 'tool' field belong to that tool; without one they belong to every single script)."""
    return [m for m in manifest['models'] if tool is None or m.get('tool') in (None, tool)]


def check_updates(manifest, spec, folder=None):
    """-> dict(models=[(entry, state)], script_version, script_url, script_newer)."""
    folder = folder or download_dir(spec)
    sc = manifest.get('script')
    newer = bool(sc) and _version_tuple(sc['version']) > (_version_tuple(VERSION) or (0,))
    for m in manifest['models']:
        _MANIFEST_NAMES[(m.get('tool') or spec.key, Path(m['file']).stem)] = m['name']
    return dict(models=[(m, model_state(m, folder)) for m in entries_for(manifest, spec.key)],
                script_version=sc['version'] if sc else None, script_url=sc['url'] if sc else None, script_newer=newer)


def download_model(entry, upd, folder, progress=None, cancel=None):
    """Download one manifest entry into the models folder: temp file, size and SHA-256 check, then moved into place.  Returns the path."""
    if not _valid_entry(entry, upd):
        raise DownloadError('Refused: not a valid model entry.')
    d = Path(folder)
    try:
        d.mkdir(parents=True, exist_ok=True)
        free = __import__('shutil').disk_usage(d).free
    except OSError as e:
        raise DownloadError(f'Cannot write to the models folder {d}: {e}') from e
    size = int(entry['size_bytes'])
    if free < size * 1.1:
        raise DownloadError(f'Not enough free disk space in {d} ({size / 1e6:.0f} MB needed).')
    part, final = d / f'.{entry["file"]}.part', d / entry['file']
    h = hashlib.sha256(); got = 0
    try:
        with _open(entry['url'], 30) as resp, open(part, 'wb') as out:
            while True:
                if cancel and cancel():
                    raise DownloadError('Download cancelled.')
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                got += len(chunk)
                if got > size * 1.01 + 1024:
                    raise DownloadError('The download is larger than expected; stopped.')
                out.write(chunk); h.update(chunk)
                if progress: progress(got, size)
        if got != size:
            raise DownloadError(f'Incomplete download ({got} of {size} bytes).')
        if h.hexdigest() != entry['sha256']:
            raise DownloadError('Checksum mismatch: the file was not installed. Try again later.')
        os.replace(part, final)
        return final
    except DownloadError:
        raise
    except Exception as e:
        raise DownloadError(f'Download failed ({type(e).__name__}: {e}).') from e
    finally:
        try:
            if part.exists():
                part.unlink()
        except OSError:
            pass


def describe_updates(res, spec):
    """Plain-text summary lines for a check_updates() result."""
    lines = []
    if res['script_newer']:
        lines.append(f"A newer script is available: {res['script_version']} (you have {VERSION}). {res['script_url']}")
    elif res['script_version']:
        lines.append(f'The script is up to date ({VERSION}).')
    for m, st in res['models']:
        if st == 'missing':
            lines.append(f"Available to download: {m['name']} ({m['size_bytes'] / 1e6:.0f} MB).")
        elif st == 'update':
            lines.append(f"Update available for {m['name']} (the installed file differs from the published one).")
    if not any(st != 'current' for _, st in res['models']):
        lines.append('Installed models are up to date.')
    return lines


def missing_model_message(spec, res=None):
    """Status text when no model is installed."""
    if res:
        d = next((m for m, st in res['models'] if st == 'missing' and m.get('default')), None) or next((m for m, st in res['models'] if st == 'missing'), None)
        if d:
            return f"No model installed. Press Download model to get {d['name']} ({d['size_bytes'] / 1e6:.0f} MB)."
    return f'No model installed. Press Download model, or put a .onnx file in {model_dir(spec)[0]}.'


# ----------------------------------------------------------------------------------------------- Siril I/O
def read_pixels(siril):
    data = siril.get_image_pixeldata()
    if data is None:
        fit = siril.get_image(with_pixels=True)
        data = fit.data if fit is not None else None
    if data is None:
        raise RuntimeError('No image loaded in Siril.')
    return np.array(data, copy=True)


def decode_pixels(raw):
    """Siril pixels -> float32 HxW or CxHxW (uint16 scaled to 0..1)."""
    raw = np.asarray(raw)
    if raw.dtype not in (np.dtype('float32'), np.dtype('uint16')):
        raise ValueError('Expected Siril float32 or uint16 pixels.')
    if raw.ndim == 2 or (raw.ndim == 3 and raw.shape[0] in (1, 3)):
        data = raw.astype(np.float32)
    else:
        raise ValueError('Only monochrome or channel-first RGB images are supported.')
    if raw.dtype == np.uint16:
        data /= 65535.0
    return np.ascontiguousarray(data), dict(dtype=raw.dtype, shape=raw.shape)


def encode_pixels(data, meta):
    if meta['dtype'] == np.uint16:
        out = np.rint(np.clip(data, 0, 1) * 65535).astype(np.uint16)
    else:
        out = np.asarray(data, np.float32)
    if out.shape != meta['shape']:
        raise ValueError('Output shape does not match the captured image.')
    return np.ascontiguousarray(out)


def fingerprint(raw):
    return raw.shape, raw.dtype.str, hashlib.sha256(np.ascontiguousarray(raw).tobytes()).digest()


def write_result(siril, data, meta, expected, label):
    """Apply to Siril with an Undo state, only if the image is still the one that was processed."""
    final = encode_pixels(data, meta)
    with siril.image_lock():
        current = read_pixels(siril)
        if fingerprint(current) != expected:
            raise RuntimeError('The image in Siril changed since Apply started. Nothing was written.')
        if siril.undo_save_state(label) is False:
            raise RuntimeError('Siril could not save Undo. Pixels were not overwritten.')
        if siril.set_image_pixeldata(final) is False:
            raise RuntimeError('Siril refused the modified pixels.')
        actual = read_pixels(siril)
        if actual.shape != final.shape or not np.allclose(actual, final, rtol=0, atol=0 if final.dtype == np.uint16 else 2e-7, equal_nan=True):
            raise RuntimeError('Siril readback differs from the result. Use Siril Undo if needed.')
    return fingerprint(actual)


def siril_header_text(siril):
    parts = []
    for fn in (lambda: siril.get_image_fits_header(return_as='str'), lambda: ' '.join(str(h) for h in (siril.get_image_history() or [])),
               lambda: siril.get_image_filename()):
        try:
            parts.append(str(fn()))
        except Exception:
            pass
    return ' '.join(parts)


def unique_path(p):
    p = Path(p)
    if not p.exists():
        return p
    for i in range(1, 1000):
        q = p.with_name(f'{p.stem}_{i}{p.suffix}')
        if not q.exists():
            return q
    raise FileExistsError(p)


def side_path(siril, suffix):
    """<image>_<suffix>.fit next to the image Siril has loaded (falls back to Siril's working folder)."""
    name = None
    try:
        name = siril.get_image_filename()
    except Exception:
        pass
    if name and Path(name).parent.is_dir() and str(Path(name).parent) != '.':
        return unique_path(Path(name).with_name(Path(name).stem.removesuffix('.fit') + f'_{suffix}.fit'))
    wd = None
    try:
        wd = siril.get_siril_wd()
    except Exception:
        pass
    return unique_path(Path(wd or Path.home()) / f'umbra_{suffix}.fit')


def write_fits(path, data, header=None, history=''):
    from astropy.io import fits
    hdr = header.copy() if header is not None else fits.Header()
    for k in ('BSCALE', 'BZERO', 'BLANK'):
        hdr.pop(k, None)
    if history:
        hdr['HISTORY'] = history
    fits.PrimaryHDU(np.asarray(data, np.float32), header=hdr).writeto(path)


def siril_fits_header(siril):
    try:
        from astropy.io import fits
        return fits.Header.fromstring(siril.get_image_fits_header(return_as='str'), sep='\n')
    except Exception:
        return None


def region_from_selection(siril, minimum=64):
    """Siril's current selection (x, y, w, h) when at least `minimum` px each way, else None."""
    try:
        sel = siril.get_siril_selection()
    except Exception:
        return None
    if sel and sel[2] >= minimum and sel[3] >= minimum:
        return tuple(int(v) for v in sel)
    return None


def crop_region(data, rect=None, size=768):
    """data (H,W) or (C,H,W) -> (cropped view copy, (x, y, w, h)).  rect given: that region; else a centred size x size crop (whole image if smaller)."""
    H, W = data.shape[-2:]
    if rect is None:
        w, h = min(size, W), min(size, H)
        rect = ((W - w) // 2, (H - h) // 2, w, h)
    x, y, w, h = rect
    x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    w, h = min(w, W - x), min(h, H - y)
    return np.ascontiguousarray(data[..., y:y + h, x:x + w]), (x, y, w, h)


# ----------------------------------------------------------------------------------------------- display (auto-stretch)
def _mtf(m, x):
    return (m - 1.0) * x / ((2.0 * m - 1.0) * x - m)


def stretch_params(img, target=0.25, shadow_k=-2.8):
    """Linked auto-stretch (one set of parameters for all channels), like an STF: (c0, midtone)."""
    v = np.asarray(img, np.float32)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 0.5, 1.0
    if v.size > 2_000_000:
        v = v[::max(1, v.size // 2_000_000)]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    c0 = min(max(0.0, med + shadow_k * mad), max(med - 1e-6, 0.0)) if mad > 0 else 0.0
    hi = float(np.percentile(v, 99.9)) if v.size else 1.0
    span = max(hi - c0, 1e-6) if hi > c0 else 1.0
    medn = min(max((med - c0) / span, 1e-6), 0.999)
    m = float(_mtf(target, medn))
    return c0, min(max(m, 1e-4), 0.9999), span


def apply_stretch(img, params):
    c0, m, span = params
    x = np.clip((np.nan_to_num(np.asarray(img, np.float32)) - c0) / span, 0, 1)
    y = _mtf(m, x)
    return np.clip(y, 0, 1)


def pool_for_display(img, max_side=1600):
    """Block-mean downsample (C,H,W) so the longer side is <= max_side."""
    H, W = img.shape[-2:]
    f = int(math.ceil(max(H, W) / max_side))
    if f <= 1:
        return img
    h, w = H // f * f, W // f * f
    a = np.nan_to_num(img[..., :h, :w]).reshape(*img.shape[:-2], h // f, f, w // f, f)
    return a.mean((-3, -1)).astype(np.float32)


def to_rgb8(img, params, flip=False):
    """(H,W) or (C,H,W) linear -> (H,W,3) uint8 with the given stretch."""
    a = np.asarray(img, np.float32)
    if a.ndim == 2:
        a = a[None]
    a = apply_stretch(a, params)
    if a.shape[0] == 1:
        a = np.repeat(a, 3, 0)
    rgb = np.ascontiguousarray((a.transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8))
    return rgb[::-1].copy() if flip else rgb


# ----------------------------------------------------------------------------------------------- UI
STYLE = """
QWidget { background: #242424; color: #dddddd; font-size: 10pt; }
QWidget#Card { background: #303030; border-radius: 10px; }
QLabel, QCheckBox, QSlider { background: transparent; }
QLabel#Title { font-size: 15pt; font-weight: bold; color: #dddddd; }
QLabel#Badge { color: #888888; font-size: 8pt; }
QLabel#Note { color: #999999; font-size: 9pt; }
QLabel#Value { color: #dddddd; font-weight: bold; min-width: 40px; }
QLabel#Caption { color: #aaaaaa; font-size: 9pt; }
QComboBox { background: #404040; border: 1px solid #666666; border-radius: 4px; padding: 3px 6px; }
QComboBox QAbstractItemView { background: #303030; selection-background-color: #1274d5; }
QPushButton { min-height: 20px; background: #505050; border: 1px solid #666666; border-radius: 5px; padding: 6px 12px; }
QPushButton:hover { background: #616161; }
QPushButton:checked { background: #1274d5; border-color: #338be7; color: #ffffff; }
QPushButton:disabled { color: #777777; border-color: #444444; }
QPushButton#Apply { min-height: 20px; background: #1274d5; color: #ffffff; font-weight: bold; border-color: #338be7; }
QPushButton#Apply:disabled { background: #354656; color: #808890; border-color: #43505c; }
QSlider::groove:horizontal { height: 4px; background: #505050; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #1474d5; border-radius: 2px; }
QSlider::handle:horizontal { background: #dddddd; width: 14px; margin: -6px 0; border-radius: 7px; }
QSlider::sub-page:horizontal:disabled { background: #555555; }
QProgressBar { border: 1px solid #444444; border-radius: 3px; max-height: 6px; }
QProgressBar::chunk { background: #1474d5; }
"""

IDLE_SIRIL = 'Set the controls, then press Preview or Apply. Apply replaces the image in Siril (Undo available).'
IDLE_TEST = 'Test mode (no Siril).'
_QT = {}


def _qt():
    """Build the Qt classes once (PyQt6 is imported only when a window is opened)."""
    if _QT:
        return _QT['ns']
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QProgressBar,
                                 QComboBox, QCheckBox, QSizePolicy, QStackedWidget, QMenu, QLayout)

    class Worker(QThread):
        done = pyqtSignal(object, object, float, str, object)
        failed = pyqtSignal(str, bool)       # message, during model load
        cancelled = pyqtSignal()
        prog = pyqtSignal(int, int)
        stage = pyqtSignal(str)

        def __init__(self, spec, model_path, data, params):
            super().__init__()
            self.spec, self.model_path, self.data, self.params = spec, model_path, data, params
            self.cancel_flag = False

        def run(self):
            loading = True
            try:
                t0 = time.time()
                if str(self.model_path) not in [c[0] for c in _ENGINE_CACHE.values()]:
                    self.stage.emit('Loading model…')
                eng = get_engine(self.spec, self.model_path)
                loading = False
                self.stage.emit('Running…')
                out, info = eng.run(self.data, self.params, progress=lambda n, t: self.prog.emit(int(n), int(t)), cancel=lambda: self.cancel_flag)
                self.done.emit(out, info, time.time() - t0, eng.provider, eng.note)
            except Exception as e:
                if not loading and str(e) == 'Cancelled':
                    self.cancelled.emit()
                    return
                import traceback
                tb = traceback.format_exc()
                write_error_log(self.spec, tb)
                self.failed.emit(f'{type(e).__name__}: {e}' if not loading else str(e), loading)

    class Task(QThread):
        done = pyqtSignal(object)
        failed = pyqtSignal(str)
        prog = pyqtSignal(int, int)

        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def run(self):
            try:
                self.done.emit(self.fn(lambda a, b: self.prog.emit(int(a), int(b))))
            except Exception as e:
                self.failed.emit(str(e) if isinstance(e, DownloadError) else f'{type(e).__name__}: {e}')

    def qimage(rgb):
        h, w = rgb.shape[:2]
        return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

    class PreviewWindow(QWidget):
        """Umbra <Tool> Preview (BETA): before/after with the same linked auto-stretch; hold Space (or the button) for the original."""
        def __init__(self, spec):
            super().__init__()
            self.spec = spec
            self.setWindowTitle(f'{spec.name} Preview (BETA)')
            self.setStyleSheet(STYLE)
            self.resize(820, 760)
            self.layers = {}
            self.pix = {}
            lay = QVBoxLayout(self)
            lay.setContentsMargins(8, 8, 8, 8)
            self.view = QLabel()
            self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.view.setMinimumSize(200, 200)
            self.view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            lay.addWidget(self.view, 1)
            row = QHBoxLayout()
            self.caption = QLabel('', objectName='Caption', wordWrap=True)
            row.addWidget(self.caption, 1)
            self.orig_btn = QPushButton('Show original')
            self.show_orig = False
            self.orig_btn.setToolTip('Hold the button (or the Space key) to see the original; release to go back to the result.')
            self.orig_btn.pressed.connect(lambda: self.hold(True))
            self.orig_btn.released.connect(lambda: self.hold(False))
            row.addWidget(self.orig_btn)
            self.alt_btn = None
            if spec.alt:
                self.alt_btn = QPushButton(spec.alt[0])
                self.alt_btn.setCheckable(True)
                self.alt_btn.toggled.connect(lambda c: (self.hold(False) if c else None, self.refresh()))
                row.addWidget(self.alt_btn)
            lay.addLayout(row)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        def hold(self, flag):
            self.show_orig = bool(flag)
            self.refresh()

        def show_result(self, before, after, alt, caption):
            """before / after / alt: (H,W,3) uint8 (alt may be None)."""
            self.pix = {'before': QPixmap.fromImage(qimage(before)), 'after': QPixmap.fromImage(qimage(after))}
            if alt is not None:
                self.pix['alt'] = QPixmap.fromImage(qimage(alt))
            if self.alt_btn is not None:
                self.alt_btn.setVisible(alt is not None)
                if alt is None:
                    self.alt_btn.setChecked(False)
            self.show_orig = False
            self.caption.setText(caption)
            self.refresh()
            self.show(); self.raise_()

        def current_key(self):
            if self.alt_btn is not None and self.alt_btn.isChecked() and 'alt' in self.pix:
                return 'alt'
            return 'before' if self.show_orig else 'after'

        def refresh(self):
            p = self.pix.get(self.current_key())
            if p is None or p.isNull():
                return
            self.view.setPixmap(p.scaled(self.view.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        def resizeEvent(self, e):
            super().resizeEvent(e)
            self.refresh()

        def keyPressEvent(self, e):
            if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
                self.hold(True)
            else:
                super().keyPressEvent(e)

        def keyReleaseEvent(self, e):
            if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
                self.hold(False)
            else:
                super().keyReleaseEvent(e)

    class Ui:
        """Control builders handed to the tool panels so every tool looks the same."""
        def __init__(self, view):
            self.view = view

        def slider(self, layout, text, lo, hi, default, fmt, tip=None):
            row = QHBoxLayout()
            label = QLabel(text)
            row.addWidget(label)
            val = QLabel(fmt(default), objectName='Value')
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(val)
            layout.addLayout(row)
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(lo, hi); s.setValue(default)
            s.valueChanged.connect(lambda v, l=val: l.setText(fmt(v)))
            if tip:
                label.setToolTip(tip); s.setToolTip(tip); val.setToolTip(tip)
            layout.addWidget(s)
            s.label_widget, s.value_widget = label, val
            return s

        def check(self, layout, text, tip=None, checked=False):
            c = QCheckBox(text)
            c.setChecked(checked)
            if tip:
                c.setToolTip(tip)
            layout.addWidget(c)
            return c

        def combo(self, layout, text, items, tip=None):
            row = QHBoxLayout()
            label = QLabel(text)
            row.addWidget(label)
            cb = QComboBox()
            for t, v in items:
                cb.addItem(t, v)
            if tip:
                label.setToolTip(tip); cb.setToolTip(tip)
            row.addWidget(cb, 1)
            layout.addLayout(row)
            cb.label_widget = label
            return cb

        def note(self, layout, text):
            l = QLabel(text, objectName='Note', wordWrap=True)
            layout.addWidget(l)
            return l

        @staticmethod
        def show(widget, flag):
            for w in (widget, getattr(widget, 'label_widget', None), getattr(widget, 'value_widget', None)):
                if w is not None:
                    w.setVisible(bool(flag))
            QTimer.singleShot(0, lambda: widget.window().adjustSize())

        @staticmethod
        def enable(widget, flag):
            for w in (widget, getattr(widget, 'label_widget', None), getattr(widget, 'value_widget', None)):
                if w is not None:
                    w.setEnabled(bool(flag))

        def model_dir(self):
            return model_dir(self.view.spec)[0]

    class ToolView(QWidget):
        """One tool: title, model card + tool controls, Preview / Apply, progress, status, Close. Used alone or as a page of the combined window."""
        def __init__(self, spec, siril=None, fake_image=None, prefs=None, upd=None, show_title=True, smoke=False):
            super().__init__()
            self.spec, self.siril, self.fake, self.prefs, self.smoke = spec, siril, fake_image, prefs, smoke
            self.upd = upd or updspec_for(spec)
            self.worker = self.net = self.preview = None
            self.info = None
            self.job = None
            self.cache = None            # (identity, params key, out, info) of the last whole-image preview (Gradient)
            self.last_preview = None     # (before, after, alt) uint8 of the last preview
            self.result = None
            self.last_error = None
            self.busy = False
            saved_model, saved_settings = (prefs.load(spec.key) if prefs else (None, {}))
            self.saved_model = saved_model
            lay = QVBoxLayout(self)
            lay.setContentsMargins(14, 12, 14, 12)
            lay.setSpacing(8)
            self.lay = lay
            if show_title:
                lay.addWidget(QLabel(spec.title, objectName='Title'))
                lay.addWidget(QLabel(f'v{VERSION}', objectName='Badge'))
            for n in spec.notes:
                lay.addWidget(QLabel(n, objectName='Note', wordWrap=True))
            card = QWidget(objectName='Card')
            card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            cl = QVBoxLayout(card)
            cl.setContentsMargins(12, 10, 12, 12)
            mrow = QHBoxLayout()
            mrow.addWidget(QLabel('Model'))
            self.model_box = QComboBox()
            mrow.addWidget(self.model_box, 1)
            cl.addLayout(mrow)
            brow = QHBoxLayout()
            self.dl_btn = QPushButton('Download model')
            self.dl_btn.setMinimumHeight(30)
            self.dl_btn.setToolTip("Download a model from this project's GitHub releases (checked against a SHA-256 before it is installed).")
            self.dl_btn.clicked.connect(self.show_dl_menu)
            self.upd_btn = QPushButton('Check for updates')
            self.upd_btn.setMinimumHeight(30)
            self.upd_btn.setToolTip('Looks for newer models and a newer script. Nothing is downloaded or replaced until you ask.')
            self.upd_btn.clicked.connect(lambda: self.fetch_info(manual=True))
            brow.addWidget(self.dl_btn); brow.addWidget(self.upd_btn)
            cl.addLayout(brow)
            self.upd_note = QLabel('', objectName='Note', wordWrap=True)
            self.upd_note.setOpenExternalLinks(True)
            self.upd_note.setTextFormat(Qt.TextFormat.RichText)
            self.upd_note.hide()
            cl.addWidget(self.upd_note)
            self.ui = Ui(self)
            self.panel = spec.build_panel(self.ui, cl, self)
            self.panel.set_settings(saved_settings)
            lay.addWidget(card)
            brow2 = QHBoxLayout()
            self.preview_btn = QPushButton('Preview')
            self.preview_btn.setToolTip('Runs on the selection in Siril (or a centred crop) and shows the result here. The image is not changed.' if not spec.preview_whole
                                        else 'Runs on the whole image and shows the result here, downsampled. The image is not changed.')
            self.preview_btn.clicked.connect(lambda: self.start('preview'))
            self.apply_btn = QPushButton('Apply', objectName='Apply')
            self.apply_btn.clicked.connect(lambda: self.start('apply'))
            brow2.addWidget(self.preview_btn); brow2.addWidget(self.apply_btn)
            lay.addLayout(brow2)
            prow = QHBoxLayout()
            self.progress = QProgressBar()
            self.progress.setRange(0, 0); self.progress.setTextVisible(False); self.progress.hide()
            self.cancel_btn = QPushButton('Cancel')
            self.cancel_btn.clicked.connect(self.cancel)
            self.cancel_btn.hide()
            prow.addWidget(self.progress, 1); prow.addWidget(self.cancel_btn)
            lay.addLayout(prow)
            self.idle_msg = IDLE_SIRIL if siril is not None else IDLE_TEST
            self.status = QLabel(self.idle_msg, objectName='Note', wordWrap=True)
            lay.addWidget(self.status)
            close = QPushButton('Close')
            close.clicked.connect(lambda: self.window().close())
            lay.addWidget(close)
            self.model_box.currentIndexChanged.connect(lambda *_: self.model_changed())
            self.folder_note = None
            self.refresh_models()

        # ---- status
        def set_status(self, msg, error=False):
            self.status.setText(msg)
            self.status.setStyleSheet('color: #e05050;' if error else '')
            if self.isVisible():                      # a wrapped status line may need more (or less) height
                QTimer.singleShot(0, self.window().adjustSize)

        def model_path(self):
            return self.model_box.currentData()

        def save_prefs(self):
            if self.prefs:
                self.prefs.save(self.spec.key, self.model_path(), self.panel.settings(), getattr(self.window(), 'current_key', None))

        def set_busy(self, flag, net=False):
            self.busy = flag
            has = bool(self.model_path())
            self.preview_btn.setEnabled(not flag and has); self.apply_btn.setEnabled(not flag and has)
            self.dl_btn.setEnabled(not flag); self.upd_btn.setEnabled(not flag)
            self.model_box.setEnabled(not flag)
            self.progress.setVisible(flag); self.cancel_btn.setVisible(flag and not net)
            if flag:
                self.progress.setRange(0, 0)

        # ---- models
        def refresh_models(self, select=None):
            sp = self.spec
            cur = select or self.model_path() or default_model_path(sp, self.saved_model)
            self.model_box.blockSignals(True)
            self.model_box.clear()
            items = list_models(sp)
            for label, path in items:
                self.model_box.addItem(label, path)
            i = self.model_box.findData(cur) if cur else -1
            if i < 0 and cur and Path(cur).is_file():
                self.model_box.addItem(model_label(sp, Path(cur).stem), cur)
                i = self.model_box.count() - 1
            if self.model_box.count():
                self.model_box.setCurrentIndex(max(i, 0))
            self.model_box.blockSignals(False)
            has = bool(self.model_path())
            self.preview_btn.setEnabled(has and not self.busy); self.apply_btn.setEnabled(has and not self.busy)
            self.dl_btn.setStyleSheet('' if has else 'background: #1274d5; color: #ffffff; font-weight: bold;')
            d, fb = model_dir(sp)
            if fb and self.folder_note is None:
                self.folder_note = f"Using models from the old '{sp.fallback_folder}' folder."
            self.model_changed(remember=bool(select))

        def model_changed(self, *_, remember=False):
            path = self.model_path()
            if not path:
                self.panel.model_changed(None)
                self.set_status(missing_model_message(self.spec, self.info))
                return
            self.panel.model_changed(path)
            if remember:
                self.save_prefs()
            idle = self.idle_msg
            if self.folder_note:
                idle = self.folder_note + ' ' + idle
            self.set_status(idle)

        # ---- updates / downloads (a button press starts every network action, except the manifest lookup at launch)
        def run_net(self, fn, on_done, busy_text):
            if self.net is not None and self.net.isRunning():
                return
            self.net = Task(fn)
            self.net.prog.connect(self.net_progress)
            self.net.done.connect(lambda r: (self.set_busy(False), on_done(r)))
            self.net.failed.connect(lambda m: (self.set_busy(False), self.net_failed(m)))
            self.set_busy(True, net=True)
            self.set_status(busy_text)
            self.net.start()

        def net_progress(self, got, total):
            if total > 0:
                self.progress.setRange(0, 1000); self.progress.setValue(int(1000 * got / total))
                self.set_status(f'Downloading… {got / 1e6:.0f} / {total / 1e6:.0f} MB')

        def net_failed(self, msg):
            self.set_status(msg + ('' if self.model_path() else ' No model is installed yet.'), True)

        def fetch_info(self, manual=False, then=None):
            sp = self.spec
            if offline_mode(sp):
                self.upd_note.setText(f'Offline mode ({sp.prefix}OFFLINE is set); using the installed models.'); self.upd_note.show()
                return
            def work(prog):
                return check_updates(fetch_manifest(self.upd, use_cache=not manual), sp)
            def done(res):
                self.info = res
                if manual:
                    from html import escape
                    lines = describe_updates(res, sp)
                    url = res['script_url']
                    html = '<br>'.join(escape(l).replace(escape(url), f'<a href="{escape(url)}">release page</a>') if url else escape(l) for l in lines)
                    self.upd_note.setText(html); self.upd_note.show()
                if then:
                    then()
                elif not self.model_path():
                    self.set_status(missing_model_message(sp, res))
            def failed(msg):
                self.info = None
                if manual or not self.model_path():
                    self.upd_note.setText(msg); self.upd_note.show()
                self.set_status(msg if not self.model_path() else self.idle_msg, not self.model_path())
            self.net = Task(work)
            self.net.done.connect(lambda r: (self.set_busy(False), done(r)))
            self.net.failed.connect(lambda m: (self.set_busy(False), failed(m)))
            self.set_busy(True, net=True)
            self.set_status('Checking the model list…')
            self.net.start()

        def show_dl_menu(self):
            if self.info is None:
                if offline_mode(self.spec):
                    self.set_status(f'Offline mode ({self.spec.prefix}OFFLINE is set). Put a .onnx file in {download_dir(self.spec)}.')
                else:
                    self.fetch_info(manual=False, then=self.show_dl_menu)
                return
            menu = QMenu(self)
            for m, st in self.info['models']:
                if st == 'current':
                    continue
                tag = 'Update ' if st == 'update' else 'Download '
                menu.addAction(f"{tag}{m['name']} ({m['size_bytes'] / 1e6:.0f} MB)").triggered.connect(lambda _=False, e=m: self.start_download(e))
            if menu.isEmpty():
                self.set_status('All published models are already installed and up to date.')
                return
            menu.exec(self.dl_btn.mapToGlobal(self.dl_btn.rect().bottomLeft()))

        def start_download(self, entry):
            def work(prog):
                return download_model(entry, self.upd, download_dir(self.spec), progress=prog)
            def done(path):
                self.info = None
                self.refresh_models(select=str(path))
                self.set_status(f"Installed {entry['name']}. " + self.idle_msg)
            self.run_net(work, done, f"Downloading {entry['name']}…")

        # ---- run
        def cancel(self):
            if self.worker is not None and self.worker.isRunning():
                self.worker.cancel_flag = True
                self.cancel_btn.setEnabled(False)

        def read_image(self, kind):
            """-> (data, meta, identity, header_text, region note, rect)"""
            sp = self.spec
            if self.siril is not None:
                with self.siril.image_lock():
                    raw = read_pixels(self.siril)
                    data, meta = decode_pixels(raw)
                    identity = fingerprint(raw)
                    htxt = siril_header_text(self.siril)
            else:
                data, meta, identity, htxt = self.fake, dict(dtype=np.dtype('float32'), shape=self.fake.shape), fingerprint(self.fake), ''
            return data, meta, identity, htxt

        def start(self, kind):
            if self.busy:
                return
            sp = self.spec
            path = self.model_path()
            if not path:
                self.set_status(missing_model_message(sp, self.info), True)
                return
            try:
                data, meta, identity, htxt = self.read_image(kind)
            except Exception as e:
                self.set_status(f'Could not read the image: {e}', True)
                return
            params = dict(self.panel.params())
            params.update(sp.image_params(htxt))
            rect, region, crop = None, 'whole image', None
            full = data
            if kind == 'preview':
                params = sp.preview_params(params)
                if not sp.preview_whole:
                    sel = region_from_selection(self.siril) if self.siril is not None else None
                    H, W = full.shape[-2:]
                    # Siril selections are in display coordinates; pixel rows are bottom-up (rows H-y-h .. H-y)
                    _, arect = crop_region(full, None if sel is None else (sel[0], H - sel[1] - sel[3], sel[2], sel[3]))
                    rect = sel if sel is not None else arect
                    x, y, w, h = arect
                    data = np.ascontiguousarray(full[..., y:y + h, x:x + w])
                    if w < 512 or h < 512:                   # fixed 512 px models: give small regions context, crop back afterwards
                        tw, th = min(max(w, 640), W), min(max(h, 640), H)
                        x0 = min(max(x - (tw - w) // 2, 0), W - tw); y0 = min(max(y - (th - h) // 2, 0), H - th)
                        if (tw, th) != (w, h):
                            data = np.ascontiguousarray(full[..., y0:y0 + th, x0:x0 + tw])
                            crop = (x - x0, y - y0, w, h)
                    region = (f'selection {rect[2]}×{rect[3]} px at {rect[0]},{rect[1]}' if sel is not None
                              else (f'centre crop {rect[2]}×{rect[3]} px' if (rect[2], rect[3]) != (W, H) else f'whole image {W}×{H} px'))
                else:
                    region = f'whole image {data.shape[-1]}×{data.shape[-2]} px'
            key = json.dumps(params, sort_keys=True, default=str) + str(path)
            if kind == 'apply' and sp.reuse_preview and self.cache and identity is not None and self.cache[0] == identity and self.cache[1] == key:
                self.job = dict(kind=kind, meta=meta, identity=identity, params=params, region=region, data=data, key=key, htxt=htxt)
                self.finish(self.cache[2], self.cache[3], 0.0, self.cache[4], self.cache[5], reused=True)
                return
            if sp.linear_only:
                w = stretched_warning(data)
                if w:
                    msg = f'{sp.name}: Warning: {w}'
                    if self.siril is not None:
                        try:
                            self.siril.log(msg)
                        except Exception:
                            pass
                    else:
                        print(msg, flush=True)
            self.job = dict(kind=kind, meta=meta, identity=identity, params=params, region=region, data=data, key=key, htxt=htxt, rect=rect, crop=crop)
            self.result = None; self.last_error = None
            self.save_prefs()
            self.set_busy(True)
            self.cancel_btn.setEnabled(True)
            self.set_status('Loading model…' if str(path) not in [c[0] for c in _ENGINE_CACHE.values()] else 'Running…')
            self.worker = Worker(sp, path, data, params)
            self.worker.stage.connect(self.set_status)
            self.worker.prog.connect(self.on_progress)
            self.worker.done.connect(self.finish)
            self.worker.failed.connect(self.fail)
            self.worker.cancelled.connect(self.was_cancelled)
            self.worker.start()

        def on_progress(self, n, total):
            if total > 0:
                self.progress.setRange(0, total); self.progress.setValue(n)
                self.set_status(f'Running… {self.spec.progress_unit} {n} / {total}')

        def was_cancelled(self):
            self.set_busy(False)
            self.set_status('Cancelled. The image was not changed.')

        def fail(self, msg, loading=False):
            self.set_busy(False)
            self.last_error = msg
            self.set_status((f'Could not load the model: {msg}' if loading else msg), True)

        def finish(self, out, info, secs, provider, note, reused=False):
            sp, job = self.spec, self.job
            self.set_busy(False)
            extra = ' '.join(dict.fromkeys(info.get('notes') or []))
            if note and note not in extra:
                extra = (extra + ' ' + note).strip()
            try:
                if job['kind'] == 'preview':
                    if job.get('crop'):                       # run on a larger context region; show only the selected part
                        cx, cy, cw, ch = job['crop']
                        sl = (Ellipsis, slice(cy, cy + ch), slice(cx, cx + cw))
                        out = out[sl]
                        job = dict(job, data=job['data'][sl])
                        info = dict(info, **{k: np.asarray(v)[sl] for k, v in info.items() if isinstance(v, np.ndarray) and v.ndim >= 2})
                    self.show_preview(out, info, job)
                    if sp.reuse_preview and job['identity'] is not None:
                        self.cache = (job['identity'], job['key'], out, info, provider, note)
                    self.result = out
                    self.set_status(f'Preview done in {secs:.1f} s. Ran on {provider}. The image was not changed.' + (' ' + extra if extra else ''))
                    return
                label = f"{sp.name} {VERSION} {sp.label(job['params'])}"
                write_out = sp.output(job['params'], out, info)
                msg = ''
                ctx = types.SimpleNamespace(siril=self.siril, header_text=job['htxt'], history=label, view=self)
                if self.siril is not None:
                    self.cache = None
                    self.job['identity'] = write_result(self.siril, write_out, job['meta'], job['identity'], label)
                    msg = sp.post_apply(ctx, job['params'], info) or ''
                    self.siril.log(f'{label} done in {secs:.1f} s (Undo available).' + (' ' + msg if msg else ''))
                H0, W0 = job['data'].shape[-2:]
                if not sp.preview_whole and min(H0, W0) < 512:
                    extra = (extra + ' Images smaller than 512 px give less accurate results near the edges.').strip()
                self.result = write_out
                self.set_status(f'Done in {secs:.1f} s. Ran on {provider}. ' + ('Applied to Siril; use Siril Undo to revert.' if self.siril is not None else 'Test mode: nothing was written.')
                                + (' ' + msg if msg else '') + (' ' + extra if extra else ''))
            except Exception as e:
                self.fail(str(e))

        def show_preview(self, out, info, job):
            sp = self.spec
            before = pool_for_display(job['data'])
            after = pool_for_display(out)
            flip = self.siril is not None
            sp_before = stretch_params(before)
            b8, a8 = to_rgb8(before, sp_before, flip), to_rgb8(after, sp_before, flip)
            alt8 = None
            if sp.alt and info.get(sp.alt[1]) is not None:
                alt = pool_for_display(np.asarray(info[sp.alt[1]], np.float32))
                alt8 = to_rgb8(alt, stretch_params(alt), flip)
            self.last_preview = (b8, a8, alt8)
            if self.preview is None:
                self.preview = PreviewWindow(sp)
            self.preview.show_result(b8, a8, alt8, f'Preview of the {job["region"]}. Same auto-stretch for before and after.')

        def shutdown(self):
            for t in (self.worker, self.net):
                if t is not None and t.isRunning():
                    if t is self.worker:
                        t.cancel_flag = True
                    t.wait(8000)
            if self.preview is not None:
                self.preview.close()

    class MainWindow(QWidget):
        def __init__(self, specs, combined, siril=None, fake_image=None, smoke=False):
            super().__init__()
            self.specs = {s.key: s for s in specs}
            self.combined = combined
            self.setStyleSheet(STYLE)
            self.setFixedWidth(380)
            self.views = {}
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)
            lay.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)   # window height always follows its content (wrapped notes, hidden rows)
            if combined:
                self.setWindowTitle('Umbra Astro AI (BETA)')
                prefs = Prefs(SCRIPT_DIR / 'Umbra_Astro_AI_prefs.json', combined=True, enabled=not smoke)
                top = QWidget()
                tl = QVBoxLayout(top)
                tl.setContentsMargins(14, 12, 14, 0)
                tl.setSpacing(4)
                tl.addWidget(QLabel('Umbra Astro AI (BETA)', objectName='Title'))
                tl.addWidget(QLabel(f'v{VERSION}', objectName='Badge'))
                row = QHBoxLayout()
                row.addWidget(QLabel('Tool'))
                self.tool_box = QComboBox()
                for s in specs:
                    self.tool_box.addItem(s.name, s.key)
                row.addWidget(self.tool_box, 1)
                tl.addLayout(row)
                lay.addWidget(top)
                self.stack = None
                for s in specs:
                    v = ToolView(s, siril, fake_image, prefs, combined_upd(specs),
                                 show_title=False, smoke=smoke)
                    self.views[s.key] = v
                    lay.addWidget(v)
                    v.hide()
                self.prefs = prefs
                first = prefs.load_tool()
                self.current_key = first if first in self.views else specs[0].key
                self.tool_box.setCurrentIndex(self.tool_box.findData(self.current_key))
                self.tool_box.currentIndexChanged.connect(lambda *_: self.switch(self.tool_box.currentData()))
                self.switch(self.current_key, initial=True)
            else:
                s = specs[0]
                self.setWindowTitle(s.title)
                self.current_key = s.key
                v = ToolView(s, siril, fake_image, Prefs(SCRIPT_DIR / f'{s.script}_prefs.json', enabled=not smoke), show_title=True, smoke=smoke)
                self.views[s.key] = v
                lay.addWidget(v)

        def view(self):
            return self.views[self.current_key]

        def refit(self):
            if self.layout() is not None:
                self.layout().activate()
            self.adjustSize()

        def switch(self, key, initial=False):
            if not initial:
                self.view().save_prefs()
                unload_engines()
            self.current_key = key
            for k, v in self.views.items():           # only the current page is visible, so the window is exactly as tall as that page needs
                v.setVisible(k == key)
            self.refit()
            QTimer.singleShot(0, self.refit)
            self.view().save_prefs() if not initial else None

        def closeEvent(self, e):
            for v in self.views.values():
                v.shutdown()
            try:
                self.view().save_prefs()
            except Exception:
                pass
            super().closeEvent(e)

    _QT['ns'] = types.SimpleNamespace(**{k: v for k, v in locals().items() if k in (
        'Worker', 'Task', 'PreviewWindow', 'Ui', 'ToolView', 'MainWindow', 'QApplication', 'QTimer')})
    return _QT['ns']


def fake_image(seed=0, size=192):
    """Synthetic linear RGB frame (noise + a few stars + a gentle gradient) for --ui-test."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    img = (0.02 + 0.003 * rng.standard_normal((3, size, size)) + (0.01 * xx / size)[None]).astype(np.float32)
    for cy, cx in ((40, 50), (110, 120), (150, 40), (70, 150)):
        img += (0.3 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.6 ** 2)))[None].astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


def run_app(specs, combined, siril=None, fake=None, smoke=False, shots=None):
    """Open the window. smoke: run Preview once for every tool on the fake image, report, exit (return code 0 = all fine)."""
    q = _qt()
    app = q.QApplication.instance() or q.QApplication(sys.argv)
    win = q.MainWindow(specs, combined, siril, fake, smoke)
    win.show()
    if not smoke:
        for v in win.views.values():
            if v is win.view():
                q.QTimer.singleShot(200, lambda v=v: (v.fetch_info(manual=False) if not offline_mode(v.spec) else None))
        if combined:
            win.tool_box.currentIndexChanged.connect(lambda *_: (win.view().info is None and not offline_mode(win.view().spec)) and win.view().fetch_info(manual=False))
        return app.exec()
    state = dict(i=0, ok=True, t0=time.time())
    keys = [s.key for s in specs]
    outdir = Path(shots) if shots else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    def snap(v, key):
        if not outdir:
            return
        pre = f'Umbra_Astro_AI_{key}' if combined else win.view().spec.script
        win.grab().save(str(outdir / f'{pre}_panel.png'))
        if v.preview is not None and v.preview.isVisible():
            v.preview.grab().save(str(outdir / f'{pre}_preview.png'))
            if v.preview.alt_btn is not None and v.preview.alt_btn.isVisible():
                v.preview.alt_btn.setChecked(True)
                v.preview.grab().save(str(outdir / f'{pre}_preview_{v.spec.alt[1]}.png'))
                v.preview.alt_btn.setChecked(False)

    def next_tool():
        if state['i'] >= len(keys):
            print('UI TEST', 'OK' if state['ok'] else 'FAILED', flush=True)
            win.close(); app.quit()
            return
        key = keys[state['i']]
        if combined:
            win.tool_box.setCurrentIndex(win.tool_box.findData(key))
        v = win.view()
        if not v.model_path():
            # no model: Preview/Apply must be disabled and the status must say so
            good = (not v.preview_btn.isEnabled() and not v.apply_btn.isEnabled() and 'No model installed' in v.status.text())
            print(f'UI TEST {key}: no model installed; controls {"OK" if good else "BAD"}: {v.status.text()}', flush=True)
            state['ok'] &= good
            app.processEvents(); snap(v, key)
            state['i'] += 1
            q.QTimer.singleShot(50, next_tool)
            return
        v.start('preview')
        state['t0'] = time.time()
        q.QTimer.singleShot(250, lambda: poll(v, key))

    def poll(v, key):
        if v.busy and time.time() - state['t0'] < 900:
            q.QTimer.singleShot(250, lambda: poll(v, key))
            return
        ok = v.last_error is None and v.result is not None and bool(np.isfinite(v.result).all()) and v.last_preview is not None
        if ok:
            v.preview.orig_btn.pressed.emit(); ok &= v.preview.current_key() == 'before'; v.preview.orig_btn.released.emit(); ok &= v.preview.current_key() == 'after'
        print(f'UI TEST {key}: {"OK" if ok else "FAILED"} {v.status.text()}', flush=True)
        state['ok'] &= ok
        app.processEvents()
        snap(v, key)
        if v.preview is not None:
            v.preview.close()
        state['i'] += 1
        q.QTimer.singleShot(50, next_tool)
    q.QTimer.singleShot(100, next_tool)
    app.exec()
    return 0 if state['ok'] else 1


# ----------------------------------------------------------------------------------------------- CLI
def read_fits(path):
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        hdu = next(x for x in h if x.data is not None)
        data = np.asarray(hdu.data, np.float32)
        header = hdu.header.copy()
        text = ' '.join(repr(x.header) for x in h) + ' ' + ' '.join(str(x.name) for x in h) + ' ' + Path(path).name
    return data, header, text


def run_file(spec, a):
    src, dst = Path(a.image).resolve(), Path(a.out).resolve()
    if dst == src:
        raise SystemExit('Refusing to overwrite the input file (--out equals --image).')
    if dst.exists():
        raise SystemExit(f'Output exists, refusing to overwrite: {dst}')
    data, header, text = read_fits(src)
    params = spec.params_from_args(a)
    params.update(spec.image_params(text))
    if spec.linear_only:
        w = stretched_warning(data)
        if w:
            print(f'{spec.name}: Warning: {w}', flush=True)
    path = a.model or default_model_path(spec)
    if not path:
        raise SystemExit(f'No model installed. Run with --download, or put a .onnx file in {model_dir(spec)[0]}.')
    t0 = time.time()
    eng = get_engine(spec, path)
    out, info = eng.run(data, params, progress=None, cancel=None)
    label = f"{spec.name} {VERSION} {spec.label(params)}"
    write_out = spec.output(params, out, info)
    for k in ('BSCALE', 'BZERO', 'BLANK'):
        header.pop(k, None)
    header['HISTORY'] = label
    from astropy.io import fits
    fits.PrimaryHDU(np.asarray(write_out, np.float32), header=header).writeto(dst)
    msg = spec.post_cli(a, dst, info, header, label)
    print(f'wrote {dst} shape={write_out.shape} finite={bool(np.isfinite(write_out).all())} device={eng.provider} model={Path(path).name} '
          f'{time.time() - t0:.1f} s' + (f' {msg}' if msg else '') + (f' NOTE: {" ".join(info.get("notes") or [])}' if info.get('notes') else '')
          + (f' NOTE: {eng.note}' if eng.note else ''))


def cli_updates(specs, a, upd_for=None):
    """--check-updates / --download for one or several tools. Returns an exit code."""
    try:
        for spec in specs:
            upd = upd_for(spec) if upd_for else updspec_for(spec)
            if offline_mode(spec):
                print(f'Offline mode ({spec.prefix}OFFLINE is set); nothing to check.')
                continue
            res = check_updates(fetch_manifest(upd), spec)
            if len(specs) > 1:
                print(f'== {spec.name}')
            if a.check_updates:
                print('\n'.join(describe_updates(res, spec)))
            if a.download:
                want = [e for e, st in res['models'] if st != 'current' and (a.download == 'all' or (a.download == 'default' and e.get('default')) or a.download == e['id'])]
                if not want and len(specs) == 1:
                    print('Nothing to download (installed and up to date, or unknown model id).')
                for e in want:
                    last = [0]
                    def pr(g, t, e=e, last=last):
                        if g - last[0] >= 20_000_000 or g == t:
                            last[0] = g; print(f"  {e['name']}: {g / 1e6:.0f} / {t / 1e6:.0f} MB", flush=True)
                    print('installed', download_model(e, upd, download_dir(spec), pr))
        return 0
    except DownloadError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1


def main(specs, combined=False, doc=''):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', help='standalone: FITS file to process (never modified)')
    ap.add_argument('--out', help='standalone: output FITS (must not exist)')
    ap.add_argument('--model', help='model file (*.onnx); default: the remembered / first model')
    if combined:
        ap.add_argument('--tool', choices=[s.key for s in specs], help='which tool to run / check (default for --check-updates and --download: all)')
    for s in specs:
        s.add_cli(ap)
    ap.add_argument('--check-updates', action='store_true', help='compare installed models and this script with the published manifest, then exit')
    ap.add_argument('--download', nargs='?', const='default', metavar='MODEL_ID|all', help='download a published model (default: the default model) into the models folder, then exit')
    ap.add_argument('--ui-test', action='store_true', help='open the window on a fake image, run Preview once, exit')
    ap.add_argument('--screenshots', help=argparse.SUPPRESS)
    a = ap.parse_args()
    chosen = [s for s in specs if not combined or not a.tool or s.key == a.tool]
    if a.check_updates or a.download:
        if combined:
            return cli_updates(chosen, a, lambda s: combined_upd(specs))
        return cli_updates(chosen, a)
    if a.ui_test:
        return run_app(specs, combined, None, fake_image(), smoke=True, shots=a.screenshots)
    if a.image:
        if not a.out:
            ap.error('--out is required with --image')
        spec = chosen[0] if not combined else (TOOLS.get(a.tool) if a.tool else None)
        if spec is None:
            ap.error('--tool is required with --image in the combined script')
        spec.check_cli(a, ap)
        run_file(spec, a)
        return 0
    try:
        import_engine(specs[0])   # import onnxruntime before Qt starts; any failure is kept and shown in the window
    except Exception:
        pass
    import sirilpy as s
    siril = s.SirilInterface()
    try:
        siril.connect()
        siril.cmd('requires', '1.4.0')
        return run_app(specs, combined, siril)
    finally:
        siril.disconnect()


# ============================================================================================ engines
def _engine_deblur():
    # SPDX-License-Identifier: GPL-3.0-only
    # Umbra Deblur engine, Copyright (c) 2026 Kishan Patel. Own code; no code taken from other scripts.
    """Umbra Deblur engine: pure numpy + scipy + onnxruntime (via a session factory). No Qt, no Siril, no torch.

    Processing code is the Umbra Deconvolve 0.0.14 engine, moved unchanged (the maths is identical).

    Engine(model_path, session_factory).run(data, params, progress=None, cancel=None) -> (out, info)
    params (all optional):
      sharpening      0-8, default 3.0 (models with the Sharpening control: v1.0, v1.1 experimental, legacy fb / fs / umbra_deconvolution_*)
      strength        0-1, default 0.5 (older single-strength models only)
      correct_only    bool, default False: Correct Only (strength, star and halo forced to 0, sharpening 0)
      fwhm            None = Automatic PSF, else the PSF diameter in pixels
      star, halo      0-1, default 0 (Sharpen Stars / Halo Reduction)
      drizzled        None / bool: True raises the noise estimate for correlated (drizzled) noise
      protect_stars   bool: for fine-detail models, stars come from the v1.0 model (needs star_model_path)
      star_model_path path of the v1.0 model (used when protect_stars and the model is fine-detail)
    info: 'notes' (list of str), 'fwhm' (measured global FWHM, px) and the processing details of the 0.0.14 engine.
    """
    import math
    from pathlib import Path

    import numpy as np
    from scipy.ndimage import (gaussian_filter, maximum_filter, uniform_filter, correlate1d, label, find_objects, binary_dilation)

    TOOL = 'deblur'

    TILE = 512
    OVERLAP = 64
    K_COMP = 8.0
    AUTO_PSF_MAX = 9.0       # larger Automatic PSF readings come from star residues (starless images)
    STARLESS_DENSITY = 15.0  # isolated stars per megapixel below which the image counts as starless
    STARLESS_FWHM = 4.0      # fallback used then
    SCHED_R = 0.8
    SCHED_FLOOR = 1.0
    CORR_REF = 2.2
    CORR_TRIGGER = 2.8
    CORR_C2R = 1.64


    # ------------------------------------------------------------------------------------------------ helpers
    def lmedian(a, axis=None):
        """torch.median semantics: the LOWER of the two middle values for an even count (np.median would average them)."""
        a = np.asarray(a)
        if axis is None:
            a = a.reshape(-1); axis = 0
        n = a.shape[axis]
        k = (n - 1) // 2
        return np.take(np.partition(a, k, axis=axis), k, axis=axis)


    def _resize_matrix(n_out, n_in):
        """Bilinear (align_corners=False, torch semantics) as an (n_out, n_in) matrix."""
        src = np.maximum((np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / n_out) - 0.5, 0.0)
        i0 = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
        i1 = np.minimum(i0 + 1, n_in - 1)
        w = (src - i0).astype(np.float32)
        M = np.zeros((n_out, n_in), np.float32)
        M[np.arange(n_out), i0] += 1 - w
        M[np.arange(n_out), i1] += w
        return M


    def resize_bilinear(a, H, W):
        """(C,h,w) -> (C,H,W), same as F.interpolate(mode='bilinear', align_corners=False)."""
        Ay = _resize_matrix(H, a.shape[1]); Ax = _resize_matrix(W, a.shape[2])
        return np.einsum('Hh,chw,Ww->cHW', Ay, a.astype(np.float32), Ax, optimize=True).astype(np.float32)


    def _sub(x, max_px=4_000_000):
        st = max(1, int((x.shape[-1] * x.shape[-2] / max_px) ** 0.5))
        return x[:, ::st, ::st]


    def haar_sigma(x):
        x = np.asarray(x, np.float32)
        x = x[:, : x.shape[1] // 2 * 2, : x.shape[2] // 2 * 2]
        a, b, c, d_ = x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]
        d = (a - b - c + d_) / 2.0
        ok = np.isfinite(a) & np.isfinite(d_) & (a != 0) & np.isfinite(d)
        out = []
        for ch in range(x.shape[0]):
            v = _sub(d[ch:ch + 1])[_sub(ok[ch:ch + 1])]
            out.append(np.float32(1.4826) * lmedian(np.abs(v - lmedian(v))) if v.size > 100 else np.float32(0.0))
        return np.array(out, np.float32)


    def image_stats(x):
        """(C,H,W) -> (m (C,), s (C,)): per-channel median offset and noise sigma (floored)."""
        x = np.asarray(x, np.float32)
        xs = _sub(x)
        m = np.array([lmedian(np.nan_to_num(xs[c][np.isfinite(xs[c])])) for c in range(x.shape[0])], np.float32)
        s = haar_sigma(x)
        rng = np.array([np.quantile(np.nan_to_num(xs[c]).ravel(), 0.999) if xs[c].size < 16_000_000 else xs[c].max() for c in range(x.shape[0])], np.float32) - m
        return m, np.maximum(s, 1e-5 * np.clip(np.abs(rng), 1e-12, None)).astype(np.float32)


    def compress(xn):
        return (K_COMP * np.arcsinh(xn / K_COMP)).astype(np.float32)


    def decompress(y):
        return (K_COMP * np.sinh(np.clip(y, -150, 150) / K_COMP)).astype(np.float32)


    # ------------------------------------------------------------------------------------------------ Automatic PSF
    def _interp4(k):
        """(N,K,K) -> (N,4K,4K), bilinear x4 (align_corners=False)."""
        M = _resize_matrix(4 * k.shape[-1], k.shape[-1])
        return np.einsum('ak,nkl,bl->nab', M, k, M, optimize=True)


    def eq_fwhm(k):
        u = _interp4(np.asarray(k, np.float32))
        pk = u.reshape(u.shape[0], -1).max(1)
        a = (u >= 0.5 * pk[:, None, None]).sum((1, 2)).astype(np.float32) / 16.0
        return 2 * np.sqrt(a / math.pi)


    def shape_e(k, fw):
        N, K, _ = k.shape
        r = np.arange(K, dtype=np.float32) - K // 2
        yy, xx = np.meshgrid(r, r, indexing='ij')
        w = np.clip(k, 0, None)
        t = np.clip(w.sum((1, 2)), 1e-12, None)
        my = (w * yy).sum((1, 2)) / t; mx = (w * xx).sum((1, 2)) / t
        dy, dx = yy[None] - my[:, None, None], xx[None] - mx[:, None, None]
        w = w * (((dy * dy + dx * dx) <= (1.5 * fw)[:, None, None] ** 2).astype(np.float32))
        t = np.clip(w.sum((1, 2)), 1e-12, None)
        cyy = (w * dy * dy).sum((1, 2)) / t; cxx = (w * dx * dx).sum((1, 2)) / t; cxy = (w * dy * dx).sum((1, 2)) / t
        tr = np.clip(cxx + cyy, 1e-9, None)
        return (cxx - cyy) / tr, 2 * cxy / tr


    def _blur_reflect(x, sg):
        r = int(3 * sg) + 1
        k = np.exp(-0.5 * (np.arange(-r, r + 1, dtype=np.float32) / sg) ** 2).astype(np.float32); k /= k.sum()
        p = np.pad(x, ((0, 0), (r, r)), mode='reflect')
        p = correlate1d(p, k, axis=1, mode='constant')[:, r:-r]
        p = np.pad(p, ((r, r), (0, 0)), mode='reflect')
        return correlate1d(p, k, axis=0, mode='constant')[r:-r]


    def _shift_stamps(st, cx, cy, h):
        """F.grid_sample(bilinear, zeros, align_corners=True) of centred stamps re-centred by (cx, cy): out[i,j] = st[i+cy, j+cx] (zero outside)."""
        N, S, _ = st.shape
        ii = np.arange(S, dtype=np.float32)
        y = ii[None, :, None] + cy[:, None, None]; x = ii[None, None, :] + cx[:, None, None]
        y = np.broadcast_to(y, (N, S, S)); x = np.broadcast_to(x, (N, S, S))
        y0 = np.floor(y).astype(np.int64); x0 = np.floor(x).astype(np.int64)
        wy = (y - y0).astype(np.float32); wx = (x - x0).astype(np.float32)
        n = np.arange(N)[:, None, None]
        def g(yi, xi):
            ok = (yi >= 0) & (yi < S) & (xi >= 0) & (xi < S)
            return np.where(ok, st[n, np.clip(yi, 0, S - 1), np.clip(xi, 0, S - 1)], 0.0)
        return (g(y0, x0) * (1 - wy) * (1 - wx) + g(y0, x0 + 1) * (1 - wy) * wx + g(y0 + 1, x0) * wy * (1 - wx) + g(y0 + 1, x0 + 1) * wy * wx).astype(np.float32)


    def measure_psf(Y, region=512, stamp=15, min_stars=8):
        """Y (3,H,W) linear.  Returns dict(fmap (H,W), global, grid, shape (2,H,W), n_stars) like udc.infer.measure_psf."""
        H, W = Y.shape[-2:]
        m, s = image_stats(Y)
        L = ((Y - m.reshape(-1, 1, 1)) / s.reshape(-1, 1, 1)).mean(0).astype(np.float32)
        d = _blur_reflect(L, 1.0) - _blur_reflect(L, 6.0)
        mx = float(L.max())
        pk = (d == maximum_filter(d, size=11, mode='constant', cval=-np.inf)) & (d > 20) & (L < 0.5 * mx)
        h = stamp // 2
        pk[:h + 2] = False; pk[-h - 2:] = False; pk[:, :h + 2] = False; pk[:, -h - 2:] = False
        ys, xs = np.nonzero(pk)
        if ys.size == 0:
            return dict(fmap=np.full((H, W), 3.0, np.float32), gl=3.0, grid=None, shape=np.zeros((2, H, W), np.float32), n=0)
        cnt = uniform_filter(pk.astype(np.float64), size=stamp, mode='constant') * stamp * stamp
        iso = cnt[ys, xs] < 1.5
        ys, xs = ys[iso], xs[iso]
        n_iso = int(ys.size)
        off = np.arange(-h, h + 1)
        ix = (ys[:, None, None] + off[None, :, None], xs[:, None, None] + off[None, None, :])
        st = L[ix]
        edge = lmedian(np.concatenate([st[:, 0], st[:, -1], st[:, :, 0], st[:, :, -1]], 1), 1)
        st = np.clip(st - edge[:, None, None], 0, None)
        st = st / np.clip(st.sum((1, 2), keepdims=True), 1e-9, None)
        oy, ox = np.meshgrid(off.astype(np.float32), off.astype(np.float32), indexing='ij')
        cy = (st * oy).sum((1, 2)); cx = (st * ox).sum((1, 2))
        st = _shift_stamps(st, cx, cy, h)
        Yn = (Y - m.reshape(-1, 1, 1)) / s.reshape(-1, 1, 1)
        stc = []
        for c in range(Yn.shape[0]):
            a = Yn[c][ix]
            ed = lmedian(np.concatenate([a[:, 0], a[:, -1], a[:, :, 0], a[:, :, -1]], 1), 1)
            a = np.clip(a - ed[:, None, None], 0, None); a = a / np.clip(a.sum((1, 2), keepdims=True), 1e-9, None)
            cyc = (a * oy).sum((1, 2)); cxc = (a * ox).sum((1, 2))
            stc.append(_shift_stamps(a, cxc, cyc, h))

        def fw(sel):
            if sel.sum() < min_stars:
                return None
            k = lmedian(st[sel], 0)[None]
            f = eq_fwhm(k)
            es = []
            for sc in stc:
                med = lmedian(sc[sel], 0)
                es.append(shape_e(med[None] / med.sum(), f))
            return float(f[0]), float(sum(e[0][0] for e in es) / len(es)), float(sum(e[1][0] for e in es) / len(es))

        g0 = fw(np.ones_like(ys, dtype=bool)) or (3.0, 0.0, 0.0)
        gl = g0[0]
        ny, nx = max(1, round(H / region)), max(1, round(W / region))
        grid = np.tile(np.array(g0, np.float32).reshape(3, 1, 1), (1, ny, nx))
        for i in range(ny):
            for j in range(nx):
                sel = (ys * ny // H == i) & (xs * nx // W == j)
                v = fw(sel)
                if v is not None:
                    grid[:, i, j] = v
        maps = resize_bilinear(grid, H, W)
        return dict(fmap=maps[0], gl=gl, grid=grid[0], shape=maps[1:], n=n_iso)


    def auto_psf(Y0, fwhm=None):
        """Automatic PSF (with the starless guard) or manual FWHM -> (fmap, global, grid, emap, psf_note)."""
        p = measure_psf(Y0)
        fmap, gl, grid, emap = p['fmap'], p['gl'], p['grid'], p['shape']
        psf_note = None
        dens = p['n'] / max(Y0.shape[-1] * Y0.shape[-2] / 1e6, 1e-6)
        if fwhm is None and (gl > AUTO_PSF_MAX or dens < STARLESS_DENSITY):
            psf_note = (f"Auto PSF: image looks starless ({dens:.0f} isolated stars per MP, measured {gl:.1f} px). Using {STARLESS_FWHM:.1f} px; "
                        "for starless images set the PSF Diameter to the value measured on the starred version.")
            fmap, gl, grid = np.full(Y0.shape[-2:], STARLESS_FWHM, np.float32), STARLESS_FWHM, None
            emap = np.zeros_like(emap)
        if fwhm is not None:
            fmap, gl, grid = np.full(Y0.shape[-2:], float(fwhm), np.float32), float(fwhm), None
        return fmap, gl, grid, emap, psf_note


    # ------------------------------------------------------------------------------------------------ tiled network
    def _window(n, ov):
        w = np.ones(n, np.float32)
        if ov > 0:
            r = np.linspace(0, 1, ov + 2, dtype=np.float32)[1:-1]; r = 0.5 - 0.5 * np.cos(np.pi * r)
            w[:ov] = r; w[-ov:] = r[::-1]
        return w


    def deconvolve_once(net, y, strength=1.0, fwhm=None, progress=None, star=0.0, halo=0.0, psf_fixed=None, sigma_k=1.0):
        """net: callable (x (1,3,T,T), cond (1,6,T,T)) -> r (1,3,T,T), T = 512.  y (3,H,W) float32 linear."""
        Y = np.ascontiguousarray(y, np.float32)
        fin = np.isfinite(Y); Y0 = np.nan_to_num(Y)
        if psf_fixed is not None:
            fmap, gl, emap = psf_fixed; grid = None; psf_note = None
        else:
            fmap, gl, grid, emap, psf_note = auto_psf(Y0, fwhm)
        m, s = image_stats(Y0); m = m.reshape(-1, 1, 1); s = s.reshape(-1, 1, 1) * np.float32(sigma_k)
        xn = ((Y0 - m) / s).astype(np.float32)
        C, H, W = xn.shape
        T, ov = TILE, OVERLAP
        st = T - ov
        ny = max(1, math.ceil((H - ov) / st)) if H > T else 1
        nx = max(1, math.ceil((W - ov) / st)) if W > T else 1
        Hp, Wp = max(H, (ny - 1) * st + T), max(W, (nx - 1) * st + T)
        mode = 'reflect' if (Hp - H < H and Wp - W < W) else 'edge'
        xp = np.pad(xn, ((0, 0), (0, Hp - H), (0, Wp - W)), mode=mode)
        cond = np.stack([np.full((H, W), float(strength), np.float32), fmap / 4.0, 3 * emap[0], 3 * emap[1],
                         np.full((H, W), float(star), np.float32), np.full((H, W), float(halo), np.float32)]).astype(np.float32)
        cp = np.pad(cond, ((0, 0), (0, Hp - H), (0, Wp - W)), mode='edge')
        xc = compress(xp)
        acc = np.zeros((C, Hp, Wp), np.float32); wsum = np.zeros((1, Hp, Wp), np.float32)
        w2 = (_window(T, ov if ny > 1 else 0)[:, None] * _window(T, ov if nx > 1 else 0)[None, :])[None]
        n, tot = 0, ny * nx
        for iy in range(ny):
            for ix in range(nx):
                y0, x0 = iy * st, ix * st
                r = net(np.ascontiguousarray(xc[None, :, y0:y0 + T, x0:x0 + T]), np.ascontiguousarray(cp[None, :, y0:y0 + T, x0:x0 + T]))[0]
                acc[:, y0:y0 + T, x0:x0 + T] += r * w2; wsum[:, y0:y0 + T, x0:x0 + T] += w2
                n += 1
                if progress: progress(n, tot)
        r = (acc / np.clip(wsum, 1e-8, None))[:, :H, :W]
        out = decompress(compress(xn) + r) * s + m
        return np.where(fin, out, Y).astype(np.float32), dict(fwhm_global=gl, fwhm_grid=grid, psf_note=psf_note)


    def deconvolve_chw(net, y, strength=1.0, fwhm=None, progress=None, star=0.0, halo=0.0, two_pass=True):
        """Correct Only first, then the requested sharpening (single-strength models / Sharpening 0)."""
        if two_pass and (strength > 0 or star > 0 or halo > 0):
            yc, _ = deconvolve_once(net, y, 0.0, fwhm, None, 0.0, 0.0)
            return deconvolve_once(net, yc, strength, fwhm, progress, star, halo)
        return deconvolve_once(net, y, strength, fwhm, progress, star, halo)


    # ------------------------------------------------------------------------------------------------ guards / masks
    def noise_corr_factor(y, b=64):
        L = np.nan_to_num(np.asarray(y, np.float32)).mean(0)
        _, s = image_stats(np.nan_to_num(np.asarray(y, np.float32))); sh = float(s.mean())
        H, W = L.shape
        if H < 2 * b or W < 2 * b or not sh > 0:
            return 1.0, float('nan')
        blocks = sorted((float(L[yy:yy + b, xx:xx + b].std()), yy, xx) for yy in range(0, H - b + 1, b) for xx in range(0, W - b + 1, b))
        mad = lambda a: 1.4826 * float(np.median(np.abs(a - np.median(a))))
        r, r2 = [], []
        for _, yy, xx in blocks[:max(8, len(blocks) // 20)]:
            c0 = L[yy:yy + b, xx:xx + b]; c = c0 - gaussian_filter(c0, 12); q = c.reshape(b // 4, 4, b // 4, 4).mean((1, 3))
            r.append(mad(q) * 4 / sh)
            h = c0 - gaussian_filter(c0, 6); r2.append(mad(h.reshape(b // 2, 2, b // 2, 2).mean((1, 3))) * 2 / max(mad(h), 1e-30))
        c4, c2r = float(np.median(r)), float(np.median(r2))
        noise_corr_factor.last = dict(c4=c4, c2r=c2r)
        return (float(min(2.0, c4 / CORR_REF)) if (c4 > CORR_TRIGGER and c2r >= CORR_C2R) else 1.0), c4


    def saturation_mask(y, fwhm):
        Y = np.nan_to_num(np.asarray(y, np.float32)); Lm = Y.max(0); gmax = float(Lm.max())
        if not gmax > 0: return None
        med = float(np.median(Lm)); _, s = image_stats(Y); sg = float(s.mean())
        Lm = Lm - med; gmax = gmax - med
        if not gmax > 0: return None
        Mx = maximum_filter(Lm, 7)
        cand = (Lm >= 0.97 * Mx) & (Mx > 0.25 * gmax) & (Mx > 50 * sg)
        lb, n = label(cand)
        if n == 0: return None
        area = np.bincount(lb.ravel()); amin = max(5.0, 0.15 * fwhm * fwhm)
        keep = np.zeros(n + 1, bool); H, W = Lm.shape; tops = []
        for k, sl in enumerate(find_objects(lb), 1):
            if sl is None or area[k] < amin: continue
            comp = lb[sl] == k; v = Lm[sl]; top = comp & (v >= 0.97 * float(v[comp].max()))
            if top.sum() < amin: continue
            yy, xx = np.nonzero(top); cy, cx = yy.mean() + sl[0].start, xx.mean() + sl[1].start
            rp = math.sqrt(top.sum() / math.pi); rr = rp + fwhm; pk = float(v[top].mean())
            ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            ry = np.clip(np.round(cy + rr * np.sin(ang)).astype(int), 0, H - 1); rx = np.clip(np.round(cx + rr * np.cos(ang)).astype(int), 0, W - 1)
            if float(np.median(Lm[ry, rx])) < 0.6 * pk: keep[k] = True; tops.append((sl, top))
        sat = np.zeros_like(Lm, bool)
        for sl, top in tops: sat[sl] |= top
        if not sat.any(): return None
        m = binary_dilation(sat, iterations=max(2, int(round(2 * fwhm)))).astype(np.float32)
        return np.clip(gaussian_filter(m, max(1.0, fwhm)) * 1.5, 0, 1)


    def star_mask(y, fwhm):
        Y = np.nan_to_num(np.asarray(y, np.float32)); L = Y.mean(0)
        _, s = image_stats(Y); sg = float(s.mean())
        sb = max(0.8, fwhm / 2.355); D = gaussian_filter(L, sb * 0.7) - gaussian_filter(L, 3 * sb)
        pk = (D == maximum_filter(D, size=int(2 * fwhm) | 1)) & (D > 25 * sg)
        ys, xs = np.nonzero(pk); H, W = L.shape; M = np.zeros((H, W), np.float32)
        bgl = gaussian_filter(L, 4 * fwhm)
        ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        for y0, x0 in zip(ys, xs):
            ry = np.clip(np.round(y0 + 2 * fwhm * np.sin(ang)).astype(int), 0, H - 1); rx = np.clip(np.round(x0 + 2 * fwhm * np.cos(ang)).astype(int), 0, W - 1)
            c = L[y0, x0] - bgl[y0, x0]; ring = float(np.median(L[ry, rx] - bgl[ry, rx]))
            if c <= 0 or ring > c / 3: continue
            r = min(4.0, 2.0 + 0.6 * math.log10(max(D[y0, x0] / (25 * sg), 1.0))) * fwhm; R = int(r + 2 * fwhm)
            ya, yb, xa, xb = max(0, y0 - R), min(H, y0 + R + 1), max(0, x0 - R), min(W, x0 + R + 1)
            yy, xx = np.mgrid[ya:yb, xa:xb]; d = np.hypot(yy - y0, xx - x0)
            M[ya:yb, xa:xb] = np.maximum(M[ya:yb, xa:xb], np.clip((r - d) / (0.7 * fwhm) + 0.5, 0, 1))
        return gaussian_filter(M, 0.5 * fwhm).clip(0, 1)


    def background_weight(y, fwhm):
        Y = np.nan_to_num(np.asarray(y, np.float32)); L = Y.mean(0)
        _, s = image_stats(Y); sg = float(s.mean())
        sky = float(np.percentile(gaussian_filter(L, 4)[::4, ::4], 10))
        sm = max(1.0, 0.7 * fwhm); snr = (gaussian_filter(L, sm) - sky) / (sg / max(1.0, 2 * math.sqrt(math.pi) * sm))
        w = np.clip((snr - 1.5) / 3.5, 0, 1)
        return gaussian_filter(w, sm).astype(np.float32)


    # ------------------------------------------------------------------------------------------------ scheduled passes
    def deconvolve_scheduled(net, y, plan, fwhm=None, progress=None, star=0.0, halo=0.0, r=None, floor=None, sigma_fix=True,
                             sat_protect=True, drizzled=None, bg_protect=False, net_detail=None, blend='pass'):
        r = SCHED_R if r is None else r; floor = SCHED_FLOOR if floor is None else floor
        y = np.ascontiguousarray(y, np.float32)
        fmap, g0, grid, emap, psf_note = auto_psf(np.nan_to_num(y), fwhm)
        g0 = float(g0)
        sk, c4 = noise_corr_factor(y) if sigma_fix else (1.0, float('nan'))
        if sigma_fix and drizzled is True: sk = float(np.clip(c4 / CORR_REF, 1.3, 2.0)) if c4 == c4 else 1.3
        else: sk = 1.0
        P = len(plan)
        def pr(k):
            return (lambda n, t: progress(k, n, t)) if progress else None
        z, _ = deconvolve_once(net, y, 0.0, None, pr(0), psf_fixed=(fmap, g0, emap), sigma_k=sk)
        fws = []
        smk = star_mask(y, g0) if net_detail is not None else None
        bwd = background_weight(y, g0) if net_detail is not None else None
        zd = z
        for k, st in enumerate(plan, 1):
            fk = floor + (g0 - floor) * r ** (k - 1) if g0 > floor else g0
            sc = fk / g0; last = k == P; kw = dict(psf_fixed=(fmap * sc, fk, emap), sigma_k=sk)
            zin = z
            z, _ = deconvolve_once(net, zin, float(st), None, pr(k), star if last else 0.0, halo if last else 0.0, **kw)
            if net_detail is not None:
                src = zin if blend == 'pass' else zd
                zd, _ = deconvolve_once(net_detail, src, float(st), None, None, star if last else 0.0, halo if last else 0.0, **kw)
                zd = src + bwd * (zd - src)
                if blend == 'pass': z = smk * z + (1 - smk) * zd; zd = z
            fws.append(round(fk, 3))
        if net_detail is not None and blend != 'pass': z = (smk * z + (1 - smk) * zd).astype(np.float32)
        sat_px = 0
        mk = saturation_mask(y, g0) if (sat_protect and P) else None
        if mk is not None:
            sat_px = int((mk > 0.5).sum())
            z = np.where(np.isfinite(y), z * (1 - mk) + np.nan_to_num(y) * mk, y).astype(np.float32)
        if bg_protect and P:
            bw = background_weight(y, g0)
            z = np.where(np.isfinite(y), np.nan_to_num(y) + bw * (z - np.nan_to_num(y)), y).astype(np.float32)
        return z, dict(fwhm_global=g0, fwhm_grid=grid, psf_note=psf_note, fwhm_per_pass=fws, sigma_k=sk, noise_corr=c4, sat_px=sat_px)



    # ------------------------------------------------------------------------------------------------ ONNX session adapter
    def probe_feed():
        """Random inputs matching the ONNX inputs (x (1,3,512,512), cond (1,6,512,512)), for the GPU-vs-CPU probe."""
        rng = np.random.default_rng(1)
        x = (rng.standard_normal((1, 3, TILE, TILE)) * 2).astype(np.float32)
        c = rng.random((1, 6, TILE, TILE)).astype(np.float32)
        return {'x': x, 'cond': c}


    class OnnxNet:
        """Thin adapter: net(x, cond) -> r (1,3,T,T) through the run function of a session_factory."""

        def __init__(self, path, session_factory):
            self.path = str(path)
            self._run, self.provider, self.note = session_factory(self.path, probe_feed())

        def __call__(self, x, cond):
            return self._run({'x': x, 'cond': cond})[0]


    # ----------------------------------------------------------------------------------------------- glue
    SHARP_DEFAULT = 3.0


    def _stem(model_path):
        return Path(str(model_path or '')).stem.lower()


    def is_fb(model_path):
        """fb models are trained as a small repeatable step (0.65x blur per pass) and use the Sharpening control (0-8 passes)."""
        st = _stem(model_path)
        return (st.startswith('umbra_deblur_v') or st.startswith('umbra_deconv_fb') or st.startswith('umbra_deconv_fs')
                or st.startswith('umbra_deconvolution_'))


    def is_finedetail(model_path):
        st = _stem(model_path)
        return ('finedetail' in st or st.startswith('umbra_deconvolution_1.1') or st.startswith('umbra_deblur_v1.1'))


    def pass_plan(v):
        """Sharpening v (0..8, steps of 0.25) -> list of per-pass strengths after Correct Only: floor(v) x 1.0, then frac(v) if > 0."""
        v = max(0.0, min(8.0, round(float(v) * 4) / 4))
        n, f = int(v), v - int(v)
        return [1.0] * n + ([round(f, 4)] if f > 1e-6 else [])


    def deconvolve_array(model, data, strength, fwhm, progress=None, star=0.0, halo=0.0, sharpening=None, drizzled=None, finedetail=False, star_model=None):
        """data HxW, CxHxW (C = 1 or 3) or HxWxC (C = 1 or 3) float32 -> (same shape, info). Mono is replicated to 3 channels and averaged back.
        sharpening None (older models): Correct Only + one sharpen pass at `strength` (0.0.6 behaviour).
        sharpening v (fb models; `strength` ignored): Correct Only once, then the passes of pass_plan(v) (Sharpen Stars / Halo Reduction on the
        last pass); 0.0.11: the PSF is measured once and scheduled per pass (udc.infer.deconvolve_scheduled); every pass is tiled like a single pass (same memory)."""
        data = np.asarray(data, np.float32)
        layout = 'hw'
        if data.ndim == 3:
            if data.shape[0] in (1, 3):
                layout = 'chw'
            elif data.shape[2] in (1, 3):
                layout, data = 'hwc', np.ascontiguousarray(data.transpose(2, 0, 1))
            else:
                raise ValueError(f'Unsupported image shape {data.shape}')
        chw = data[None] if data.ndim == 2 else data
        mono = chw.shape[0] == 1
        y = np.ascontiguousarray(np.repeat(chw, 3, 0) if mono else chw, np.float32)
        fw = None if fwhm is None else float(fwhm)
        plan = None if sharpening is None else pass_plan(sharpening)
        if plan is None or not plan:
            out, info = deconvolve_chw(model, y, strength=float(strength) if plan is None else 0.0, fwhm=fw, progress=progress,
                                       star=float(star), halo=float(halo))           # older models, or Sharpening 0 (= Correct Only)
        else:
            P = len(plan)
            def prog(k, n, t):                                                 # progress per pass: (pass, tile) over Correct Only + P passes
                if progress: progress(k * t + n, (P + 1) * t)
            if star_model is not None:                                         # 0.0.12 Protect star shapes: stars from star_model, rest from the fine-detail model
                out, info = deconvolve_scheduled(star_model, y, plan, fw, progress=prog, star=float(star), halo=float(halo), drizzled=drizzled,
                                                 net_detail=model, blend='end')
            else:
                out, info = deconvolve_scheduled(model, y, plan, fw, progress=prog, star=float(star), halo=float(halo), drizzled=drizzled,
                                                 bg_protect=bool(finedetail))   # 0.0.11 / 0.0.12 (sky mask for fine-detail models)
            notes = [info.get('psf_note')] if info.get('psf_note') else []
            if info.get('sigma_k', 1.0) > 1.0:
                notes.append(f"Drizzled stack: noise estimate x{info['sigma_k']:.2f} so it is not sharpened into beads.")
            if info.get('sat_px', 0) > 0:
                notes.append('Saturated star cores kept as in the original (they cannot be deconvolved).')
            info['psf_note'] = ' '.join(notes) if notes else None
            info['notes'] = notes
        out = np.asarray(out, np.float32)
        if mono:
            out = out.mean(0, keepdims=True)
        if layout == 'hw':
            out = out[0]
        elif layout == 'hwc':
            out = np.ascontiguousarray(out.transpose(1, 2, 0))
        return out, info


    class Engine:
        """Loads the model once. provider / note describe the session (GPU or CPU)."""

        def __init__(self, model_path, session_factory):
            path = Path(str(model_path))
            if not path.is_file():
                raise FileNotFoundError(f'Model not found: {path}')
            self.model_path = str(path)
            self._factory = session_factory
            self.model = OnnxNet(path, session_factory)
            self.provider = self.model.provider
            self.note = self.model.note
            self.n_cond = 6
            self.has_star_controls = True

        def run(self, data, params=None, progress=None, cancel=None):
            p = dict(params or {})
            data = np.ascontiguousarray(data, np.float32)
            fb = is_fb(self.model_path)
            correct_only = bool(p.get('correct_only', False))
            strength = float(p.get('strength', 0.5)); star = float(p.get('star', 0.0)); halo = float(p.get('halo', 0.0))
            sharpening = float(p.get('sharpening', SHARP_DEFAULT)) if fb else None
            if correct_only:
                strength, star, halo = 0.0, 0.0, 0.0
                if fb:
                    sharpening = 0.0
            fwhm = p.get('fwhm')
            fd = is_finedetail(self.model_path); sm = None
            if fd and p.get('protect_stars') and sharpening:
                sp = p.get('star_model_path')
                if not sp or not Path(str(sp)).is_file():
                    raise FileNotFoundError(f'Protect star shapes needs the Umbra Deblur v1.0 model ({sp})')
                sm = OnnxNet(sp, self._factory)
            if cancel is not None:
                inner = progress
                def progress(n, t, _inner=inner):
                    if cancel():
                        raise RuntimeError('Cancelled')
                    if _inner:
                        _inner(n, t)
            out, info = deconvolve_array(self.model, data, strength, fwhm, progress, star, halo, sharpening, p.get('drizzled'),
                                         finedetail=fd, star_model=sm)
            g = info.get('fwhm_global')
            info['fwhm'] = None if g is None else float(g)
            if 'notes' not in info:
                info['notes'] = [info['psf_note']] if info.get('psf_note') else []
            return out, info


    def stretched_warning(data):
        """Message if the data looks stretched (median > 0.2 of max), else None."""
        d = np.asarray(data, np.float32)
        fin = d[np.isfinite(d)]
        if fin.size == 0:
            return None
        mx = float(fin.max())
        if mx > 0 and float(np.median(fin)) > 0.2 * mx:
            return 'the data looks stretched (median above 20% of the maximum); this tool expects linear data. Running anyway.'
        return None
    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})


ENGINES['deblur'] = _engine_deblur()

def _engine_denoise():
    """Umbra Denoise engine: numpy + ONNX Runtime only (no Qt, no Siril, no torch).

    Same maths as the PyTorch pipeline (umbra/pipeline.py run_deployed): whole-image median / noise-sigma normalisation, asinh
    compression, 512 px tiles with a cosine overlap, regional noise guard, Detail split, mean fix, Pattern high-pass, then the
    model-independent saturated-star protection.

    params (all optional):
        denoise  0..1  strength of the noise removal                 (default 1.0)
        detail   0..1  gives back part of the removed fine detail    (default 0.0)
        pattern  0..1  strength of the structured-pattern removal    (default 0.0)
    data: float32 linear image, (H, W) mono or (C, H, W) with C in {1, 3}; NaN is kept as NaN.
    Returns (out, info), info = {'notes': [str], 'saturated': int}.

    Images smaller than one tile: PyTorch runs the network at the image's own size (a multiple of 32, e.g. 320 x 320 for
    300 x 250).  The network has global channel attention, so it depends on the tile size; the ONNX tile is fixed at 512, so
    the small padded tile is extended to 512 by mirror-tiling (period 2T), which keeps its global statistics (see PARITY_denoise.md).
    """
    import math
    import numpy as np

    TOOL = 'denoise'
    TILE = 512
    OVERLAP = 64
    K_COMP = 8.0
    GUARD_K = 4.0
    MEAN_FIX = 16.0
    COND_CH = 4
    NOTHING_TO_DO = 'Nothing to do: both sliders are at 0. The image was not changed.'


    def probe_feed():
        """Random inputs matching the ONNX inputs, for the GPU-vs-CPU probe."""
        rng = np.random.default_rng(1)
        return {'x': (rng.standard_normal((1, 3, TILE, TILE)) * 2).astype(np.float32),
                'c': np.zeros(1, np.float32),
                'cond': (rng.standard_normal((1, COND_CH)) * 0.5).astype(np.float32)}


    def _check(cancel):
        if cancel is not None and cancel():
            raise RuntimeError('Cancelled')




    def lmedian(a, axis=None):
        """torch.median semantics: the LOWER of the two middle values for an even count."""
        a = np.asarray(a)
        if axis is None:
            a = a.reshape(-1); axis = 0
        n = a.shape[axis]
        k = (n - 1) // 2
        return np.take(np.partition(a, k, axis=axis), k, axis=axis)


    def _resize_matrix(n_out, n_in):
        """Bilinear, align_corners=False (torch semantics), as an (n_out, n_in) matrix."""
        src = np.maximum((np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / n_out) - 0.5, 0.0)
        i0 = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
        i1 = np.minimum(i0 + 1, n_in - 1)
        w = (src - i0).astype(np.float32)
        M = np.zeros((n_out, n_in), np.float32)
        M[np.arange(n_out), i0] += 1 - w
        M[np.arange(n_out), i1] += w
        return M


    def resize_bilinear(a, H, W):
        Ay = _resize_matrix(H, a.shape[1]); Ax = _resize_matrix(W, a.shape[2])
        return np.einsum('Hh,chw,Ww->cHW', Ay, a.astype(np.float32), Ax, optimize=True).astype(np.float32)


    def _sub(x, max_px=4_000_000):
        st = max(1, int((x.shape[-1] * x.shape[-2] / max_px) ** 0.5))
        return x[:, ::st, ::st]


    def haar_sigma(x):
        x = np.asarray(x, np.float32)
        x = x[:, : x.shape[1] // 2 * 2, : x.shape[2] // 2 * 2]
        a, b, c, d_ = x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]
        d = (a - b - c + d_) / 2.0
        ok = np.isfinite(a) & np.isfinite(d_) & (a != 0) & np.isfinite(d)
        out = []
        for ch in range(x.shape[0]):
            v = _sub(d[ch:ch + 1])[_sub(ok[ch:ch + 1])]
            out.append(np.float32(1.4826) * lmedian(np.abs(v - lmedian(v))) if v.size > 100 else np.float32(0.0))
        return np.array(out, np.float32)


    def image_stats(x):
        x = np.asarray(x, np.float32)
        xs = _sub(x)
        m = np.array([lmedian(np.nan_to_num(xs[c][np.isfinite(xs[c])])) for c in range(x.shape[0])], np.float32)
        s = haar_sigma(x)
        rng = np.array([np.quantile(np.nan_to_num(xs[c]).ravel(), 0.999) if xs[c].size < 16_000_000 else xs[c].max()
                        for c in range(x.shape[0])], np.float32) - m
        return m, np.maximum(s, 1e-5 * np.clip(np.abs(rng), 1e-12, None)).astype(np.float32)


    def local_sigma(x, block=32):
        """(C,H,W) -> local fine-scale noise sigma map (diagonal-Haar MAD per 32 px block, bilinear upsampled)."""
        C, H, W = x.shape
        x = x[:, : H // 2 * 2, : W // 2 * 2].astype(np.float32)
        d = np.abs(x[:, 0::2, 0::2] - x[:, 1::2, 0::2] - x[:, 0::2, 1::2] + x[:, 1::2, 1::2]) / 2.0
        b = block // 2
        h2, w2 = d.shape[1] // b * b, d.shape[2] // b * b
        if h2 == 0 or w2 == 0:
            return np.broadcast_to(haar_sigma(x)[:, None, None], (C, H, W)).astype(np.float32)
        t = np.nan_to_num(d[:, :h2, :w2]).reshape(C, h2 // b, b, w2 // b, b).transpose(0, 1, 3, 2, 4).reshape(C, h2 // b, w2 // b, b * b)
        med = np.float32(1.4826) * lmedian(t, axis=-1)
        return resize_bilinear(med, H, W)


    def _robust_sigma(d):
        out = []
        for c in range(d.shape[0]):
            v = _sub(d[c:c + 1]).ravel()
            v = v[np.isfinite(v)]
            out.append(np.float32(1.4826) * lmedian(np.abs(v - lmedian(v))) if v.size > 100 else np.float32(1.0))
        return np.maximum(np.array(out, np.float32), 1e-12)


    def noise_descriptor(x):
        """Noise description of the whole image for v3+ models (log binned/1x sigma at 2x and 4x, row/column anisotropy, 0)."""
        x = np.nan_to_num(np.asarray(x, np.float32))
        C, H, W = x.shape

        def det(z):
            z = z[:, : z.shape[1] // 2 * 2, : z.shape[2] // 2 * 2]
            a, b, c, d = z[:, 0::2, 0::2], z[:, 0::2, 1::2], z[:, 1::2, 0::2], z[:, 1::2, 1::2]
            return (a - b - c + d) / 2.0, (a + b - c - d) / 2.0, (a - b + c - d) / 2.0

        def pool(z, f):
            z = z[:, : z.shape[1] // f * f, : z.shape[2] // f * f]
            return z.reshape(C, z.shape[1] // f, f, z.shape[2] // f, f).mean((2, 4))

        dg, dh, dv = det(x)
        s1 = _robust_sigma(dg)
        s2 = 2.0 * _robust_sigma(det(pool(x, 2))[0])
        s4 = 4.0 * _robust_sigma(det(pool(x, 4))[0])
        v = np.array([np.log(s2 / s1).mean(), np.log(s4 / s1).mean(), np.log(_robust_sigma(dh) / _robust_sigma(dv)).mean(), 0.0], np.float32)
        return np.clip(v, -3.0, 3.0)


    def _blur(x, sigma):
        """Separable Gaussian (radius int(3 sigma)), reflect padding (replicate on tiny images); channels run in parallel."""
        from scipy.ndimage import correlate1d
        from concurrent.futures import ThreadPoolExecutor
        r = max(1, int(3 * sigma))
        t = np.arange(-r, r + 1, dtype=np.float32)
        k = np.exp(-0.5 * (t / sigma) ** 2); k = (k / k.sum()).astype(np.float32)
        mode = 'reflect' if min(x.shape[-2:]) > r else 'edge'

        def one(c):
            y = np.pad(c, ((0, 0), (r, r)), mode=mode)
            y = correlate1d(y, k, axis=1, mode='constant')[:, r:-r]
            y = np.pad(y, ((r, r), (0, 0)), mode=mode)
            return correlate1d(y, k, axis=0, mode='constant')[r:-r]
        with ThreadPoolExecutor(max_workers=min(len(x), 4)) as ex:
            return np.stack(list(ex.map(one, list(x)))).astype(np.float32)


    def _window(n, ov):
        w = np.ones(n, np.float32)
        if ov > 0:
            r = np.linspace(0, 1, ov + 2, dtype=np.float32)[1:-1]
            r = (0.5 - 0.5 * np.cos(np.pi * r)).astype(np.float32)
            w[:ov] = r; w[-ov:] = r[::-1]
        return w


    def _mirror_to(a, n):
        """(C, T, T) -> (C, n, n), n >= T, by mirror-tiling with period 2T (symmetric: ... c b a | a b c ... c b a | a b c ...)."""
        T = a.shape[-1]
        idx = np.arange(n) % (2 * T)
        idx = np.where(idx >= T, 2 * T - 1 - idx, idx)
        return np.ascontiguousarray(a[:, idx][:, :, idx])


    def residuals(net, xn, want_pattern, cond=None, progress=None, cancel=None):
        """xn (3,H,W) normalised -> r0 (and r1 = pattern-removing residual when want_pattern), sigma units.
        net(x, c, cond) runs the fixed 512 tile.  Same tiling as umbra.infer.residuals."""
        C, H, W = xn.shape
        nout = 2 if want_pattern else 1
        if max(H, W) < TILE:
            # PyTorch tile for a small image: T = ceil(max/32)*32 (min 32), one tile, padded to T x T (reflect, else replicate)
            Tt = min(TILE, max(32, int(math.ceil(max(H, W) / 32)) * 32))
            mode = 'reflect' if (Tt - H < H and Tt - W < W) else 'edge'
            xc = K_COMP * np.arcsinh(np.pad(xn, ((0, 0), (0, Tt - H), (0, Tt - W)), mode=mode) / K_COMP)
            t = _mirror_to(xc, TILE)[None] if Tt < TILE else np.ascontiguousarray(xc[None], np.float32)
            out = []
            for j in range(nout):
                _check(cancel)
                out.append(net(t.astype(np.float32, copy=False), float(j), cond)[0][:, :H, :W])
            if progress:
                progress(1, 1)
            return out[0], (out[1] if want_pattern else None)
        T, st = TILE, TILE - OVERLAP
        ny = max(1, math.ceil((H - OVERLAP) / st)) if H > T else 1
        nx = max(1, math.ceil((W - OVERLAP) / st)) if W > T else 1
        Hp, Wp = max(H, (ny - 1) * st + T), max(W, (nx - 1) * st + T)
        mode = 'reflect' if (Hp - H < H and Wp - W < W) else 'edge'
        xc = K_COMP * np.arcsinh(np.pad(xn, ((0, 0), (0, Hp - H), (0, Wp - W)), mode=mode) / K_COMP)
        acc = np.zeros((nout, C, Hp, Wp), np.float32); wsum = np.zeros((Hp, Wp), np.float32)
        w2 = _window(T, OVERLAP if ny > 1 else 0)[:, None] * _window(T, OVERLAP if nx > 1 else 0)[None, :]
        coords = [(iy * st, ix * st) for iy in range(ny) for ix in range(nx)]
        for n, (y0, x0) in enumerate(coords, 1):
            _check(cancel)
            t = np.ascontiguousarray(xc[None, :, y0:y0 + T, x0:x0 + T], np.float32)
            for j in range(nout):
                acc[j, :, y0:y0 + T, x0:x0 + T] += net(t, float(j), cond)[0] * w2
            wsum[y0:y0 + T, x0:x0 + T] += w2
            if progress:
                progress(n, len(coords))
        r = (acc / np.maximum(wsum, 1e-8))[:, :, :H, :W]
        return r[0], (r[1] if want_pattern else None)


    def denoise_chw(net, y, d=1.0, p=0.0, detail=0.0, progress=None, cancel=None):
        Y = np.ascontiguousarray(y, np.float32)
        fin = np.isfinite(Y)
        Y0 = np.nan_to_num(Y)
        m, s = image_stats(Y0)
        m, s = m[:, None, None], s[:, None, None]
        xn = (Y0 - m) / s
        cond = noise_descriptor(xn)[None]
        r0, r1 = residuals(net, xn, p > 0, cond, progress, cancel)
        _check(cancel)
        ls = local_sigma(Y0) / s                                                     # regional noise guard
        rms = np.sqrt(np.clip(_blur(r0 * r0, 16.0), 0, None))
        g = np.minimum(GUARD_K * ls / np.maximum(rms, 1e-12), 1.0)
        g = _blur(g, 8.0)
        r0g = r0 * g
        if detail and detail > 0:                                                    # Detail: give back part of the removed fine / mid residual
            D = float(min(max(detail, 0.0), 1.0))
            b1 = _blur(r0g, 1.2); b16 = _blur(r0g, 16.0)
            r0g = (r0g - b1) * (1.0 - 0.6 * D) + (b1 - b16) * (1.0 - D) + b16
        r0g = r0g - _blur(r0g, MEAN_FIX)                                             # never move levels / colours at >= ~32 px
        out = xn + d * r0g
        if r1 is not None:
            pat = r1 - r0
            pat = pat - _blur(pat, 32.0)
            out = out + p * pat
        out = out * s + m
        return np.where(fin, out, Y).astype(np.float32)


    def denoise_array(net, data, d=1.0, p=0.0, detail=0.0, progress=None, cancel=None):
        """2D mono or (C,H,W), C in {1, 3}. Mono is replicated to 3 channels and averaged back."""
        a = np.asarray(data, np.float32)
        if a.ndim == 2:
            cube = a[None]
        elif a.ndim == 3 and a.shape[0] in (1, 3):
            cube = a
        else:
            raise ValueError(f'unsupported shape {a.shape}')
        if cube.shape[0] == 3:
            out = denoise_chw(net, cube, d, p, detail, progress, cancel)
        else:
            out = denoise_chw(net, np.repeat(cube, 3, 0), d, p, detail, progress, cancel).mean(0, keepdims=True)
        return out[0] if a.ndim == 2 else out


    # ----------------------------------------------------------------------------------------------- saturated-star protection (verbatim from the Siril script)
    TIGHTEN = 0.6          # luminance never drops below background + 0.6 x (input - background) inside a protected star
    COLOR_SNR = 8.0        # input colour is used as is above this many noise sigmas over background, smoothed below it
    FOOT_FRAC = 0.12       # footprint = input luminance above background + 12 % of the star's peak above background
    _INT_MAX = (255.0, 1023.0, 4095.0, 16383.0, 65535.0)


    def _looks_clipped(c, m):
        """True when channel max `m` looks like a clipping level: integer-format maximum, or shared by >= 3 pixels."""
        if m in _INT_MAX:
            return True
        return int(np.count_nonzero(c >= m)) >= 3


    def _plateau_cores(c, hi=0.8, flat=0.99, min_px=3):
        """Flat-topped star cores at ANY level (float stacks rescaled so the clip level is below the channel max, often different per
        channel): connected regions above hi x channel max whose pixels within (1 - flat) of the region's own peak number >= min_px.
        A real (unclipped) star peak has only 1-2 pixels that close to its maximum."""
        from scipy import ndimage as ndi
        m = float(c.max())
        if not np.isfinite(m) or m <= 0:
            return None
        high = c >= hi * m
        if not high.any() or high.sum() > c.size // 20:
            return None
        lab, n = ndi.label(high, structure=np.ones((3, 3), bool))
        idx = np.flatnonzero(high)                                  # work on the few bright pixels only (ndi.maximum over the full frame took ~1.7 s)
        ll, vv = lab.ravel()[idx], c.ravel()[idx]
        pk = np.full(n + 1, -np.inf, np.float32)
        np.maximum.at(pk, ll, vv)
        nr = vv >= flat * pk[ll]
        cnt = np.bincount(ll[nr], minlength=n + 1)
        keep = cnt >= min_px
        keep[0] = False
        sel = idx[nr & keep[ll]]
        if sel.size == 0:
            return None
        out = np.zeros(c.shape, bool)
        out.ravel()[sel] = True
        return out


    def find_saturated_cores(data, frac=0.97):
        """Core mask (H, W) of saturated / near-saturated stars in a HxW or CxHxW image, or None.
        Per channel (union across channels, single-channel clips included): pixels >= frac * channel max, when the image looks clipped
        (integer-format max or max shared by >= 3 pixels) and the pixel is star-like (>= 2 of its 8 neighbours above half its value, or
        1 above half and 3 above 10%, or the other channels above 10% there, or >= 3 neighbours of the luminance above 30% of the
        luminance there; hot pixels have a dark neighbourhood and are rejected)."""
        planes = data[None] if data.ndim == 2 else data
        lumi = planes.mean(0) if len(planes) > 1 else planes[0]
        core = None
        for c in planes:                          # flat-topped cores below the channel max (rescaled float stacks)
            pc = _plateau_cores(c)
            if pc is not None:
                core = pc if core is None else (core | pc)
        for k, c in enumerate(planes):
            m = float(c.max())
            if not np.isfinite(m) or m <= 0 or not _looks_clipped(c, m):
                continue
            ys, xs = np.nonzero(c >= m * frac)
            if ys.size == 0 or ys.size > c.size // 20:
                continue
            H, W = c.shape
            v = c[ys, xs]
            lv = lumi[ys, xs]
            n50 = np.zeros(ys.size, np.int32)
            n10 = np.zeros(ys.size, np.int32)
            nl30 = np.zeros(ys.size, np.int32)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy, xx = ys + dy, xs + dx
                    ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
                    nb = np.zeros(ys.size, np.float32)
                    nb[ok] = c[yy[ok], xx[ok]]
                    n50 += (nb > 0.5 * v)
                    n10 += (nb > 0.1 * v)
                    if len(planes) > 1:
                        nl = np.zeros(ys.size, np.float32)
                        nl[ok] = lumi[yy[ok], xx[ok]]
                        nl30 += (nl > 0.3 * lv)
            keep = (n50 >= 2) | ((n50 >= 1) & (n10 >= 3))       # star-like: hot pixels have a dark neighbourhood
            if len(planes) > 1:                                   # or the other channels / the luminance light up at the same pixel (a real point source)
                oth = (sum(p[ys, xs] for j, p in enumerate(planes) if j != k) / (len(planes) - 1))
                keep |= (oth > 0.1 * v) | (nl30 >= 3)
            if not keep.any():
                continue
            cc = np.zeros((H, W), bool)
            cc[ys[keep], xs[keep]] = True
            core = cc if core is None else (core | cc)
        return core


    def protect_saturated(inp, out):
        """Saturated-star protection v3 (model independent): returns (result, n_stars).
        Each clipped star (>= 1 clipped pixel in any channel) gets a footprint from the INPUT luminance (above background + 12 % of the
        star's peak, grown 1-2 px, radius capped at 25 px; at least as large as the v2 ring), feathered at its edge. Inside it the colour
        (channel / mean ratio) follows the input, the whole core included; clipped pixels (small stars; large ones also their 1 px rim) are the input exactly. Luminance keeps
        the model's change but never drops below background + 0.6 x (input - background) and never exceeds max(input, smoothed model
        luminance). No stars found: `out` is returned untouched."""
        from scipy import ndimage as ndi
        core = find_saturated_cores(inp)
        if core is None or not core.any():
            return out, 0
        inp = np.asarray(inp, np.float32)
        out = np.asarray(out, np.float32)
        box = np.ones((3, 3), bool)
        dil = ndi.binary_dilation(core, structure=box)
        lab, n = ndi.label(dil, structure=box)
        area = np.bincount(lab[dil], minlength=n + 1).astype(np.float32)
        rad = np.sqrt(np.maximum(area, 1.0) / np.pi)
        H, W = dil.shape
        mono = inp.ndim == 2
        mi = inp if mono else inp.mean(0)
        mo = out if mono else out.mean(0)
        wc = np.zeros((H, W), np.float32)         # protection weight (1 = input colour); also the footprint feather
        wl = np.ones((H, W), np.float32)          # luminance weight (1 = model luminance); ramps up over 2 px outside the core
        bgm = np.zeros((H, W), np.float32)        # local background per pixel (from the star whose weight is largest)
        msm = np.empty_like(mo)                   # smoothed model luminance (ceiling), filled inside the windows
        floor_e = 1e-6 * max(float(mi.max()), 1.0)
        rsm = None if mono else np.empty_like(inp)  # smoothed input colour ratio (channel / mean), filled inside the windows
        for i, sl in enumerate(ndi.find_objects(lab), start=1):
            if sl is None:
                continue
            r_i = float(rad[i])
            R_ring = min(20.0, 3.0 + 2.0 * r_i)                      # v2 ring radius (kept as a minimum extent)
            R_cap = min(25.0, 6.0 + 3.0 * r_i)                       # footprint radius cap
            F = 2.0 + 0.1 * r_i                                      # footprint edge feather
            m = int(np.ceil(max(R_cap + F, R_ring))) + 6
            y0, y1 = max(sl[0].start - m, 0), min(sl[0].stop + m, H)
            x0, x1 = max(sl[1].start - m, 0), min(sl[1].stop + m, W)
            lc = lab[y0:y1, x0:x1]
            mc = lc == i
            d = ndi.distance_transform_edt(~mc).astype(np.float32)
            mw = mi[y0:y1, x0:x1]
            ann = (d > R_cap) & (d <= R_cap + 4.0) & (lc == 0)
            bg = float(np.percentile(mw[ann], 25)) if ann.sum() >= 8 else float(np.percentile(mw, 25))
            peak = float(mw[mc].max())
            thr = bg + FOOT_FRAC * max(peak - bg, 0.0)
            cand = (ndi.gaussian_filter(mw, 0.7) > thr) & (d <= R_cap)
            cl, _ = ndi.label(cand | mc, structure=box)
            fp = np.isin(cl, np.unique(cl[mc]))
            fp = ndi.binary_fill_holes(fp)
            g = 5 if r_i > 3.0 else 3
            fp = ndi.maximum_filter(fp.view(np.uint8), size=g).astype(bool) | mc      # grown 1 px (2 for large stars)
            dfp = ndi.distance_transform_edt(~fp).astype(np.float32)
            t = np.clip(dfp / F, 0, 1)
            w = 1 - t * t * (3 - 2 * t)                              # plateau 1 inside the footprint, smooth edge
            tr = np.clip(d / R_ring, 0, 1)
            w = np.maximum(w, 1 - tr * tr * (3 - 2 * tr))            # never less than the v2 ring
            dl = d if r_i > 3.0 else ndi.distance_transform_edt(~(mc & core[y0:y1, x0:x1])).astype(np.float32)   # luminance eases in from the clipped pixels
            t2 = np.clip(dl / 2.0, 0, 1)
            l = t2 * t2 * (3 - 2 * t2)
            wcs = wc[y0:y1, x0:x1]
            upd = w > wcs
            bgm[y0:y1, x0:x1] = np.where(upd, bg, bgm[y0:y1, x0:x1])
            msm[y0:y1, x0:x1] = np.where(upd, ndi.gaussian_filter(mo[y0:y1, x0:x1], 1.0), msm[y0:y1, x0:x1])
            if not mono:        # input colour ratio: the pixel's own where it is bright (high S/N), a flux-weighted smoothed one in the faint halo (no pixel-noise speckle)
                sg = 1.0 + 0.15 * r_i
                iw = inp[:, y0:y1, x0:x1]
                num = ndi.gaussian_filter(iw, (0, sg, sg))
                den = ndi.gaussian_filter(mw, sg)
                rs = np.where(den > floor_e, num / np.maximum(den, floor_e), 1.0)
                ri_w = np.where(mw > floor_e, iw / np.maximum(mw, floor_e), 1.0)
                sig = 1.4826 * float(np.median(np.abs(mw[ann] - np.median(mw[ann])))) if ann.sum() >= 8 else 0.0
                q = np.clip((mw - bg) / max(COLOR_SNR * sig, floor_e), 0, 1)
                q = q * q * (3 - 2 * q)
                rs = ri_w * q + rs * (1 - q)
                rsm[:, y0:y1, x0:x1] = np.where(upd[None], rs, rsm[:, y0:y1, x0:x1])
            wc[y0:y1, x0:x1] = np.maximum(wcs, w)
            wl[y0:y1, x0:x1] = np.minimum(wl[y0:y1, x0:x1], l)
        sel = np.nonzero(wc > 0)                          # everything below touches only these pixels (the protected footprints)
        hard = core[sel] | (dil[sel] & (area[lab[sel]] > 30))     # exactly the input: clipped pixels (small stars), plus the 1 px rim of large ones
        w_ = wc[sel]
        wf = np.clip(w_ * 1.5, 0, 1)
        mi_, mo_, bg_ = mi[sel], mo[sel], bgm[sel]
        lum = mi_ + (mo_ - mi_) * wl[sel]                     # model luminance, eased in from the input at the core edge
        lum = lum + (np.minimum(lum, np.maximum(mi_, msm[sel])) - lum) * wf       # no hot cores: not brighter than input / smoothed model
        floor = np.where(mi_ > bg_, bg_ + TIGHTEN * (mi_ - bg_), mi_)    # at most (1 - TIGHTEN) of the input's excess over background is removed
        lum = lum + (np.maximum(lum, floor) - lum) * wf
        res = out.copy()
        if mono:
            res[sel] = np.where(hard, inp[sel], lum)
        else:
            rm = np.where(mo_ > floor_e, out[(slice(None),) + sel] / np.maximum(mo_, floor_e), 1.0)    # colour = channel / mean (ratio), so it scales with the clamped luminance
            ratio = np.clip(rm + (rsm[(slice(None),) + sel] - rm) * w_, 0.0, 3.0)    # colour follows the input inside the footprint
            r_ = lum[None] * ratio
            r_ = np.minimum(r_, inp.reshape(len(inp), -1).max(1)[:, None])    # never above the input's clip level
            res[(slice(None),) + sel] = np.where(hard[None], inp[(slice(None),) + sel], r_)
        return res.astype(np.float32, copy=False), int(n)




    # ----------------------------------------------------------------------------------------------- Engine
    class Engine:
        provider = 'CPU'
        note = None

        def __init__(self, model_path, session_factory):
            self.model_path = str(model_path)
            self._run, self.provider, self.note = session_factory(self.model_path, probe_feed())

        def _net(self, x, c, cond=None):
            feed = {'x': x, 'c': np.array([c], np.float32),
                    'cond': np.asarray(cond if cond is not None else np.zeros((1, COND_CH)), np.float32)}
            return self._run(feed)[0]

        def run(self, data, params=None, progress=None, cancel=None):
            prm = dict(params or {})
            d = float(min(max(prm.get('denoise', 1.0), 0.0), 1.0))
            D = float(min(max(prm.get('detail', 0.0), 0.0), 1.0))
            p = float(min(max(prm.get('pattern', 0.0), 0.0), 1.0))
            data = np.ascontiguousarray(data, np.float32)
            if d == 0 and p == 0:
                return data.copy(), {'notes': [NOTHING_TO_DO], 'saturated': 0}
            out = denoise_array(self._net, data, d, p, D, progress, cancel)
            _check(cancel)
            out, n = protect_saturated(data, np.asarray(out, np.float32))        # always on, model independent
            notes = [f'Protected {n} saturated stars.'] if n else []
            return out, {'notes': notes, 'saturated': int(n)}
    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})


ENGINES['denoise'] = _engine_denoise()

def _engine_starless():
    """Umbra Starless engine: pure numpy + scipy + onnxruntime port of the Umbra Starless v0.0.4 pipeline (Remove Stars, then Apply
    with the default settings).  No torch, no Qt, no Siril.

    Pipeline (same order and maths as the PyTorch script):
      1. data mode: auto (is_stretched heuristic) / linear / stretched
      2. halo pass: the network runs on a 1/4-scale copy; only the extended glow of very bright saturated stars is kept (grey opening,
         seeds = big saturated cores) and subtracted before the main pass (large_halos)
      3. main pass: tiled network (fixed 512 x 512 tiles, smaller images reflect-padded to 512, 64 px cosine-blended overlap) on x - halo; whole-image normalisation (median offset,
         Haar noise sigma, asinh compression K=8); S = min(S, x)
      4. clean_residue + noise_fill: the starless image is the original outside star footprints; inside, the network answer is shifted
         to its surroundings, clamped to its local median +- tolerance, and grain-matched noise is added
      5. star layer K: linear  x - S;  stretched  unscreen 1 - (1 - x) / (1 - S)  (so S + K, or screen(S, K), gives back x)
      6. Reduce Stars: amount 1 = S exactly, 0 = x exactly, between = star layer un-stretched by one smooth curve (stars shrink)

    NOT included (not part of the published version): correction marks (Remove / Keep painting) and the Auto-keep critic.
    NaN pixels: filled from the nearest valid pixel before processing, restored to NaN in every output.

    params (all optional):
      amount  float 0..1, default 1.0   1 = full star removal, <1 = reduce stars, 0 = original
      mode    'auto' | 'linear' | 'stretched', default 'auto'
      output  'starless' | 'stars', default 'starless'   ('stars' returns the star layer as the image)
    info: {'notes': [str], 'stars': star layer (same shape/dtype as data, NaN where data is NaN), 'mode_used': 'linear'|'stretched',
           'sigma': global noise sigma}
    """
    import math
    import numpy as np

    TOOL = 'starless'
    TILE = 512
    OVERLAP = 64
    K_COMP = 8.0
    HALO_FACTOR = 4
    A_SPAN = 1.5
    FADE0 = 0.8


    def probe_feed():
        rng = np.random.default_rng(1)
        return {'x': (rng.standard_normal((1, 3, TILE, TILE)) * 3).astype(np.float32),
                'cond': (rng.random((1, 1, TILE, TILE)) > 0.9).astype(np.float32)}


    # ------------------------------------------------------------------------------------------------ numpy / scipy replacements of torch ops
    def _resize_bilinear(a, H, W):
        """(N,h,w) -> (N,H,W), bilinear, align_corners=False, no antialias (same as torch F.interpolate)."""
        n, h, w = a.shape
        def idx(inn, out):
            src = np.maximum((np.arange(out, dtype=np.float64) + 0.5) * (inn / out) - 0.5, 0.0)
            i0 = np.minimum(np.floor(src).astype(np.int64), inn - 1)
            i1 = np.minimum(i0 + 1, inn - 1)
            return i0, i1, (src - i0).astype(np.float32)
        y0, y1, ly = idx(h, H)
        x0, x1, lx = idx(w, W)
        t = a[:, y0, :] * (1 - ly)[None, :, None] + a[:, y1, :] * ly[None, :, None]
        return (t[:, :, x0] * (1 - lx)[None, None, :] + t[:, :, x1] * lx[None, None, :]).astype(np.float32)


    def _gblur(a, s):
        """Gaussian blur of the last two axes (any leading dims).  Wide blurs run at 1/f scale like the torch version."""
        from scipy import ndimage
        a = np.ascontiguousarray(a, np.float32)
        sh = a.shape
        t = a.reshape(-1, *sh[-2:])
        f = int(s // 2.5)
        if f >= 2 and min(sh[-2:]) >= 8 * f:
            H, W = sh[-2:]
            ph, pw = (-H) % f, (-W) % f
            tp = np.pad(t, ((0, 0), (0, ph), (0, pw)), mode='edge') if (ph or pw) else t
            n = tp.shape[0]
            d = tp.reshape(n, tp.shape[1] // f, f, tp.shape[2] // f, f).mean((2, 4), dtype=np.float32)
            d = _gblur(d, math.sqrt(max(s * s - f * f / 12.0, 0.25)) / f)
            u = _resize_bilinear(d.reshape(n, *d.shape[-2:]), d.shape[-2] * f, d.shape[-1] * f)[:, :H, :W]
            return u.reshape(sh)
        r = max(1, int(math.ceil(3 * s)))
        k = np.exp(-0.5 * (np.arange(-r, r + 1, dtype=np.float32) / np.float32(s)) ** 2).astype(np.float32)
        k /= k.sum()
        t = ndimage.convolve1d(t, k, axis=2, mode='mirror')
        t = ndimage.convolve1d(t, k, axis=1, mode='mirror')
        return t.reshape(sh).astype(np.float32)


    def _lmedian(v):
        """lower median (what torch.median returns)."""
        v = np.asarray(v).ravel()
        return v[np.argpartition(v, (v.size - 1) // 2)[(v.size - 1) // 2]]


    # ------------------------------------------------------------------------------------------------ normalisation (ust/norm.py)
    def _sub(x, max_px=4_000_000):
        st = max(1, int((x.shape[-1] * x.shape[-2] / max_px) ** 0.5))
        return x[:, ::st, ::st]


    def haar_sigma(x):
        x = x[:, : x.shape[1] // 2 * 2, : x.shape[2] // 2 * 2].astype(np.float32)
        d = (x[:, 0::2, 0::2] - x[:, 1::2, 0::2] - x[:, 0::2, 1::2] + x[:, 1::2, 1::2]) / 2.0
        ok = np.isfinite(x[:, 0::2, 0::2]) & np.isfinite(x[:, 1::2, 1::2]) & (x[:, 0::2, 0::2] != 0) & np.isfinite(d)
        out = []
        for c in range(x.shape[0]):
            v = _sub(d[c:c + 1])[_sub(ok[c:c + 1])]
            if v.size > 100:
                med = _lmedian(v)
                out.append(np.float32(1.4826) * _lmedian(np.abs(v - med)))
            else:
                out.append(np.float32(0.0))
        return np.array(out, np.float32)


    def image_stats(x):
        xs = _sub(x.astype(np.float32))
        m = np.array([_lmedian(np.nan_to_num(xs[c][np.isfinite(xs[c])])) for c in range(x.shape[0])], np.float32)
        s = haar_sigma(x)
        rng = np.array([np.quantile(np.nan_to_num(xs[c]).ravel(), 0.999) for c in range(x.shape[0])], np.float32) - m
        return m, np.maximum(s, (1e-5 * np.maximum(np.abs(rng), 1e-12)).astype(np.float32))


    def _compress(xn):
        return (K_COMP * np.arcsinh(xn / K_COMP)).astype(np.float32)


    def _decompress(y):
        return (K_COMP * np.sinh(np.clip(y, -150, 150) / K_COMP)).astype(np.float32)


    # ------------------------------------------------------------------------------------------------ layout / data mode
    def is_stretched(x):
        s = x[..., ::4, ::4]
        med = float(np.median(s)); hi = float(np.percentile(s, 99.9))
        return med > 0.1 and med > 0.06 * max(hi, 1e-6)


    def to_chw(data):
        a = np.asarray(data, np.float32)
        if a.ndim == 2:
            return a[None].repeat(3, 0), '2d'
        if a.shape[0] in (1, 3):
            return (a.repeat(3, 0) if a.shape[0] == 1 else a), ('1hw' if a.shape[0] == 1 else 'chw')
        return np.moveaxis(a, -1, 0), 'hwc'


    def from_chw(a, lay):
        if lay == '2d':
            return a.mean(0)
        if lay == '1hw':
            return a.mean(0, keepdims=True)
        if lay == 'hwc':
            return np.moveaxis(a, 0, -1)
        return a


    def _fill_nan(x, bad):
        """Replace non-finite pixels (union over channels) by the nearest valid pixel."""
        from scipy import ndimage
        if bad.all():
            return np.zeros_like(x)
        _, (iy, ix) = ndimage.distance_transform_edt(bad, return_indices=True)
        return x[:, iy, ix]


    # ------------------------------------------------------------------------------------------------ the network, tiled
    def _tile_starts(n):
        return list(range(0, max(n - 2 * OVERLAP, 1), TILE - 2 * OVERLAP))


    def _n_tiles(H, W):
        return len(_tile_starts(H)) * len(_tile_starts(W))


    _PAD_FILL = {'mode': 'reflect'}


    def _pad_tile(a, ph, pw):
        """Pad a (3,h,w) tile to 512 x 512.  First reflect up to a multiple of 32 (what the torch script does), then fill the rest."""
        h, w = a.shape[-2:]
        p32h, p32w = (-h) % 32, (-w) % 32
        if p32h or p32w:
            a = np.pad(a, ((0, 0), (0, p32h), (0, p32w)), mode='reflect')
        rh, rw = ph - p32h, pw - p32w
        if rh or rw:
            m = _PAD_FILL['mode']
            a = np.pad(a, ((0, 0), (0, rh), (0, rw)), mode='reflect' if m == 'reflect' else ('edge' if m == 'edge' else 'constant'))
        return a


    def _starless_core(run_fn, x, tick, tile=TILE, overlap=OVERLAP):
        """x: (3,H,W) float32 -> (S in x's units, global noise sigma).  Tiled, cosine-blended overlap."""
        C, H, W = x.shape
        xt = np.ascontiguousarray(np.nan_to_num(x), np.float32)
        m, s = image_stats(xt)
        xc = _compress((xt - m[:, None, None]) / s[:, None, None])
        step = tile - 2 * overlap
        ys = range(0, max(H - 2 * overlap, 1), step); xs = range(0, max(W - 2 * overlap, 1), step)
        acc = np.zeros((3, H, W), np.float32); wacc = np.zeros((1, H, W), np.float32)
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, math.pi, overlap + 2, dtype=np.float32)[1:-1])).astype(np.float32)
        w1 = np.ones(tile, np.float32); w1[:overlap] = ramp; w1[-overlap:] = ramp[::-1]
        wt = (w1[:, None] * w1[None, :])[None]
        for y in ys:
            for x_ in xs:
                y0, x0 = min(y, max(H - tile, 0)), min(x_, max(W - tile, 0))
                a = xc[:, y0:y0 + tile, x0:x0 + tile]
                h, w = a.shape[-2:]
                ph, pw = tile - h, tile - w                  # the ONNX models are fixed at 512 x 512: always run exactly one full tile
                if ph or pw:
                    a = _pad_tile(a, ph, pw)
                feed = {'x': np.ascontiguousarray(a[None], np.float32), 'cond': np.zeros((1, 1, tile, tile), np.float32)}
                r = np.asarray(run_fn(feed)[0], np.float32)[0][:, :h, :w]
                z = _decompress(a[:, :h, :w] + r)
                ww = wt[:, :h, :w]
                acc[:, y0:y0 + h, x0:x0 + w] += z * ww
                wacc[:, y0:y0 + h, x0:x0 + w] += ww
                tick()
        zn = acc / np.maximum(wacc, 1e-6)
        return (zn * s[:, None, None] + m[:, None, None]).astype(np.float32), float(s.mean())


    def _large_halos(run_fn, x, tick, factor=HALO_FACTOR):
        """Extended glow of very bright saturated stars, to subtract before the main pass (default settings of the script)."""
        from scipy import ndimage
        C, H, W = x.shape
        h2, w2 = H // factor, W // factor
        if min(h2, w2) < 64:
            return np.zeros_like(x)
        xs = x[:, :h2 * factor, :w2 * factor].reshape(C, h2, factor, w2, factor).mean((2, 4), dtype=np.float32)
        Sd, sgd = _starless_core(run_fn, xs, tick)
        R = np.maximum(xs - Sd, 0)
        yy, xx = np.mgrid[-4:5, -4:5]; disc = (yy * yy + xx * xx) <= 16
        Rl = np.stack([ndimage.grey_opening(R[c], footprint=disc) for c in range(R.shape[0])])
        Rl = np.minimum(Rl, _gblur(Rl, 1.5))
        xl = x.mean(0); xmax = float(np.percentile(xl, 99.999))
        sat = (xl > 0.85 * xmax) | (xl - float(np.median(xl)) > 5000 * sgd * factor)
        sl, ns = ndimage.label(sat)
        if ns:
            area = ndimage.sum(sat, sl, np.arange(1, ns + 1))
            sat = np.isin(sl, np.nonzero(area >= 150)[0] + 1)
        if not sat.any():
            return np.zeros_like(x)
        seed = sat[:h2 * factor, :w2 * factor].reshape(h2, factor, w2, factor).any((1, 3))
        lab, _ = ndimage.label(_gblur(Rl.mean(0), 1.0) > 2 * sgd)
        ids = np.unique(lab[seed]); ids = ids[ids > 0]
        if len(ids) == 0:
            return np.zeros_like(x)
        Rl = Rl * np.clip(_gblur(np.isin(lab, ids).astype(np.float32), 2.0) * 1.3, 0, 1)[None]
        up = _resize_bilinear(Rl, h2 * factor, w2 * factor)
        out = np.zeros_like(x); out[:, :h2 * factor, :w2 * factor] = up
        return _gblur(out, factor / 2).astype(np.float32)


    def _starless(run_fn, x, tick):
        """halo pass, then the network at full resolution on x - halo.  Returns (S, sigma)."""
        Hh = _large_halos(run_fn, x, tick)
        S, sg = _starless_core(run_fn, x - Hh, tick)
        return np.minimum(S, x), sg


    # ------------------------------------------------------------------------------------------------ residue cleaning and noise fill
    def clean_residue(x, S, sigma, k=0.6, half_size=6, dil=3):
        from scipy import ndimage
        C, H, W = S.shape
        rem = _gblur((x - S).mean(0), 1.0) > 3.0 * sigma
        Fp = ndimage.binary_dilation(rem, iterations=dil)
        out_w = (~Fp).astype(np.float32)
        bias = _gblur((x - S) * out_w[None], 12.0) / np.maximum(_gblur(out_w, 12.0), 1e-3)[None]
        bias = np.where(_gblur(out_w, 12.0)[None] > 0.02, bias, 0)
        S1 = S + bias
        h2, w2 = H // 2, W // 2
        Sd = S1[:, :h2 * 2, :w2 * 2].reshape(C, h2, 2, w2, 2).mean((2, 4), dtype=np.float32)
        md = np.stack([ndimage.median_filter(Sd[c], size=half_size) for c in range(C)])
        med = _resize_bilinear(np.ascontiguousarray(md, np.float32), h2 * 2, w2 * 2)
        if med.shape[-2:] != S.shape[-2:]:
            med = np.pad(med, ((0, 0), (0, H - h2 * 2), (0, W - w2 * 2)), mode='edge')
        dev = np.abs(Sd.mean(0) - md.mean(0))
        tex = 1.4826 * ndimage.median_filter(dev, size=2 * half_size - 1)
        tex = _resize_bilinear(np.ascontiguousarray(tex, np.float32)[None], h2 * 2, w2 * 2)[0]
        if tex.shape != (H, W):
            tex = np.pad(tex, ((0, H - h2 * 2), (0, W - w2 * 2)), mode='edge')
        tol = np.maximum(k * sigma, 2.0 * tex)[None]
        inner = np.clip(S1, med - tol, med + tol)
        wf = np.clip(_gblur(Fp.astype(np.float32), 1.0) * 1.5, 0, 1)
        wf = np.maximum(wf, rem)
        out = x * (1 - wf[None]) + inner * wf[None]
        ex = out.mean(0) - med.mean(0)
        hot = ex > np.maximum(4 * sigma, 4 * tol[0])
        bump = ndimage.binary_dilation(hot, iterations=2) & ~ndimage.binary_opening(hot, iterations=3)
        out = np.where(bump[None], np.minimum(out, med + tol), out)
        return np.minimum(out, x).astype(np.float32)


    def noise_fill(x, S, sigma, seed=0):
        S = clean_residue(x, S, sigma)
        rem = (x - S).mean(0)
        sm = _gblur(rem, 1.0)
        where = (sm > 1.5 * sigma).astype(np.float32)
        if where.sum() < 4:
            return S
        reg = np.clip(_gblur(np.maximum(where, 0), 2.0) * 3, 0, 1)
        hp = S - _gblur(S, 1.0)
        loc = np.sqrt(np.maximum(_gblur(hp * hp, 8.0), 0)) * 1.75
        okw = 1 - np.clip(_gblur(where, 4.0) * 4, 0, 1)
        exp_sig = _gblur(loc * okw[None], 24.0) / np.maximum(_gblur(okw, 24.0), 1e-3)[None]
        add = np.sqrt(np.maximum(exp_sig ** 2 - loc ** 2, 0))
        sky = okw > 0.99
        d = hp.mean(0)
        rho = float(np.sum(d[:, :-1] * d[:, 1:] * sky[:, :-1]) / max(np.sum(d[:, :-1] ** 2 * sky[:, :-1]), 1e-12)) if sky.any() else 0.0
        rng = np.random.default_rng(seed)
        nz = rng.standard_normal(S.shape).astype(np.float32)
        if rho > 0.05:
            sb = math.sqrt(-1 / (4 * math.log(min(max(rho, 0.06), 0.95))))
            nz = _gblur(nz, sb); nz /= max(float(nz.std()), 1e-6)
        return np.minimum(S + reg[None] * add * nz, x).astype(np.float32)


    # ------------------------------------------------------------------------------------------------ star layer and Reduce Stars
    def star_layer(x, S, stretched):
        if stretched:
            return np.clip(1 - (1 - x) / np.maximum(1 - S, 1e-4), 0, 1).astype(np.float32)
        return (x - S).astype(np.float32)


    def combine(S, K, stretched):
        return (1 - (1 - S) * (1 - K)) if stretched else (S + K)


    def display_stretch(S):
        s = S[:, ::4, ::4].reshape(S.shape[0], -1)
        med = np.median(s, 1); mad = 1.4826 * np.median(np.abs(s - med[:, None]), 1)
        b = med - 2.8 * mad; vm = float(np.clip(np.mean((med - b) / (1 - b)), 1e-7, 0.2))
        m = (0.25 - 1) * vm / ((2 * 0.25 - 1) * vm - 0.25)
        return b.astype(np.float32)[:, None, None], float(m)


    def _mtf(v, m):
        return (m - 1) * v / ((2 * m - 1) * v - m)


    def _T(x, b, m):
        v = (x - b) / (1 - b)
        return np.where(v >= 0, _mtf(np.clip(v, 0, 1), m), v * (1 - m) / m)


    def _Ti(y, b, m):
        v = np.where(y >= 0, _mtf(np.clip(y, 0, 1), 1 - m), y * m / (1 - m))
        return b + (1 - b) * v


    def reduce_gain(L, r):
        a = 10.0 ** (-A_SPAN * r)
        g = a / (1 - (1 - a) * np.clip(L, 0, 1))
        if r > FADE0:
            t = (r - FADE0) / (1 - FADE0)
            g = g * (1 - t * t * (3 - 2 * t))
        return g


    def reduce_stars(S, K, r, stretched=False):
        if r <= 0:
            return combine(S, K, stretched)
        if r >= 1:
            return S.copy()
        if stretched:
            D = np.clip(K, 0, 1)
            return combine(S, D * reduce_gain(D.mean(0), r)[None], True).astype(np.float32)
        X = S + np.maximum(K, 0)
        b, mm = display_stretch(S)
        tS, tX = _T(S, b, mm), _T(X, b, mm)
        D = np.clip(1 - (1 - tX) / np.maximum(1 - tS, 1e-4), 0, 1)
        tO = 1 - (1 - tS) * (1 - D * reduce_gain(D.mean(0), r)[None])
        return np.minimum(np.maximum(_Ti(tO, b, mm), S), X).astype(np.float32)


    # ------------------------------------------------------------------------------------------------ engine
    class Engine:
        def __init__(self, model_path, session_factory):
            self.model_path = model_path
            self._run, self.provider, self.note = session_factory(model_path, probe_feed())

        def run(self, data, params=None, progress=None, cancel=None):
            p = dict(params or {})
            amount = float(min(max(p.get('amount', 1.0), 0.0), 1.0))
            mode = p.get('mode', 'auto')
            output = p.get('output', 'starless')
            if mode not in ('auto', 'linear', 'stretched'):
                raise ValueError("mode must be 'auto', 'linear' or 'stretched'")
            if output not in ('starless', 'stars'):
                raise ValueError("output must be 'starless' or 'stars'")
            data = np.asarray(data, np.float32)
            x, lay = to_chw(data)
            bad = ~np.isfinite(x)
            notes = []
            if bad.any():
                badu = bad.any(0)
                x = np.ascontiguousarray(_fill_nan(x, badu), np.float32)
                notes.append(f'{int(badu.sum())} NaN pixels kept as NaN')
            else:
                badu = None
                x = np.ascontiguousarray(x, np.float32)
            detected = bool(is_stretched(x))
            stretched = detected if mode == 'auto' else (mode == 'stretched')
            mode_used = 'stretched' if stretched else 'linear'
            notes.append(f'data: {mode_used}' + (' (auto)' if mode == 'auto' else ' (manual)'))
            H, W = x.shape[-2:]
            total = _n_tiles(H, W) + (_n_tiles(H // HALO_FACTOR, W // HALO_FACTOR) if min(H, W) // HALO_FACTOR >= 64 else 0)
            done = [0]

            def tick():
                if cancel is not None and cancel():
                    raise RuntimeError('Cancelled')
                done[0] += 1
                if progress is not None:
                    progress(min(done[0], total), total)
            if cancel is not None and cancel():
                raise RuntimeError('Cancelled')
            S0, sg = _starless(self._run, x, tick)
            S = noise_fill(x, S0, sg)
            K = star_layer(x, S, stretched)
            if output == 'stars':
                out = K
            elif amount <= 0:
                out = x.copy()
            else:
                out = reduce_stars(S, K, amount, stretched)
            out = np.ascontiguousarray(from_chw(out, lay), np.float32)
            stars = np.ascontiguousarray(from_chw(K, lay), np.float32)
            if badu is not None:
                bm = badu if lay in ('2d', '1hw', 'chw') else badu
                if lay == '2d':
                    out[bm] = np.nan; stars[bm] = np.nan
                elif lay in ('1hw', 'chw'):
                    out[:, bm] = np.nan; stars[:, bm] = np.nan
                else:
                    out[bm] = np.nan; stars[bm] = np.nan
            if progress is not None:
                progress(total, total)
            if self.note:
                notes.append(str(self.note))
            return out, {'notes': notes, 'stars': stars, 'mode_used': mode_used, 'sigma': float(sg)}
    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})


ENGINES['starless'] = _engine_starless()

def _engine_gradient():
    """Umbra Gradient engine: AI background (gradient / sky glow) extraction for linear images.

    Pure numpy + scipy + onnxruntime port of the Umbra BGNet 0.1.3 processing:
      1. star-robust block median of the whole image down to 512x512 (per colour channel, invalid pixels ignored),
      2. area resize to 256x256 (the network's fixed input), per-image normalisation with multi-stretch features,
      3. network (ONNX, fixed input 1x12x256x256, output a 3x64x64 control grid),
      4. bicubic 64 -> 256, Gaussian de-ripple (sigma 1.5 px of the 256 grid), bicubic 256 -> full size,
      5. back to image units, additive subtraction of the model minus its mean (the pedestal is kept).
    Replacements for the torch / OpenCV ops (verified numerically against them in tests/test_gradient_engine.py):
    cv2 INTER_AREA -> exact-overlap area weights, torch bicubic (align_corners=False, A=-0.75) -> separable
    weight matrices, torch replicate-padded conv -> scipy.ndimage.convolve1d(mode='nearest').

    params (all optional, {} is the current behaviour): none yet. Unknown keys are ignored.
    info: 'notes' (list of str), 'background' (the gradient model, same shape and dtype as the input image)."""
    import math
    import numpy as np

    TOOL = 'gradient'
    TILE = 256          # fixed network input side
    CACHE = 512         # block-median target side
    GRID_SIGMA = 1.5    # de-ripple sigma in pixels of the 256 px working grid
    _CACHE = {}


    def probe_feed():
        rng = np.random.default_rng(1)
        return {'x': rng.standard_normal((1, 12, TILE, TILE)).astype(np.float32)}


    # ------------------------------------------------------------------------------------------ resampling
    def _area_matrix(n_in, n_out):
        """Weights (n_out, n_in) of cv2.INTER_AREA: each output cell averages the exact (fractional) input span it covers."""
        key = ('area', n_in, n_out)
        if key in _CACHE:
            return _CACHE[key]
        scale = n_in / n_out
        W = np.zeros((n_out, n_in), np.float64)
        if scale >= 1:
            for d in range(n_out):
                a, b = d * scale, (d + 1) * scale
                s0, s1 = int(math.floor(a)), min(n_in, int(math.ceil(b)))
                for s in range(s0, s1):
                    W[d, s] = (min(b, s + 1) - max(a, s)) / scale
        else:  # enlargement: cv2 falls back to linear-like fractional overlap with unit-width source cells
            for d in range(n_out):
                a, b = d * scale, (d + 1) * scale
                s0 = int(math.floor(a))
                if s0 + 1 >= b or s0 + 1 >= n_in:
                    W[d, min(s0, n_in - 1)] = 1.0
                else:
                    w0 = (s0 + 1 - a) / scale
                    W[d, s0] = w0
                    W[d, s0 + 1] = 1 - w0
        W = W.astype(np.float32)
        _CACHE[key] = W
        return W


    def _cubic_matrix(n_in, n_out, A=-0.75):
        """Weights (n_out, n_in) of torch bicubic interpolation, align_corners=False, no antialias, border clamp."""
        key = ('cubic', n_in, n_out)
        if key in _CACHE:
            return _CACHE[key]
        scale = n_in / n_out
        M = np.zeros((n_out, n_in), np.float64)
        i = np.arange(n_out)
        src = (i + 0.5) * scale - 0.5
        f = np.floor(src)
        t = src - f

        def c1(x):  # |x| <= 1
            return ((A + 2) * x - (A + 3)) * x * x + 1

        def c2(x):  # 1 < |x| < 2
            return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A
        w = [c2(t + 1), c1(t), c1(1 - t), c2(2 - t)]
        for k in range(4):
            idx = np.clip(f.astype(np.int64) - 1 + k, 0, n_in - 1)
            np.add.at(M, (i, idx), w[k])
        M = M.astype(np.float32)
        _CACHE[key] = M
        return M


    def _resample(a, Ry, Rx):
        """Apply Ry (H', H) along axis 0 and Rx (W', W) along axis 1 of an HxW array."""
        return (Ry @ a.astype(np.float32, copy=False)) @ Rx.T


    def area_resize(a, size):
        """cv2.resize(a, size=(W, H), INTER_AREA) for an HxW or HxWxC float32 array."""
        W, H = size
        Ry, Rx = _area_matrix(a.shape[0], H), _area_matrix(a.shape[1], W)
        if a.ndim == 2:
            return _resample(a, Ry, Rx)
        return np.stack([_resample(a[..., c], Ry, Rx) for c in range(a.shape[2])], -1)


    def bicubic_resize(a, size):
        """torch F.interpolate(mode='bicubic', align_corners=False) of an HxW array to size=(H, W)."""
        return _resample(a, _cubic_matrix(a.shape[0], size[0]), _cubic_matrix(a.shape[1], size[1]))


    def deripple(g, sigma=GRID_SIGMA):
        """Separable Gaussian blur of (C, H, W) with replicated borders (kernel radius int(3 sigma + 0.5))."""
        from scipy import ndimage
        r = int(3 * sigma + 0.5)
        t = np.arange(-r, r + 1, dtype=np.float32)
        k = np.exp(-0.5 * (t / np.float32(sigma)) ** 2).astype(np.float32)
        k /= k.sum()
        g = ndimage.convolve1d(g, k, axis=2, mode='nearest')
        return ndimage.convolve1d(g, k, axis=1, mode='nearest')


    # ------------------------------------------------------------------------------------------ preprocessing
    def valid_mask(img):
        return np.isfinite(img).all(-1) & ~(img == 0).all(-1)


    def block_median(img, mask, out=CACHE):
        from scipy import ndimage
        H, W, _ = img.shape
        b = max(1, min(H, W) // out)
        h, w = H // b, W // b
        small = np.empty((h, w, 3), np.float32)
        m = mask[: h * b, : w * b]
        frac = m.reshape(h, b, w, b).mean(axis=(1, 3))
        full = bool(m.all())
        for c in range(3):
            ch = img[: h * b, : w * b, c]
            if not full:
                ch = ch.copy()
                ch[~m] = np.nan
            blk = ch.reshape(h, b, w, b).transpose(0, 2, 1, 3).reshape(h, w, b * b)
            if full:
                small[..., c] = np.median(blk, axis=-1)
            else:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    small[..., c] = np.nanmedian(blk, axis=-1)
        cmask = (frac >= 0.25) & np.isfinite(small).all(-1)
        small = np.nan_to_num(small)
        if not cmask.all():
            if cmask.any():
                idx = ndimage.distance_transform_edt(~cmask, return_distances=False, return_indices=True)
                small = small[idx[0], idx[1]]
            else:
                small = np.zeros_like(small)
        if (h, w) != (out, out):
            small = area_resize(small, (out, out))
            cmask = area_resize(cmask.astype(np.float32), (out, out)) > 0.5
        return small, cmask


    def normalize(x, m):
        mv = m if m.any() else np.ones_like(m)
        o = np.array([np.percentile(x[c][mv], 1) for c in range(3)], np.float32)
        xc = x - o[:, None, None]
        s = float(max(np.percentile(xc[c][mv], 99) for c in range(3)))
        s = max(s, 1e-12 * max(1.0, float(np.abs(o).max())), 1e-30)
        xl = np.clip(xc / s, -0.5, 4.0)
        f = np.empty((12,) + x.shape[1:], np.float32)
        f[0:3] = xl
        f[3:6] = np.arcsinh(30 * xl) / math.asinh(30)
        f[6:9] = np.arcsinh(1000 * xl) / math.asinh(1000)
        f[0:9] *= m
        f[9] = m
        t = np.linspace(-1, 1, x.shape[-1], dtype=np.float32)
        f[10:12] = np.stack(np.meshgrid(t, t, indexing='xy'))
        return f, o, s


    # ------------------------------------------------------------------------------------------ engine
    class Engine:
        def __init__(self, model_path, session_factory):
            self._run, self.provider, self.note = session_factory(model_path, probe_feed())

        def run(self, data, params=None, progress=None, cancel=None):
            data = np.asarray(data)
            if data.ndim == 2:
                img3 = np.repeat(data[:, :, None], 3, axis=-1)
                mono = True
            elif data.ndim == 3 and data.shape[0] in (1, 3):
                mono = data.shape[0] == 1
                img3 = np.moveaxis(data, 0, -1)
                if mono:
                    img3 = np.repeat(img3, 3, axis=-1)
            else:
                raise ValueError('Expected (H, W) or (C, H, W) with C in {1, 3}.')
            img3 = np.ascontiguousarray(img3, dtype=np.float32)
            steps = 5

            def tick(i):
                if cancel is not None and cancel():
                    raise RuntimeError('Cancelled')
                if progress is not None:
                    progress(i, steps)
            notes = []
            tick(0)
            mask = valid_mask(img3)
            H, W = img3.shape[:2]
            if not mask.any():
                notes.append('No valid pixels: image returned unchanged.')
                bg = np.zeros_like(data, dtype=np.float32)
                return data.copy(), {'notes': notes, 'background': bg}
            small, cm = block_median(img3, mask)
            tick(1)
            x = np.moveaxis(area_resize(small, (TILE, TILE)), -1, 0).copy()
            m = area_resize(cm.astype(np.float32), (TILE, TILE)) > 0.5
            f, o, s = normalize(x, m)
            tick(2)
            grid = np.asarray(self._run({'x': f[None].astype(np.float32)})[0], np.float32)[0]
            tick(3)
            g = np.stack([bicubic_resize(grid[c], (TILE, TILE)) for c in range(3)])
            g = deripple(g)
            B = np.empty((H, W, 3), np.float32)
            for c in range(3):
                B[..., c] = bicubic_resize(g[c], (H, W))
            B = B * np.float32(s) + o[None, None, :]
            tick(4)
            Bc = B - B[mask].mean(0, dtype=np.float64).astype(np.float32)
            out3 = img3.copy()
            out3[mask] = img3[mask] - Bc[mask]
            if mono:
                out, bg = out3[..., 0], B[..., 0]
                if data.ndim == 3:
                    out, bg = out[None], bg[None]
            else:
                out, bg = np.moveaxis(out3, -1, 0), np.moveaxis(B, -1, 0)
            out = np.ascontiguousarray(out, dtype=data.dtype if data.dtype == np.float32 else np.float32)
            bg = np.ascontiguousarray(bg, dtype=np.float32)
            if not mask.all():
                notes.append(f'{int((~mask).sum())} invalid (NaN/zero) pixels left unchanged.')
            tick(5)
            return out, {'notes': notes, 'background': bg}
    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})


ENGINES['gradient'] = _engine_gradient()

# ============================================================================================ tools
def _tool_deblur():
    # ---- Umbra Deblur: tool layer (panel, CLI flags, labels) ----
    SHARP_DEFAULT = 3.0
    STRENGTH_TIP = 'How far to sharpen toward the sampling limit; always limited by the noise. 0 = Correct Only.'
    STAR_TIP = 'Shrinks stars toward the sampling limit (noise-limited). Only this slider changes star size.'
    HALO_TIP = 'Pulls star halos in; their light goes back into the star core (brightness kept).'
    CORRECT_TIP = 'Fixes star shapes (aberrations) without sharpening. Switches the other sliders off.'
    SHARP_TIP = ('Number of sharpening passes (PSF re-measured each time); 2.5 = 2 passes + a half-strength pass. 0 = Correct Only. '
                 'Umbra Deblur v1.0: every step is sharper than the last; 8 is the strongest; cleaner dark dust lanes. '
                 'v1.1 (experimental): sharper fine strands, less beading; sky mask on automatically; may keep slight coma tails on bright '
                 'corner stars unless Protect star shapes is ticked. v1.0 stays the default. Images differ, so judge by eye.')
    PROTECT_TIP = ('Fine-detail model (v1.1) only: stars and their coma tails come from Umbra Deblur v1.0 (best star shapes), all other '
                   'detail from v1.1. About 2x slower.')
    PSF_TIP = 'Manual star diameter (FWHM) in pixels.'
    STAR_MODEL_FILES = ('umbra_deblur_v1.0.onnx', 'umbra_deconvolution_1.0.onnx')


    def _is_drizzled(text):
        return 'drizzle' in str(text or '').lower()


    class DeblurPanel:
        def __init__(self, ui, layout, host):
            self.ui, self.host = ui, host
            self.fb = True
            self.finedetail = False
            self.correct = ui.check(layout, 'Correct Only', CORRECT_TIP)
            self.strength = ui.slider(layout, 'Strength', 0, 100, 50, lambda v: f'{v}%', STRENGTH_TIP)
            self.sharp = ui.slider(layout, 'Sharpening', 0, 32, int(SHARP_DEFAULT * 4), lambda v: f'{v / 4:.2f}', SHARP_TIP)   # 0..8 in 0.25 steps
            self.protect = ui.check(layout, 'Protect star shapes (slower)', PROTECT_TIP)
            self.auto = ui.check(layout, 'Automatic PSF', 'Measures the star size from the image. Switch off to set the PSF Diameter by hand.', True)
            self.psf = ui.slider(layout, 'PSF Diameter (px)', 15, 100, 30, lambda v: f'{v / 10:.1f}', PSF_TIP)
            self.star = ui.slider(layout, 'Sharpen Stars', 0, 100, 0, lambda v: f'{v}%', STAR_TIP)
            self.halo = ui.slider(layout, 'Halo Reduction', 0, 100, 0, lambda v: f'{v}%', HALO_TIP)
            self.correct.toggled.connect(lambda _=False: self.update())
            self.auto.toggled.connect(lambda _=False: self.update())
            self.update()

        def update(self):
            co = self.correct.isChecked()
            self.ui.enable(self.strength, not co); self.ui.enable(self.sharp, not co)
            self.ui.show(self.strength, not self.fb); self.ui.show(self.sharp, self.fb)
            self.protect.setVisible(self.finedetail)
            self.ui.enable(self.psf, not self.auto.isChecked())
            self.ui.enable(self.star, not co); self.ui.enable(self.halo, not co)

        def model_changed(self, path):
            eng = ENGINES['deblur']
            self.fb = eng.is_fb(path) if path else True
            self.finedetail = bool(path) and eng.is_finedetail(path)
            self.update()

        def star_model(self):
            d = self.ui.model_dir()
            for f in STAR_MODEL_FILES:
                if (d / f).is_file():
                    return str(d / f)
            return None

        def params(self):
            p = {'correct_only': self.correct.isChecked(), 'fwhm': None if self.auto.isChecked() else self.psf.value() / 10.0,
                 'star': self.star.value() / 100.0, 'halo': self.halo.value() / 100.0}
            if self.fb:
                p['sharpening'] = self.sharp.value() / 4.0
            else:
                p['strength'] = self.strength.value() / 100.0
            if self.protect.isVisible() and self.protect.isChecked():
                p['protect_stars'] = True
                p['star_model_path'] = self.star_model()
            return p

        def settings(self):
            return dict(correct=self.correct.isChecked(), strength=self.strength.value(), sharp=self.sharp.value(), protect=self.protect.isChecked(),
                        auto=self.auto.isChecked(), psf=self.psf.value(), star=self.star.value(), halo=self.halo.value())

        def set_settings(self, d):
            for k, w in (('strength', self.strength), ('sharp', self.sharp), ('psf', self.psf), ('star', self.star), ('halo', self.halo)):
                if isinstance(d.get(k), int):
                    w.setValue(d[k])
            for k, w in (('correct', self.correct), ('protect', self.protect), ('auto', self.auto)):
                if isinstance(d.get(k), bool):
                    w.setChecked(d[k])
            self.update()


    class DeblurSpec(ToolSpec):
        def build_panel(self, ui, layout, host):
            return DeblurPanel(ui, layout, host)

        def add_cli(self, ap):
            ap.add_argument('--strength', type=float, default=0.5, help='deblur: 0..1 (default 0.5; models without Sharpening only)')
            ap.add_argument('--sharpening', type=float, help='deblur: passes 0..8 in steps of 0.25 (default 3; 2.5 = 2 passes + half pass)')
            ap.add_argument('--stars', type=float, default=0.0, help='deblur: Sharpen Stars 0..1 (default 0)')
            ap.add_argument('--halo', type=float, default=0.0, help='deblur: Halo Reduction 0..1 (default 0)')
            ap.add_argument('--correct-only', action='store_true', help='deblur: force strength, sharpening, stars and halo to 0')
            ap.add_argument('--fwhm', type=float, help='deblur: manual PSF FWHM in px (default: automatic)')
            ap.add_argument('--protect-stars', action='store_true', help='deblur: v1.1 only: star shapes from v1.0 (~2x slower)')

        def check_cli(self, a, ap):
            if not 0 <= a.strength <= 1:
                ap.error('--strength must lie in [0, 1]')
            if a.fwhm is not None and not 0.5 <= a.fwhm <= 20:
                ap.error('--fwhm must be between 0.5 and 20 px')
            if a.sharpening is not None and not 0 <= a.sharpening <= 8:
                ap.error('--sharpening must lie in 0..8')
            if not 0 <= a.stars <= 1 or not 0 <= a.halo <= 1:
                ap.error('--stars and --halo must lie in [0, 1]')

        def params_from_args(self, a):
            p = {'correct_only': bool(a.correct_only), 'fwhm': a.fwhm, 'star': a.stars, 'halo': a.halo, 'strength': a.strength}
            if a.sharpening is not None:
                p['sharpening'] = a.sharpening
            if a.protect_stars:
                p['protect_stars'] = True
                d = model_dir(self)[0]
                p['star_model_path'] = next((str(d / f) for f in STAR_MODEL_FILES if (d / f).is_file()), None)
            return p

        def image_params(self, header_text):
            return {'drizzled': _is_drizzled(header_text)}      # drizzled stacks have correlated noise: the engine raises its noise estimate

        def label(self, p):
            head = f"sharpening {float(p['sharpening']):.2f}" if 'sharpening' in p else f"strength {float(p.get('strength', 0.5)):.2f}"
            if p.get('correct_only'):
                head = 'correct only'
            psf = 'auto PSF' if p.get('fwhm') is None else f"PSF {float(p['fwhm']):.1f} px"
            return f"{head} stars {float(p.get('star', 0)):.2f} halo {float(p.get('halo', 0)):.2f} {psf}" + (' protect-stars' if p.get('protect_stars') else '')


    SPEC = DeblurSpec(
        key='deblur', name='Umbra Deblur', script='Umbra_Deblur', repo=f'{GITHUB_USER}/umbra-deblur', legacy_repos=(f'{GITHUB_USER}/umbra-deconvolve',),
        prefix='UMBRA_DEBLUR_', folder='Umbra Deblur', fallback_folder='Umbra Deconvolve',
        labels={'umbra_deblur_v1.0': 'Umbra Deblur v1.0 (default)', 'umbra_deblur_v1.1_experimental': 'Umbra Deblur v1.1 (experimental)',
                'umbra_deconvolution_1.0': 'Umbra Deblur v1.0 (default)', 'umbra_deconvolution_1.1a_experimental': 'Umbra Deblur v1.1 (experimental)'},
        default_stems=('umbra_deblur_v1.0', 'umbra_deconvolution_1.0'),
        notes=['AI deblur (deconvolution) for linear images.'])
    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})


TOOLS['deblur'] = _tool_deblur().SPEC

def _tool_denoise():
    # ---- Umbra Denoise: tool layer ----
    DENOISE_TIP = 'Strength of the noise removal. 100% = full noise removal, 0 = off (Pattern noise can still run on its own).'
    DETAIL_TIP = ('Keeps part of what Denoise removed, split by scale: faint mottled structure (nebula texture, outer halos) comes back '
                  'together with a little grain. 0 = full noise removal. 25-50 suits faint, low-contrast targets.')
    PATTERN_TIP = ('For PATTERN noise only: blotchy mottle repeated in every sub (calibration / no dithering). Random noise is removed by Denoise. '
                   'Pattern cannot be told apart from real faint signal with certainty, so it may remove some. Leave at 0 unless you see blotches. '
                   'Works on its own, without Denoise.')


    class DenoisePanel:
        def __init__(self, ui, layout, host):
            pct = lambda v: f'{v}%'
            self.denoise = ui.slider(layout, 'Denoise', 0, 100, 100, pct, DENOISE_TIP)
            self.detail = ui.slider(layout, 'Detail', 0, 100, 0, pct, DETAIL_TIP)
            self.pattern = ui.slider(layout, 'Pattern noise', 0, 100, 0, pct, PATTERN_TIP)

        def model_changed(self, path):
            pass

        def params(self):
            return {'denoise': self.denoise.value() / 100.0, 'detail': self.detail.value() / 100.0, 'pattern': self.pattern.value() / 100.0}

        def settings(self):
            return dict(denoise=self.denoise.value(), detail=self.detail.value(), pattern=self.pattern.value())

        def set_settings(self, d):
            for k in ('denoise', 'detail', 'pattern'):
                if isinstance(d.get(k), int):
                    getattr(self, k).setValue(d[k])


    class DenoiseSpec(ToolSpec):
        def build_panel(self, ui, layout, host):
            return DenoisePanel(ui, layout, host)

        def add_cli(self, ap):
            ap.add_argument('--denoise', type=float, default=1.0, help='denoise: noise removal 0..1 (default 1)')
            ap.add_argument('--detail', type=float, default=0.0, help='denoise: gives back fine detail 0..1 (default 0)')
            ap.add_argument('--pattern', type=float, default=0.0, help='denoise: pattern-noise removal 0..1 (default 0)')

        def check_cli(self, a, ap):
            if not (0 <= a.denoise <= 1 and 0 <= a.detail <= 1 and 0 <= a.pattern <= 1):
                ap.error('--denoise, --detail and --pattern must lie in [0, 1]')

        def params_from_args(self, a):
            return {'denoise': a.denoise, 'detail': a.detail, 'pattern': a.pattern}

        def label(self, p):
            return f"d {float(p.get('denoise', 1)):.2f} detail {float(p.get('detail', 0)):.2f} pattern {float(p.get('pattern', 0)):.2f}"


    SPEC = DenoiseSpec(
        key='denoise', name='Umbra Denoise', script='Umbra_Denoise', repo=f'{GITHUB_USER}/umbra-denoise',
        prefix='UMBRA_DENOISE_', folder='Umbra Denoise',
        labels={'umbra_denoise_v3.1_experimental': 'Umbra Denoise v3.1 (experimental)'},
        default_stems=('umbra_denoise_v3.1_experimental',),
        notes=['AI denoise for linear images.'])
    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})


TOOLS['denoise'] = _tool_denoise().SPEC

def _tool_starless():
    # ---- Umbra Starless: tool layer ----
    REMOVAL_TIP = ('100% = all stars gone, 0% = stars untouched (the original image). In between every star gets smaller and dimmer '
                   'until it vanishes.')
    DATA_TIP = 'Linear or stretched. Auto detects it from the image; override if the guess is wrong.'
    OUTPUT_TIP = 'Stars only is the full star layer (ignores Star removal).'
    SAVE_TIP = 'Saves the star layer as <image>_stars.fit in the same folder (a new file; the image is not touched).'


    class StarlessPanel:
        def __init__(self, ui, layout, host):
            self.removal = ui.slider(layout, 'Star removal', 0, 100, 100, lambda v: f'{v}%', REMOVAL_TIP)
            self.data = ui.combo(layout, 'Data', [('Auto', 'auto'), ('Linear', 'linear'), ('Stretched', 'stretched')], DATA_TIP)
            self.out = ui.combo(layout, 'Output', [('Starless image', 'starless'), ('Stars only', 'stars')], OUTPUT_TIP)
            self.save = ui.check(layout, 'Also save the star layer as FITS next to the image', SAVE_TIP)

        def model_changed(self, path):
            pass

        def params(self):
            return {'amount': self.removal.value() / 100.0, 'mode': self.data.currentData(), 'output': self.out.currentData()}

        def settings(self):
            return dict(removal=self.removal.value(), data=self.data.currentData(), output=self.out.currentData(), save=self.save.isChecked())

        def set_settings(self, d):
            if isinstance(d.get('removal'), int):
                self.removal.setValue(d['removal'])
            for k, w in (('data', self.data), ('output', self.out)):
                i = w.findData(d.get(k))
                if i >= 0:
                    w.setCurrentIndex(i)
            if isinstance(d.get('save'), bool):
                self.save.setChecked(d['save'])


    class StarlessSpec(ToolSpec):
        def build_panel(self, ui, layout, host):
            return StarlessPanel(ui, layout, host)

        def add_cli(self, ap):
            ap.add_argument('--removal', type=float, default=1.0, help='starless: star removal 0..1 (1 = remove, lower = reduce stars; default 1)')
            ap.add_argument('--data', choices=['auto', 'linear', 'stretched'], default='auto', help='starless: data type (default auto)')
            ap.add_argument('--output', choices=['starless', 'stars'], default='starless', help='starless: write the starless image or the star layer')
            ap.add_argument('--stars-out', help='starless: also write the star layer to this FITS file (must not exist)')

        def check_cli(self, a, ap):
            if not 0 <= a.removal <= 1:
                ap.error('--removal must lie in [0, 1]')

        def params_from_args(self, a):
            return {'amount': a.removal, 'mode': a.data, 'output': a.output}

        def preview_params(self, p):
            return dict(p, output='starless')       # the preview shows the starless result; the star layer is one click away

        def label(self, p):
            return f"removal {float(p.get('amount', 1)):.2f} data {p.get('mode', 'auto')} output {p.get('output', 'starless')}"

        def post_apply(self, ctx, params, info):
            if not ctx.view.panel.save.isChecked() or info.get('stars') is None:
                return ''
            path = side_path(ctx.siril, 'stars')
            write_fits(path, info['stars'], siril_fits_header(ctx.siril), ctx.history + ' (star layer)')
            return f'Star layer saved to {path}.'

        def post_cli(self, a, dst, info, header, label):
            if not a.stars_out:
                return ''
            p = Path(a.stars_out).resolve()
            if p.exists() or p == dst:
                raise SystemExit(f'Output exists, refusing to overwrite: {p}')
            h = header.copy(); h['HISTORY'] = label + ' (star layer)'
            write_fits(p, info['stars'], h)
            return f'stars={p}'


    SPEC = StarlessSpec(
        key='starless', name='Umbra Starless', script='Umbra_Starless', repo=f'{GITHUB_USER}/umbra-starless',
        prefix='UMBRA_STARLESS_', folder='Umbra Starless',
        labels={'umbra_starless_v1.0': 'Umbra Starless v1.0 (default)', 'umbra_starless_v1.1_experimental': 'Umbra Starless v1.1 (experimental)'},
        default_stems=('umbra_starless_v1.0',),
        notes=['AI star remover for linear or stretched images.'], linear_only=False, alt=('Show star layer', 'stars'))
    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})


TOOLS['starless'] = _tool_starless().SPEC

def _tool_gradient():
    # ---- Umbra Gradient: tool layer ----
    SAVE_TIP = 'Saves the gradient model that was subtracted as <image>_gradient.fit in the same folder (a new file).'


    class GradientPanel:
        def __init__(self, ui, layout, host):
            self.save = ui.check(layout, 'Also save the gradient model as FITS next to the image', SAVE_TIP)

        def model_changed(self, path):
            pass

        def params(self):
            return {}

        def settings(self):
            return dict(save=self.save.isChecked())

        def set_settings(self, d):
            if isinstance(d.get('save'), bool):
                self.save.setChecked(d['save'])


    class GradientSpec(ToolSpec):
        def build_panel(self, ui, layout, host):
            return GradientPanel(ui, layout, host)

        def add_cli(self, ap):
            ap.add_argument('--gradient-out', help='gradient: also write the gradient model to this FITS file (must not exist)')

        def label(self, p):
            return 'background extraction'

        def post_apply(self, ctx, params, info):
            if not ctx.view.panel.save.isChecked() or info.get('background') is None:
                return ''
            path = side_path(ctx.siril, 'gradient')
            write_fits(path, info['background'], siril_fits_header(ctx.siril), ctx.history + ' (gradient model)')
            return f'Gradient model saved to {path}.'

        def post_cli(self, a, dst, info, header, label):
            if not a.gradient_out:
                return ''
            p = Path(a.gradient_out).resolve()
            if p.exists() or p == dst:
                raise SystemExit(f'Output exists, refusing to overwrite: {p}')
            h = header.copy(); h['HISTORY'] = label + ' (gradient model)'
            write_fits(p, info['background'], h)
            return f'gradient={p}'


    SPEC = GradientSpec(
        key='gradient', name='Umbra Gradient', script='Umbra_Gradient', repo=f'{GITHUB_USER}/umbra-gradient',
        prefix='UMBRA_GRADIENT_', folder='Umbra Gradient',
        labels={'umbra_gradient_v7.0': 'Umbra Gradient v7.0 (default)'}, default_stems=('umbra_gradient_v7.0',),
        notes=['AI background (gradient) removal for linear images.', 'Crop stacking edges before running.'],
        preview_whole=True, reuse_preview=True, alt=('Show gradient model', 'background'), progress_unit='step')
    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})


TOOLS['gradient'] = _tool_gradient().SPEC


if __name__ == '__main__':
    raise SystemExit(main([TOOLS[k] for k in ('deblur', 'denoise', 'starless', 'gradient',)], combined=True, doc=__doc__))
