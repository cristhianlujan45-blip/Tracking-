"""Copy text to the system clipboard from the app's own process.

The desktop window's web view can refuse the browser clipboard API, so "Copiar diagnóstico" asks
the local server, which runs on the same PC, to put the text on the clipboard.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time


def copy(text: str) -> bool:
    try:
        if sys.platform == "win32":
            return _copy_windows(text)
        for tool in (["pbcopy"], ["wl-copy"], ["xclip", "-selection", "clipboard"]):
            if shutil.which(tool[0]):
                return subprocess.run(tool, input=text.encode(), timeout=5).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
    return False


def _copy_windows(text: str) -> bool:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.OpenClipboard.argtypes, user32.OpenClipboard.restype = [wintypes.HWND], wintypes.BOOL
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.SetClipboardData.argtypes, user32.SetClipboardData.restype = [wintypes.UINT, wintypes.HANDLE], wintypes.HANDLE
    user32.CloseClipboard.restype = wintypes.BOOL
    kernel32.GlobalAlloc.argtypes, kernel32.GlobalAlloc.restype = [wintypes.UINT, ctypes.c_size_t], wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes, kernel32.GlobalLock.restype = [wintypes.HGLOBAL], wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]

    data = text.encode("utf-16-le") + b"\0\0"
    for _ in range(20):  # another program may hold the clipboard for a moment
        if user32.OpenClipboard(None):
            break
        time.sleep(0.05)
    else:
        return False
    try:
        user32.EmptyClipboard()
        handle = kernel32.GlobalAlloc(0x0002, len(data))  # GMEM_MOVEABLE
        if not handle:
            return False
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            kernel32.GlobalFree(handle)
            return False
        ctypes.memmove(pointer, data, len(data))
        kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(13, handle):  # CF_UNICODETEXT; the clipboard owns it from here
            kernel32.GlobalFree(handle)
            return False
        return True
    finally:
        user32.CloseClipboard()
