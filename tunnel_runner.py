import secrets
import sys
import time
from pathlib import Path

from gradio import networking
from gradio.tunneling import CURRENT_TUNNELS


LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = 7860
URL_FILE = Path("gradio_live_url.txt")


def start_tunnel():
    token = secrets.token_urlsafe(32)
    url = networking.setup_tunnel(
        local_host=LOCAL_HOST,
        local_port=LOCAL_PORT,
        share_token=token,
        share_server_address=None,
        share_server_tls_certificate=None,
    )
    URL_FILE.write_text(url, encoding="utf-8")
    print(f"Running on public URL: {url}", flush=True)
    return CURRENT_TUNNELS[-1]


def main():
    while True:
        try:
            tunnel = start_tunnel()
            while tunnel.proc is not None and tunnel.proc.poll() is None:
                time.sleep(5)
            print("Tunnel process stopped. Restarting...", flush=True)
        except Exception as e:
            print(f"Tunnel error: {e}", file=sys.stderr, flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
