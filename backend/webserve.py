"""Serve the capture client on 0.0.0.0 and proxy /v1 to the API.

HTTPS is on by default when PORT is not 80 (LAN phones need a secure context).
Docker / Nginx Proxy Manager keep PORT=80 HTTP so NPM can terminate TLS.
"""
import http.client
import http.server
import os
import shutil
import socket
import ssl
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

WEB = Path(__file__).resolve().parent / "web"
CERT_DIR = WEB / "certs"
UPSTREAM = urlparse(os.getenv("TRAFFIC_API", "http://127.0.0.1:61616"))
PORT = int(os.getenv("PORT", "61612"))
HOST = os.getenv("HOST", "0.0.0.0")
TLS = os.getenv("TRAFFIC_TLS", "0" if PORT == 80 else "1") == "1"


def local_ips():
    found = {"127.0.0.1"}
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(item[4][0])
    except OSError:
        pass
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        found.add(probe.getsockname()[0])
    except OSError:
        pass
    finally:
        probe.close()
    return [ip for ip in sorted(found) if not ip.startswith("127.")] + ["127.0.0.1"]


def openssl_bin():
    env = os.getenv("OPENSSL")
    if env:
        return env
    found = shutil.which("openssl")
    if found:
        return found
    for path in (
        r"C:\Home\Software\anaconda\Library\bin\openssl.exe",
        r"C:\Program Files\Git\usr\bin\openssl.exe",
    ):
        if Path(path).is_file():
            return path
    raise FileNotFoundError("openssl not on PATH; set OPENSSL")


def ensure_cert():
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    cert, key = CERT_DIR / "cert.pem", CERT_DIR / "key.pem"
    if cert.is_file() and key.is_file():
        return cert, key
    sans = ["DNS:localhost", "DNS:cam.muqin.ccwu.cc"] + [f"IP:{ip}" for ip in local_ips()]
    openssl = openssl_bin()
    cmd = [openssl, "req", "-x509", "-newkey", "rsa:2048", "-sha256", "-days", "3650", "-nodes",
           "-keyout", str(key), "-out", str(cert), "-subj", "/CN=traffic-lan",
           "-addext", "subjectAltName=" + ",".join(sans)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        cfg = CERT_DIR / "san.cnf"
        cfg.write_text("[req]\ndistinguished_name=dn\n[dn]\n[v3]\nsubjectAltName=" + ",".join(sans) + "\n", encoding="ascii")
        subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-sha256", "-days", "3650", "-nodes",
                        "-keyout", str(key), "-out", str(cert), "-subj", "/CN=traffic-lan",
                        "-config", str(cfg), "-extensions", "v3"], check=True, capture_output=True, text=True)
    return cert, key


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def do_GET(self):
        if self._blocked() or (self._api() and self.proxy("GET")):
            return
        return super().do_GET()

    def do_HEAD(self):
        if self._blocked() or (self._api() and self.proxy("HEAD")):
            return
        return super().do_HEAD()

    def do_POST(self):
        self.proxy("POST") if self._api() else self.send_error(404)

    def do_PUT(self):
        self.proxy("PUT") if self._api() else self.send_error(404)

    def do_DELETE(self):
        self.proxy("DELETE") if self._api() else self.send_error(404)

    def do_OPTIONS(self):
        self.proxy("OPTIONS") if self._api() else self.send_error(404)

    def _blocked(self):
        path = self.path.split("?", 1)[0]
        if path.startswith("/certs") or path.endswith(".pem") or path.endswith(".cnf"):
            self.send_error(404)
            return True
        return False

    def _api(self):
        path = self.path.split("?", 1)[0]
        return path.startswith("/v1") or path in ("/health", "/docs", "/openapi.json")

    def proxy(self, method):
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length else None
        conn = http.client.HTTPConnection(UPSTREAM.hostname, UPSTREAM.port or 80, timeout=600)
        try:
            skip = {"host", "content-length", "connection"}
            headers = {k: v for k, v in self.headers.items() if k.lower() not in skip}
            conn.request(method, self.path, body=body, headers=headers)
            response = conn.getresponse()
            payload = response.read()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in {"transfer-encoding", "connection", "content-encoding"}:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if method != "HEAD":
                self.wfile.write(payload)
        finally:
            conn.close()
        return True

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main():
    server = http.server.ThreadingHTTPServer((HOST, PORT), Handler)
    scheme = "http"
    if TLS:
        cert, key = ensure_cert()
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    urls = "  ".join(f"{scheme}://{ip}:{PORT}" for ip in local_ips() if ip != "127.0.0.1")
    extra = "  (certificate warning: continue)" if TLS else ""
    print(f"Client {scheme} {HOST}:{PORT} → {UPSTREAM.geturl()}  {urls}{extra}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
