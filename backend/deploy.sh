#!/bin/sh
# Pull, fetch models if missing, recreate containers. Point NPM at 127.0.0.1:61616
# for https://traffic.muqin.ccwu.cc. Phone default is that HTTPS origin.
set -eu
cd "$(dirname "$0")"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1 && [ -n "$(git remote 2>/dev/null || true)" ]; then
  git pull --ff-only
fi

if [ ! -f .env ]; then
  umask 077
  printf 'TRAFFIC_API_TOKEN=%s\n' "$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" > .env
  echo "Created .env. Open access is on (TRAFFIC_OPEN=1). Do not commit .env."
fi

if [ ! -f models/yolox_s.onnx ]; then
  python3 download_model.py
fi

docker compose up -d --build --wait
echo "Listening on 127.0.0.1:61616 — NPM: traffic.muqin.ccwu.cc → http://127.0.0.1:61616"
echo "Web client on 127.0.0.1:61612 — NPM: cam.muqin.ccwu.cc → http://127.0.0.1:61612"
echo "Open hello: phones/web/iOS POST /v1/hello, no token to type. NPM should forward X-Real-IP."
