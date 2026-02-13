set -e

cd poc-datascaler-proxy

pip install uv
uv sync --frozen --allow-insecure-host pypi.org
