ENV_BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${ENV_BASE_DIR}:${PYTHONPATH}"
export PATH="${ENV_BASE_DIR}/bin:${PATH}"
export RPMDGLE="${ENV_BASE_DIR}"

unset ENV_BASE_DIR
