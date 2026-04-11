#!/usr/bin/env bash
# Restore a knowledge-db PostgreSQL backup.
# Usage: ./scripts/restore.sh backup_20260411_120000.sql

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <backup_file.sql>"
    exit 1
fi

BACKUP_FILE="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: file not found: $BACKUP_FILE"
    exit 1
fi

# Load .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

echo "Warning: this will drop and recreate the '${POSTGRES_DB:-knowledge}' database."
read -p "Continue? [y/N] " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Aborted."
    exit 0
fi

docker compose -f "$PROJECT_DIR/docker-compose.yml" exec -T postgres \
    psql -U "${POSTGRES_USER:-knowledge}" -d postgres \
    -c "DROP DATABASE IF EXISTS ${POSTGRES_DB:-knowledge};" \
    -c "CREATE DATABASE ${POSTGRES_DB:-knowledge} OWNER ${POSTGRES_USER:-knowledge};"

docker compose -f "$PROJECT_DIR/docker-compose.yml" exec -T postgres \
    psql -U "${POSTGRES_USER:-knowledge}" "${POSTGRES_DB:-knowledge}" \
    < "$BACKUP_FILE"

echo "Restore complete."
