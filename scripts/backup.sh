#!/usr/bin/env bash
# Backup the rag-base PostgreSQL database to a timestamped SQL file.
# Usage: ./scripts/backup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$PROJECT_DIR/backup_${TIMESTAMP}.sql"

docker compose -f "$PROJECT_DIR/docker-compose.yml" exec -T postgres \
    pg_dump -U "${POSTGRES_USER:-knowledge}" "${POSTGRES_DB:-knowledge}" \
    > "$BACKUP_FILE"

echo "Backup saved to: $BACKUP_FILE"
