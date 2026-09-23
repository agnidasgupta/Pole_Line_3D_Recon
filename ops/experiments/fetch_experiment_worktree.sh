#!/usr/bin/env bash
# Fetch a complete experiment revision listed in experiments/REGISTRY.tsv.

main() {
  local repo="${1:-}" experiment_id="${2:-}" destination="${3:-}"
  local row source_ref source_commit
  if [ -z "$repo" ] || [ -z "$experiment_id" ] || [ -z "$destination" ]; then
    printf 'Usage: bash fetch_experiment_worktree.sh REPO EXPERIMENT_ID DESTINATION\n'
    return 0
  fi
  row=$(awk -F '\t' -v wanted="$experiment_id" '$1 == wanted {print $2 "\t" $3; exit}' \
    "$repo/experiments/REGISTRY.tsv")
  if [ -z "$row" ]; then
    printf 'EXPERIMENT_NOT_FOUND=%s\n' "$experiment_id"
    return 0
  fi
  source_ref=${row%%$'\t'*}
  source_commit=${row#*$'\t'}
  git -C "$repo" fetch origin "+refs/heads/$source_ref:refs/remotes/origin/$source_ref"
  git -C "$repo" worktree add "$destination" "$source_commit"
  printf 'EXPERIMENT_WORKTREE=%s\nSOURCE_REF=%s\nSOURCE_COMMIT=%s\n' \
    "$destination" "$source_ref" "$source_commit"
}

main "$@"
