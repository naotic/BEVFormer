#!/usr/bin/env bash
set -euo pipefail
repo=${1:?repository required}
commit=${2:?commit required}
target=/opt/src/$repo
mkdir -p "$target"
git -C "$target" init --quiet
GIT_TERMINAL_PROMPT=0 git -C "$target" \
    -c credential.helper= \
    -c 'credential.helper=!f() { if [ "$1" = get ]; then printf "username=x-access-token\npassword=%s\n" "$(cat /run/secrets/github_token)"; fi; }; f' \
    fetch --quiet --depth=1 "https://github.com/naotic/$repo.git" "$commit"
git -C "$target" -c advice.detachedHead=false checkout --quiet --detach FETCH_HEAD
rm -rf "$target/.git"
