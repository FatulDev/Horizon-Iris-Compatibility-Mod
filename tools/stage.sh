#!/bin/bash
# Applies the bundle to the pinned pristine source, into ~/hz-stage.
set -e
HM=${HM:-$HOME/hm-main/target/debug/horizon-mixins}
rm -rf $HOME/iris-mod/mods && mkdir -p $HOME/iris-mod/mods && cp -r $HOME/iris-mod/out/iris-features $HOME/iris-mod/mods/; [ -n "$TESTS" ] && cp -r $HOME/iris-mod/out/iris-features-tests $HOME/iris-mod/mods/ || true
$HM apply --workspace $HOME/hz-pristine --mods $HOME/iris-mod/mods --out $HOME/hz-stage > "${TMPDIR:-/tmp}/iris-apply.json" 2>&1 || { cat "${TMPDIR:-/tmp}/iris-apply.json"; exit 1; }
echo applied
