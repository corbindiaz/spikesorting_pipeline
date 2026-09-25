import json
import os
import shutil

DEFAULTS_FILE = 'params.default.json'
USER_FILE = 'params.json'

def merge_params(old, new_defaults):
    merged = dict(new_defaults)  # start from new schema
    for k, v in old.items():
        if k in merged:
            if isinstance(v, dict) and isinstance(merged[k], dict):
                merged[k] = merge_params(v, merged[k])
            else:
                merged[k] = v  # keep user's existing value
        # keys removed from new schema are dropped silently
    return merged

def main():
    if not os.path.exists(DEFAULTS_FILE):
        # Nothing to merge against; skip quietly so the hook never blocks a pull
        return

    with open(DEFAULTS_FILE) as f:
        new_defaults = json.load(f)

    if not os.path.exists(USER_FILE):
        # First time setup: just copy defaults over
        shutil.copy(DEFAULTS_FILE, USER_FILE)
        print(f"Created {USER_FILE} from defaults.")
        return

    with open(USER_FILE) as f:
        old = json.load(f)

    updated = merge_params(old, new_defaults)

    if updated == old:
        return  # nothing changed, no need to rewrite

    with open(USER_FILE, 'w') as f:
        json.dump(updated, f, indent=2)

    print(f"{USER_FILE} updated with new parameters from {DEFAULTS_FILE}.")

if __name__ == '__main__':
    main()