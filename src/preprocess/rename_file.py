import os
import sys
from urllib.parse import quote

def encode_and_rename_files(root_dir: str) -> None:
    """
    Walk through root_dir and rename any files whose names contain
    special characters like @, space, #, etc. to their URL-encoded equivalents.
    """
    # Characters that need encoding
    special_chars = ['@', ' ', '#', '%', '&', '+', '/', ':', '?', '=', '[', ']', '(', ')']
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if filename contains any special characters
            if not any(char in filename for char in special_chars):
                continue

            # Encode the filename (safe='.' to keep dots for extensions)
            encoded = quote(filename, safe='.')  # e.g. "foo bar@0.jpg" -> "foo%20bar%400.jpg"

            if encoded == filename:
                # Nothing changed after encoding, skip
                continue

            src = os.path.join(dirpath, filename)
            dst = os.path.join(dirpath, encoded)

            # Avoid overwriting existing files
            if os.path.exists(dst):
                print(f"[SKIP] Target already exists:\n       {src} -> {dst}")
                continue

            print(f"[RENAME] {src} -> {dst}")
            os.rename(src, dst)


def main():
    if len(sys.argv) != 2:
        print("Usage: python rename_url_encoded_files.py <root_directory>")
        sys.exit(1)

    root_dir = sys.argv[1]
    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a directory or does not exist.")
        sys.exit(1)

    encode_and_rename_files(root_dir)
    print("Done.")


if __name__ == "__main__":
    main()
