import os
import json
import requests
import tarfile
import zipfile
import argparse

def download_file(url, dest_path):
    """Download file from url to dest_path"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def extract_file(file_path, dest_dir):
    """Extract .tgz/.tar.gz/.zip archives"""
    if file_path.endswith((".tgz", ".tar.gz", ".tar")):
        with tarfile.open(file_path, "r:*") as tar:
            tar.extractall(dest_dir)
    elif file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
    else:
        print(f"Skipping unsupported file format: {file_path}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Download and unpack class assets.")
    parser.add_argument("root_dir", help="Root directory for assets")
    parser.add_argument("mapping_file", help="Path to class_mapping.json")
    args = parser.parse_args()

    root_dir = args.root_dir
    mapping_file = args.mapping_file

    with open(mapping_file, "r") as f:
        mappings = json.load(f)

    os.makedirs(root_dir, exist_ok=True)

    for entry in mappings:
        class_dir = os.path.join(root_dir, entry["class_dir"])
        os.makedirs(class_dir, exist_ok=True)

        for link in entry.get("links", []):
            filename = os.path.basename(link)
            file_path = os.path.join(class_dir, filename)

            if not os.path.exists(file_path):
                print(f"Downloading {link} -> {file_path}")
                try:
                    download_file(link, file_path)
                except Exception as e:
                    print(f"Failed to download {link}: {e}")
                    continue
            else:
                print(f"File already exists: {file_path}")

            print(f"Extracting {file_path} -> {class_dir}")
            try:
                if extract_file(file_path, class_dir):
                    print(f"Deleting archive {file_path}")
                    os.remove(file_path)
            except Exception as e:
                print(f"Failed to extract {file_path}: {e}")

if __name__ == "__main__":
    main()
