import os
import re
import shutil
import sys

def extract_paths_from_file(file_path):
    """
    Extract sample paths from the invalid samples text file.
    
    Args:
        file_path (str): Path to the text file containing invalid sample paths
    
    Returns:
        list: List of paths to be deleted
    """
    paths_to_delete = []
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Use regex to find all paths in the file
        path_pattern = r'Path: ([\w./]+)'
        paths = re.findall(path_pattern, content)
        
        for path in paths:
            if os.path.exists(path):
                paths_to_delete.append(path)
            else:
                print(f"Warning: Path does not exist: {path}")
        
        return paths_to_delete
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

def delete_folders(paths, dry_run=True):
    """
    Delete the folders at the specified paths.
    
    Args:
        paths (list): List of folder paths to delete
        dry_run (bool): If True, only print what would be deleted without actually deleting
    
    Returns:
        int: Number of folders successfully deleted
    """
    deleted_count = 0
    
    for path in paths:
        try:
            if dry_run:
                print(f"Would delete: {path}")
                deleted_count += 1
            else:
                print(f"Deleting: {path}")
                shutil.rmtree(path)
                deleted_count += 1
                print(f"✓ Deleted successfully")
        except Exception as e:
            print(f"✗ Error deleting {path}: {e}")
    
    return deleted_count

def main():
    # Check if file path is provided
    if len(sys.argv) < 2:
        print("Usage: python delete_invalid_samples.py <file_path> [--force]")
        print("  <file_path>: Path to the text file containing invalid sample paths")
        print("  --force: Add this flag to actually delete the folders (otherwise runs in dry-run mode)")
        return
    
    file_path = sys.argv[1]
    force_delete = "--force" in sys.argv
    
    # Extract paths from the file
    paths = extract_paths_from_file(file_path)
    
    if not paths:
        print("No valid paths found in the file.")
        return
    
    print(f"Found {len(paths)} paths to delete:")
    for i, path in enumerate(paths):
        print(f"{i+1}. {path}")
    
    # Confirm deletion
    if not force_delete:
        print("\nRunning in DRY-RUN mode. No folders will be deleted.")
        print("To actually delete the folders, run with the --force flag.")
        print("\nSimulating deletion:")
    else:
        confirm = input(f"\nAre you sure you want to delete these {len(paths)} folders? This cannot be undone! (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return
        print("\nDeleting folders:")
    
    # Delete the folders
    deleted = delete_folders(paths, dry_run=not force_delete)
    
    # Print summary
    if force_delete:
        print(f"\nDeleted {deleted} folders out of {len(paths)} paths.")
    else:
        print(f"\nWould delete {deleted} folders out of {len(paths)} paths.")
        print("Run with --force to actually delete the folders.")

if __name__ == "__main__":
    main()