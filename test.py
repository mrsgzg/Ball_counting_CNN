import os
import json
import re
import argparse
from pathlib import Path

def validate_data_structure(root_dir):
    """
    Validate that the JSON count matches the folder names in a nested structure
    
    Args:
        root_dir (str): Path to the root directory containing numbered folders
        
    Returns:
        dict: Summary of validation results
    """
    root_path = Path(root_dir)
    
    # Results tracking
    results = {
        "total_folders": 0,
        "total_experiments": 0,
        "valid_experiments": 0,
        "invalid_experiments": 0,
        "missing_metadata": 0,
        "issues": []
    }
    
    # Pattern to extract number from folder name (for counting folders)
    folder_number_pattern = re.compile(r'^(\d+)$')
    
    # Go through all immediate subdirectories in the root directory (numbered folders)
    for count_folder in root_path.iterdir():
        if not count_folder.is_dir():
            continue
            
        folder_match = folder_number_pattern.match(count_folder.name)
        if not folder_match:
            print(f"Skipping non-numbered folder: {count_folder.name}")
            continue
            
        folder_count = int(folder_match.group(1))
        print(f"\nChecking count folder: {count_folder.name} (count = {folder_count})")
        results["total_folders"] += 1
        
        # Check if there are experiment folders inside the count folder
        experiment_found = False
        
        # Go through all experiment folders inside the count folder
        for experiment_folder in count_folder.iterdir():
            if not experiment_folder.is_dir():
                continue
                
            experiment_found = True
            results["total_experiments"] += 1
            print(f"  Examining experiment: {experiment_folder.name}")
            
            # Look for metadata.json in the experiment folder
            metadata_path = experiment_folder / "metadata.json"
            
            if not metadata_path.exists():
                print(f"    ❌ metadata.json not found in {experiment_folder.name}")
                results["missing_metadata"] += 1
                results["issues"].append(f"Folder {count_folder.name}/{experiment_folder.name}: metadata.json not found")
                continue
                
            # Read the metadata.json file
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Check if there's a count field in the metadata
                if "count" not in metadata:
                    # Try to find count in alternative fields
                    count_found = False
                    for field in ["ball_count", "counted_balls", "total_count"]:
                        if field in metadata:
                            json_count = metadata[field]
                            count_found = True
                            break
                            
                    # If we still don't have a count, check for nested structures
                    if not count_found and isinstance(metadata, dict):
                        # Try looking in nested fields
                        for key, value in metadata.items():
                            if isinstance(value, dict) and "count" in value:
                                json_count = value["count"]
                                count_found = True
                                break
                                
                    if not count_found:
                        # Check for the last frame in the joint_data directory
                        joint_data_dir = experiment_folder / "joint_data"
                        if joint_data_dir.exists() and joint_data_dir.is_dir():
                            json_files = list(joint_data_dir.glob("*.json"))
                            if json_files:
                                # Sort by numeric part of filename
                                json_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x.name))) if any(c.isdigit() for c in x.name) else 0)
                                latest_json = json_files[-1]
                                
                                try:
                                    with open(latest_json, 'r') as jf:
                                        joint_data = json.load(jf)
                                        
                                    # Handle both array and dict formats
                                    if isinstance(joint_data, list) and joint_data:
                                        last_frame = joint_data[-1]
                                        if "count" in last_frame:
                                            json_count = last_frame["count"]
                                            count_found = True
                                    elif isinstance(joint_data, dict) and "count" in joint_data:
                                        json_count = joint_data["count"]
                                        count_found = True
                                except Exception as e:
                                    print(f"    ⚠️ Error reading joint data: {str(e)}")
                                    
                        if not count_found:
                            print(f"    ❌ No 'count' field found in metadata or joint data")
                            results["issues"].append(f"Folder {count_folder.name}/{experiment_folder.name}: No count field found")
                            results["invalid_experiments"] += 1
                            continue
                else:
                    json_count = metadata["count"]
                    
                # Compare count with folder name
                if json_count == folder_count:
                    print(f"    ✅ Count matches: folder={folder_count}, json={json_count}")
                    results["valid_experiments"] += 1
                else:
                    print(f"    ❌ Count mismatch: folder={folder_count}, json={json_count}")
                    results["issues"].append(f"Folder {count_folder.name}/{experiment_folder.name}: Count mismatch, folder={folder_count}, json={json_count}")
                    results["invalid_experiments"] += 1
                    
            except Exception as e:
                print(f"    ❌ Error processing metadata.json: {str(e)}")
                results["issues"].append(f"Folder {count_folder.name}/{experiment_folder.name}: Error: {str(e)}")
                results["invalid_experiments"] += 1
        
        if not experiment_found:
            print(f"  ⚠️ No experiment folders found in {count_folder.name}")
    
    return results

def print_summary(results):
    """Print a summary of the validation results"""
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total count folders examined: {results['total_folders']}")
    print(f"Total experiment folders examined: {results['total_experiments']}")
    print(f"Valid experiments (count matches): {results['valid_experiments']}")
    print(f"Invalid experiments (count mismatch or error): {results['invalid_experiments']}")
    print(f"Experiments with missing metadata: {results['missing_metadata']}")
    
    if results["issues"]:
        print("\nIssues found:")
        for issue in results["issues"]:
            print(f"- {issue}")
    
    # Calculate percentage of valid experiments
    if results["total_experiments"] > 0:
        valid_percent = (results["valid_experiments"] / results["total_experiments"]) * 100
        print(f"\nOverall data validity: {valid_percent:.1f}%")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Validate that JSON count values match folder names')
    parser.add_argument('data_dir', help='Path to the root directory containing numbered folders')
    parser.add_argument('--save-report', '-s', action='store_true', 
                       help='Save the validation report to a file')
    
    args = parser.parse_args()
    
    print(f"Validating data structure in: {args.data_dir}")
    results = validate_data_structure(args.data_dir)
    print_summary(results)
    
    # Save report if requested
    if args.save_report:
        report_path = os.path.join(args.data_dir, "validation_report.txt")
        try:
            with open(report_path, 'w') as f:
                f.write("VALIDATION REPORT\n")
                f.write("="*60 + "\n")
                f.write(f"Total count folders examined: {results['total_folders']}\n")
                f.write(f"Total experiment folders examined: {results['total_experiments']}\n")
                f.write(f"Valid experiments (count matches): {results['valid_experiments']}\n")
                f.write(f"Invalid experiments (count mismatch or error): {results['invalid_experiments']}\n")
                f.write(f"Experiments with missing metadata: {results['missing_metadata']}\n\n")
                
                if results["issues"]:
                    f.write("Issues found:\n")
                    for issue in results["issues"]:
                        f.write(f"- {issue}\n")
                
                if results["total_experiments"] > 0:
                    valid_percent = (results["valid_experiments"] / results["total_experiments"]) * 100
                    f.write(f"\nOverall data validity: {valid_percent:.1f}%\n")
                
            print(f"\nReport saved to: {report_path}")
        except Exception as e:
            print(f"Error saving report: {str(e)}")

if __name__ == "__main__":
    main()