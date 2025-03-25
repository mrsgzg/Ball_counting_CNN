import os
import json
import re

def check_sample_json_data(root_folder):
    """
    Checks each sample folder in the root folder, finds the JSON file with the highest 
    number in each sample's joint_data folder, and reports the content of 'counted_balls'.
    
    Args:
        root_folder (str): Path to the root folder containing sample folders
    
    Returns:
        list: Results of the check for each sample
    """
    results = []
    
    # Get the root folder number (if it's numeric)
    root_name = os.path.basename(os.path.normpath(root_folder))
    try:
        root_number = int(root_name)
    except ValueError:
        root_number = None
    
    # Get all sample folders in the root folder
    try:
        sample_folders = []
        for item in os.listdir(root_folder):
            item_path = os.path.join(root_folder, item)
            if os.path.isdir(item_path):
                # Check if this folder has a joint_data subfolder
                if os.path.exists(os.path.join(item_path, 'joint_data')):
                    sample_folders.append(item_path)
        
        if not sample_folders:
            return [{"error": f"No sample folders with joint_data found in {root_folder}"}]
        
        # Process each sample folder
        for sample_folder in sample_folders:
            sample_name = os.path.basename(sample_folder)
            joint_data_path = os.path.join(sample_folder, 'joint_data')
            
            # Get all JSON files in the joint_data folder
            json_files = []
            for file in os.listdir(joint_data_path):
                if file.endswith('.json') and not file == 'metadata.json':
                    json_files.append(file)
            
            if not json_files:
                results.append({
                    "sample": sample_name,
                    "error": "No JSON files found in joint_data folder"
                })
                continue
            
            # Extract file numbers using regex and find the highest one
            file_numbers = []
            for file in json_files:
                match = re.search(r'data_(\d+)\.json', file)
                if match:
                    file_numbers.append((int(match.group(1)), file))
            
            if not file_numbers:
                results.append({
                    "sample": sample_name,
                    "error": "Could not parse file numbers"
                })
                continue
            
            # Sort by number and get the highest one
            file_numbers.sort(reverse=True)
            highest_file = file_numbers[0][1]
            highest_number = file_numbers[0][0]
            
            # Read the file with the highest number
            try:
                with open(os.path.join(joint_data_path, highest_file), 'r') as f:
                    data = json.load(f)
                
                # Check if the data is an array
                if isinstance(data, list):
                    # Find the last frame in the array
                    if data:
                        last_frame = data[-1]
                        
                        if "counted_balls" in last_frame:
                            counted_balls = last_frame["counted_balls"]
                            
                            # Check if the counted_balls matches the expectation based on folder name
                            expected_balls = []
                            if root_number is not None:
                                if root_number >= 2:
                                    expected_balls = list(range(root_number - 1))
                            
                            is_valid = counted_balls == expected_balls
                            
                            results.append({
                                "sample": sample_name,
                                "highest_file": highest_file,
                                "highest_number": highest_number,
                                "last_frame": last_frame.get("frame", "unknown"),
                                "counted_balls": counted_balls,
                                "root_folder": root_name,
                                "expected_balls": expected_balls,
                                "is_valid": is_valid
                            })
                        else:
                            results.append({
                                "sample": sample_name,
                                "highest_file": highest_file,
                                "highest_number": highest_number,
                                "error": "No 'counted_balls' field found in the last frame of the JSON file"
                            })
                    else:
                        results.append({
                            "sample": sample_name,
                            "highest_file": highest_file,
                            "highest_number": highest_number,
                            "error": "JSON file is an empty array"
                        })
                else:
                    # If data is not an array, try to access counted_balls directly
                    if "counted_balls" in data:
                        counted_balls = data["counted_balls"]
                        
                        # Check if the counted_balls matches the expectation based on folder name
                        expected_balls = []
                        if root_number is not None:
                            if root_number >= 2:
                                expected_balls = list(range(root_number - 1))
                        
                        is_valid = counted_balls == expected_balls
                        
                        results.append({
                            "sample": sample_name,
                            "highest_file": highest_file,
                            "highest_number": highest_number,
                            "counted_balls": counted_balls,
                            "root_folder": root_name,
                            "expected_balls": expected_balls,
                            "is_valid": is_valid
                        })
                    else:
                        results.append({
                            "sample": sample_name,
                            "highest_file": highest_file,
                            "highest_number": highest_number,
                            "error": "No 'counted_balls' field found in the JSON file"
                        })
            
            except Exception as e:
                results.append({
                    "sample": sample_name,
                    "highest_file": highest_file,
                    "highest_number": highest_number,
                    "error": str(e)
                })
        
        return results
    
    except Exception as e:
        return [{"error": f"Error processing root folder: {str(e)}"}]

def print_results(results):
    """Print the results in a readable format"""
    print(f"\n{'=' * 60}")
    print(f"JSON DATA CHECK RESULTS")
    print(f"{'=' * 60}")
    
    # Count valid and invalid samples
    valid_count = 0
    invalid_count = 0
    error_count = 0
    
    for result in results:
        if "error" in result:
            error_count += 1
        elif result.get("is_valid", False):
            valid_count += 1
        else:
            invalid_count += 1
    
    print(f"Total samples: {len(results)}")
    print(f"Valid samples: {valid_count}")
    print(f"Invalid samples: {invalid_count}")
    print(f"Samples with errors: {error_count}")
    print(f"{'=' * 60}")
    
    for idx, result in enumerate(results):
        print(f"\nSample {idx+1}: {result.get('sample', 'Unknown')}")
        print(f"{'-' * 60}")
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue
        
        print(f"Highest file checked: {result['highest_file']}")
        if "last_frame" in result:
            print(f"Last frame: {result['last_frame']}")
        print(f"Counted balls: {result['counted_balls']}")
        
        if "is_valid" in result:
            if result["is_valid"]:
                print(f"✅ Valid: The counted_balls array matches the expected value")
            else:
                print(f"❌ Invalid: The counted_balls array does not match the expected value")
                print(f"   Expected: {result['expected_balls']}")
                print(f"   Found: {result['counted_balls']}")
    
    print(f"\n{'=' * 60}")
    print(f"Summary: {valid_count} valid, {invalid_count} invalid, {error_count} errors")
    print(f"{'=' * 60}\n")

# Example usage
if __name__ == "__main__":
    # Use the specified path to the data folder
    # You can use the absolute path if needed
    import os
    # Uncomment and modify the line below if you need absolute path
    # root_folder = os.path.join(os.path.expanduser("~"), "embody_data/raw/1")
    root_folder = "embody_data/raw/8"
    results = check_sample_json_data(root_folder)
    print_results(results)