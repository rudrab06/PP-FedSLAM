import sys
import os

# Define the default path based on run_vo_experiment.py configuration
BASE_DATA_ROOT = 'tum_data/rgbd_dataset_freiburg1_xyz'
OUTPUT_FILENAME = 'associations.txt'
MAX_DIFFERENCE = 0.02 # Maximum difference between timestamps for a match

def read_file_list(filename):
    """
    Reads a file containing timestamped data, separated by spaces.
    Returns a list of (timestamp, line_content_after_timestamp) tuples.
    """
    file_list = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split()
                    try:
                        timestamp = float(parts[0])
                        # Content is the part after the timestamp (filename or pose data)
                        content = ' '.join(parts[1:])
                        file_list.append((timestamp, content))
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"Error: Input file not found: {filename}")
        sys.exit(1)
    return file_list

def associate(first_list, second_list, max_difference):
    """
    Associates two lists of timestamped data greedily.
    """
    associations = []
    
    # Simple, efficient association based on nearest neighbor in time
    for i, (t1, v1) in enumerate(first_list):
        # Find the best match in the second list
        min_diff = float('inf')
        best_match_index = -1
        
        for j, (t2, v2) in enumerate(second_list):
            diff = abs(t1 - t2)
            
            if diff < min_diff:
                min_diff = diff
                best_match_index = j
            
        if min_diff < max_difference:
            associations.append((i, best_match_index))

    return associations

def generate_associations(data_dir):
    """
    Main function to generate the associations file.
    """
    rgb_file = os.path.join(data_dir, 'rgb.txt')
    depth_file = os.path.join(data_dir, 'depth.txt')

    print(f"--- Loading data from: {data_dir} ---")

    # 1. Read file lists
    rgb_list = read_file_list(rgb_file)
    depth_list = read_file_list(depth_file)
    
    if not rgb_list or not depth_list:
        print("Error: RGB or Depth list is empty. Cannot associate.")
        return False

    # 2. Perform association
    associations = associate(rgb_list, depth_list, max_difference=MAX_DIFFERENCE)

    # 3. Write the output file
    output_path = os.path.join(data_dir, OUTPUT_FILENAME)
    with open(output_path, 'w') as f:
        for i, j in associations:
            ts_rgb, fn_rgb = rgb_list[i]
            ts_depth, fn_depth = depth_list[j]
            
            # TUM format: rgb_ts rgb_filename depth_ts depth_filename
            f.write(f"{ts_rgb:.6f} {fn_rgb} {ts_depth:.6f} {fn_depth}\n")

    print(f"Successfully created {OUTPUT_FILENAME} with {len(associations)} associations at {output_path}.")
    return True

if __name__ == '__main__':
    if not os.path.exists(BASE_DATA_ROOT):
        print(f"Error: Data directory not found at {BASE_DATA_ROOT}. Ensure your extraction was successful.")
        sys.exit(1)
        
    if generate_associations(BASE_DATA_ROOT):
        print("\nAssociation setup complete. You can now run the VO experiment.")
    else:
        sys.exit(1)