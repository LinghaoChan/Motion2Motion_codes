import os

def get_rest_pose(root_folder, output_folder):
    """
    Generate rest pose BVH files by converting the first BVH file found in each character folder
    to a zero-motion state (all joint rotations and translations set to 0).
    
    Args:
        root_folder (str): Path to the directory containing character folders with BVH files
        output_folder (str): Path to the directory where rest pose BVH files will be saved
    """
    def modify_bvh_to_zero(input_file, output_file):
        """
        Modify a BVH file by setting all motion data to zero while preserving the skeleton hierarchy.
        
        Args:
            input_file (str): Path to the input BVH file
            output_file (str): Path to the output BVH file with zeroed motion data
        """
        # Read all lines from the input BVH file
        with open(input_file, 'r') as f:
            lines = f.readlines()

        # Find the start of the MOTION section
        motion_start_index = None
        for i, line in enumerate(lines):
            if line.strip() == 'MOTION':
                motion_start_index = i
                break

        if motion_start_index is None:
            raise ValueError(f"BVH file {input_file} does not contain a MOTION section.")

        # Parse the Frames line to get the number of frames
        frames_line = lines[motion_start_index + 1].strip()
        frames_info = frames_line.split()
        if len(frames_info) < 2:
            raise ValueError(f"Invalid BVH format in {input_file} (Frames line).")

        # Extract the number of frames
        try:
            num_frames = int(frames_info[1])
        except ValueError:
            raise ValueError(f"Invalid number of frames in {input_file}: {frames_info[1]}")

        # Get the motion data starting from the 4th line after MOTION
        motion_data = lines[motion_start_index + 3:]
        if len(motion_data) < num_frames:
            raise ValueError(f"{input_file} does not have enough motion data lines!")

        # Analyze the first frame to determine the number of channels
        first_frame_line = motion_data[0].strip()
        first_frame_values = first_frame_line.split()
        frame_length = len(first_frame_values)
        
        # Create new motion data with all values set to 0
        new_motion_data = [" ".join(["0"] * frame_length) + "\n" for _ in range(num_frames)]

        # Combine the skeleton hierarchy with the zeroed motion data
        new_lines = lines[:motion_start_index + 3] + new_motion_data
        with open(output_file, 'w') as f:
            f.writelines(new_lines)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each character folder in the root directory
    for animal_name in os.listdir(root_folder):
        animal_path = os.path.join(root_folder, animal_name)
        
        # Skip if not a directory
        if not os.path.isdir(animal_path):
            continue

        # Find all BVH files in the character folder
        bvh_files = [f for f in os.listdir(animal_path) if f.endswith('.bvh')]
        if not bvh_files:
            print(f"Warning: No BVH files found in {animal_path}")
            continue

        # Use the first BVH file found as the source for the rest pose
        input_bvh = os.path.join(animal_path, bvh_files[0])
        output_bvh = os.path.join(output_folder, f"{animal_name}_rest.bvh")

        # Generate the rest pose BVH file
        try:
            modify_bvh_to_zero(input_bvh, output_bvh)
            print(f"Generated rest pose for {animal_name}: {output_bvh}")
        except Exception as e:
            print(f"Error processing {input_bvh}: {e}")

# Example usage: Generate rest poses for human characters
if __name__ == "__main__":
    get_rest_pose("data/human/", "data/human_rest_poses")
