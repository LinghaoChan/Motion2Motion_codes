import os
import json


def get_tree(input_folder, output_folder_txt, output_folder_json):
    """
    Parse BVH files to extract skeleton hierarchy and save as both text tree and JSON format.

    Args:
        input_folder (str): Directory containing BVH files to process
        output_folder_txt (str): Directory to save text-based tree representations
        output_folder_json (str): Directory to save JSON skeleton data
    """

    def parse_bvh(file_path):
        """
        Parse a BVH file to extract the skeleton hierarchy with joint information.

        Args:
            file_path (str): Path to the BVH file to parse

        Returns:
            dict: Skeleton tree structure with joints, offsets, and motion channel info
        """
        with open(file_path, "r") as f:
            lines = f.readlines()

        skeleton = {}
        stack = []  # Stack to track parent-child relationships
        inside_end_site = False  # Flag to track if we're inside an End Site block
        motion_index = 0  # Track current motion channel index

        for line in lines:
            words = line.strip().split()
            if not words:
                continue

            # Parse ROOT or JOINT declarations
            if words[0] in ["ROOT", "JOINT"]:
                joint_name = words[1]
                joint_data = {
                    "name": joint_name,
                    "offset": None,
                    "motion_start": None,  # Starting index in motion data
                    "motion_end": None,  # Ending index in motion data
                    "children": [],
                }
                # Add to parent's children or set as root
                if stack:
                    stack[-1]["children"].append(joint_data)
                else:
                    skeleton = joint_data
                stack.append(joint_data)

            # Parse joint offset (3D position relative to parent)
            elif words[0] == "OFFSET":
                if stack:
                    stack[-1]["offset"] = [
                        float(words[1]),
                        float(words[2]),
                        float(words[3]),
                    ]

            # Parse motion channels (rotation/translation channels)
            elif words[0] == "CHANNELS":
                num_channels = int(words[1])
                if stack:
                    stack[-1]["motion_start"] = motion_index
                    stack[-1]["motion_end"] = motion_index + num_channels - 1
                    motion_index += num_channels

            # Handle End Site (terminal joint with no children)
            elif words[0] == "End":
                inside_end_site = True

            # Handle closing braces
            elif words[0] == "}":
                if inside_end_site:
                    inside_end_site = False
                elif stack:
                    stack.pop()  # Move back up the hierarchy

        return skeleton

    def print_skeleton_tree(skeleton, prefix="", is_last=True):
        """
        Generate a visual tree representation of the skeleton hierarchy.

        Args:
            skeleton (dict): Skeleton data structure
            prefix (str): Prefix string for tree indentation
            is_last (bool): Whether this is the last child at current level

        Returns:
            str: Text representation of the skeleton tree
        """
        # Choose appropriate tree connector
        connector = "└── " if is_last else "├── "
        output = f"{prefix}{connector}{skeleton['name']} (start: {skeleton['motion_start']}, end: {skeleton['motion_end']})\n"

        children = skeleton.get("children", [])

        # Recursively process children with appropriate indentation
        for i, child in enumerate(children):
            new_prefix = prefix + ("    " if is_last else "│   ")
            output += print_skeleton_tree(child, new_prefix, i == len(children) - 1)

        return output

    def save_json(data, file_path):
        """
        Save skeleton data to JSON file with proper formatting.

        Args:
            data (dict): Skeleton data to save
            file_path (str): Path to output JSON file
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    # Create output directories if they don't exist
    if not os.path.exists(output_folder_txt):
        os.makedirs(output_folder_txt)
    if not os.path.exists(output_folder_json):
        os.makedirs(output_folder_json)

    # Process all BVH files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".bvh"):
            input_path = os.path.join(input_folder, file_name)
            output_txt_path = os.path.join(
                output_folder_txt, file_name.replace(".bvh", ".txt")
            )
            output_json_path = os.path.join(
                output_folder_json, file_name.replace(".bvh", ".json")
            )

            print(f"Processing: {file_name} -> {output_txt_path} / {output_json_path}")

            try:
                # Parse the BVH file to get skeleton structure
                skeleton_tree = parse_bvh(input_path)

                # Generate and save text tree representation
                tree_output = print_skeleton_tree(skeleton_tree)
                with open(output_txt_path, "w", encoding="utf-8") as f:
                    f.write(tree_output)

                # Save JSON representation of skeleton data
                save_json(skeleton_tree, output_json_path)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")


# Example usage: Process rest pose BVH files to extract skeleton hierarchies
if __name__ == "__main__":
    get_tree("./data/rest_poses/", "./data/skeleton/", "./data/skeleton/")
