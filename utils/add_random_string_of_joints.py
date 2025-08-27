import re
import random
import string
import os


def generate_random_suffix(length=3):
    """Generate a random string of specified length composed of letters and digits"""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def modify_bvh_joints(input_bvh_path, output_bvh_path):
    """Read BVH file and add random suffix to each joint name"""
    with open(input_bvh_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    modified_lines = []
    joint_pattern = re.compile(r"^(\s*)(JOINT|ROOT)\s+([A-Za-z0-9_]+)(\s*)$")

    for line in lines:
        match = joint_pattern.match(line)
        if match:
            spaces, joint_type, joint_name, trailing_spaces = match.groups()
            new_joint_name = f"{joint_name}__{generate_random_suffix()}"
            modified_lines.append(
                f"{spaces}{joint_type} {new_joint_name}{trailing_spaces}"
            )
        else:
            modified_lines.append(line)

    with open(output_bvh_path, "w", encoding="utf-8") as file:
        file.writelines(modified_lines)

    print(f"Modified BVH file saved to: {output_bvh_path}")


def process_all_bvh_files(directory):
    """Traverse the specified directory and process all BVH files"""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".bvh"):
                input_bvh_path = os.path.join(root, file)
                output_bvh_path = os.path.join(root, file)
                modify_bvh_joints(input_bvh_path, output_bvh_path)


if __name__ == "__main__":
    # Example usage
    process_all_bvh_files("your_path")
