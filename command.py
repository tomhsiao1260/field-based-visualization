import subprocess

tasks = [
    {"z": 6039, "y": 2551, "x": 4008, "labels": "2"},
    {"z": 6039, "y": 1783, "x": 4008, "labels": "1"},
    {"z": 5271, "y": 2815, "x": 4075, "labels": "2"},
    {"z": 5271, "y": 2047, "x": 4075, "labels": "1 2"},
    {"z": 4503, "y": 3154, "x": 4380, "labels": "2"},
    {"z": 4503, "y": 2386, "x": 4380, "labels": "1 2"},
    {"z": 3903, "y": 3780, "x": 4347, "labels": "2"},
    {"z": 3903, "y": 2963, "x": 4400, "labels": "1 2"},
   
]

commands_template = [
    "python potential_init.py --z {z} --y {y} --x {x} --labels {labels} --auto_conductor --plot",
    "python potential_generate.py --z {z} --y {y} --x {x} --labels {labels} --auto_conductor --plot",
    "python potential_transform.py --z {z} --y {y} --x {x}",
    "python potential_transform.py --z {z} --y {y} --x {x} --mask"
]

for task in tasks:
    print(f"Processing parameter set: {task}")
    for command_template in commands_template:
        command = command_template.format(**task)
        print(f"Executing: {command}")
        try:
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            print(f"Completed: {command}\nOutput: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Execution failed: {command}\nError message: {e.stderr}")
            break

print("All tasks have been executed.")
