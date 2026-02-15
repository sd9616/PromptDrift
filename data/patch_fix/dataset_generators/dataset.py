import subprocess
import os
import re
import csv

projects = subprocess.run(
    ["defects4j", "pids"],
    capture_output=True,
    text=True,
    check=True
).stdout.strip().split("\n")

base_prompt_dataset = []
test_dataset = []
original_files_dataset = []

# projects = ["Cli"]

for project in projects:
    bugs = subprocess.run(
    ["defects4j", "bids", "-p", project],
    capture_output=True,
    text=True,
    check=True
    ).stdout.strip().split("\n")  

    # bugs = ["9"]  

    for bug in bugs:
        try:
            workdir = "/tmp/" + project + "_" + bug 

            result = subprocess.run(
                ["defects4j", "info", "-p", project, "-b", bug],
                capture_output=True,
                text=True,
                check=True
            )

            info_output = result.stdout

            match = re.search(r"List of modified sources:\n((?:\s*-\s.*\n)+)", info_output)
            if match:
                files_text = match.group(1)
                modified_sources = [line.strip()[2:] for line in files_text.strip().splitlines()]
            else:
                modified_sources = []

            if os.path.exists(workdir):
                subprocess.run(["rm", "-rf", workdir])

            subprocess.run(
                ["defects4j", "checkout", "-p", project, "-v", bug + "b", "-w", workdir],
                check=True
            )

            file_content = {}
            for class_path in modified_sources:
                java_file = os.path.join(workdir, "src", "main", "java", *class_path.split(".")) + ".java"
                if os.path.exists(java_file):
                    with open(java_file, "r") as f:
                        content = f.read()
                    file_content[f"{class_path}.java"] = f"{content}\n"
                else:
                    java_file = os.path.join(workdir, "src", "java", *class_path.split(".")) + ".java"
                    if os.path.exists(java_file):
                        with open(java_file, "r") as f:
                            content = f.read()
                        file_content[f"{class_path}.java"] = f"{content}\n"
                    else:
                        print(f"Warning: {java_file} not found!")


            bug_url_match = re.search(r"Bug report url:\n\s*(.*)", info_output)
            bug_url = bug_url_match.group(1) if bug_url_match else "No URL"

            tests = {}
            test_section_match = re.search(
                r"Root cause in triggering tests:\n(.*?)(?:\n-{3,}|\Z)",  # stop at line of dashes or end
                info_output,
                re.DOTALL
            )

            if test_section_match:
                section = test_section_match.group(1)

                # First split: by "- " to separate each test
                test_entries = [e.strip() for e in section.split("\n- ") if e.strip()]

                for entry in test_entries:
                    # Split entry on "-->" to separate test name and reason
                    parts = entry.split("-->", 1)
                    test_name = parts[0].strip()
                    reason = parts[1].strip() if len(parts) > 1 else "No reason provided"
                    tests[test_name] = reason


            instruction = """Given bug report URL and Original files, fix the bug by rewriting the files in orignal files

            STRICTLY follow this format for each file. Do NOT add any extra text, explanations, or quotes. Repeat the format for multiple files.

            ---FILE_START---
            FILENAME: <full_file_name>
            CONTENT:
            <full file content here>
            ---FILE_END---
            """

            prompt = f"""
            Bug report URL: {bug_url}

            Original files:
            {file_content}

            Instruction: {instruction}
            """

            if len(prompt) <= 5000:
                base_prompt_dataset.append((project, bug, info_output, bug_url, len(tests), len(modified_sources), prompt))
                for name_key, content_value in file_content.items():
                    original_files_dataset.append((project, bug, name_key, content_value))
                for test_name_key, test_reason_value in tests.items():
                    test_dataset.append((project, bug, test_name_key, test_reason_value))
        except:
            pass

with open("base_prompt_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["project", "bug_id", "bug_info", "bug_url", "num_of_failing_tests", "num_of_modified_sources", "prompt"])
    writer.writerows(base_prompt_dataset)

with open("original_files_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["project", "bug_id", "file_name", "file_content"])
    writer.writerows(original_files_dataset)

with open("test_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["project", "bug_id", "test_name", "reason"])
    writer.writerows(test_dataset)