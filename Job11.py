import pandas as pd
from collections import defaultdict

def read_excel_dependencies(file_path):
    df = pd.read_excel(file_path, usecols=[0, 1], engine='openpyxl')
    df.columns = ['Job', 'PostDependent']

    # Build complete dependency graph (job -> list of dependents)
    dependency_map = defaultdict(list)
    for _, row in df.iterrows():
        job = str(row['Job']).strip()
        dependent = str(row['PostDependent']).strip()
        if job and dependent:
            dependency_map[job].append(dependent)

    return dependency_map

def get_all_chains(job, dependency_map, visited=None):
    if visited is None:
        visited = set()

    if job in visited:
        return [[job + " (Cycle Detected)"]]

    visited.add(job)

    # If no dependents, it's a terminal node
    if job not in dependency_map or not dependency_map[job]:
        return [[job]]

    chains = []
    for dependent in dependency_map[job]:
        sub_chains = get_all_chains(dependent, dependency_map, visited.copy())
        for chain in sub_chains:
            chains.append([job] + chain)

    return chains

def print_job_hierarchy(job_name, dependency_map):
    job_name = job_name.strip()
    print(f"\nHierarchy for job: {job_name}")
    chains = get_all_chains(job_name, dependency_map)

    if not chains:
        print("No post-dependents found.")
    else:
        for idx, chain in enumerate(chains, 1):
            print(f"{idx}. {' → '.join(chain)}")

# -------- Main Program --------

if __name__ == "__main__":
    input_excel_path = 'job_dependencies.xlsx'  # Replace with your file path

    dependency_map = read_excel_dependencies(input_excel_path)

    # Set of all known jobs from both columns
    all_jobs = set(dependency_map.keys()) | {j for sublist in dependency_map.values() for j in sublist}

    while True:
        job_input = input("\nEnter job name (or 'exit' to quit): ").strip()
        if job_input.lower() == 'exit':
            break
        if job_input == "":
            print("Please enter a valid job name.")
            continue
        if job_input not in all_jobs:
            print(f"Job '{job_input}' not found in the data.")
            continue

        print_job_hierarchy(job_input, dependency_map)
