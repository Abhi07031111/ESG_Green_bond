import pandas as pd
from collections import defaultdict

def read_excel_dependencies(file_path):
    df = pd.read_excel(file_path, usecols=[0, 1], engine='openpyxl')
    df.columns = ['Job', 'PostDependent']

    dependency_map = defaultdict(set)
    for _, row in df.iterrows():
        job = str(row['Job']).strip()
        dependent = str(row['PostDependent']).strip()
        if job and dependent:
            dependency_map[job].add(dependent)

    return {k: list(v) for k, v in dependency_map.items()}

def get_all_chains(job, dependency_map, visited=None):
    if visited is None:
        visited = set()

    if job in visited:
        return [[job + " (Cycle Detected)"]]

    visited.add(job)

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
        return

    unique_chains = []
    seen = set()
    for chain in chains:
        chain_tuple = tuple(chain)
        if chain_tuple not in seen:
            seen.add(chain_tuple)
            unique_chains.append(chain)

    for idx, chain in enumerate(unique_chains, 1):
        print(f"{idx}. {' → '.join(chain)}")

def remove_job_from_dependency_map(dependency_map, job_to_remove):
    new_map = {}
    for job, dependents in dependency_map.items():
        if job == job_to_remove:
            continue
        filtered_dependents = [d for d in dependents if d != job_to_remove]
        new_map[job] = filtered_dependents
    return new_map

# -------- Main Program --------

if __name__ == "__main__":
    input_excel_path = 'job_dependencies.xlsx'  # Replace with your file path

    dependency_map = read_excel_dependencies(input_excel_path)

    while True:
        all_jobs = set(dependency_map.keys()) | {j for sublist in dependency_map.values() for j in sublist}
        print("\nAvailable jobs:", ", ".join(sorted(all_jobs)))
        
        action = input("\nChoose action: 'view', 'remove', or 'exit': ").strip().lower()

        if action == 'exit':
            break
        elif action == 'remove':
            job_to_remove = input("Enter job name to remove from hierarchy: ").strip()
            if job_to_remove not in all_jobs:
                print(f"Job '{job_to_remove}' not found.")
                continue
            dependency_map = remove_job_from_dependency_map(dependency_map, job_to_remove)
            print(f"Removed job '{job_to_remove}' from dependency tree.")
        elif action == 'view':
            job_input = input("Enter job name to view hierarchy: ").strip()
            if job_input not in all_jobs:
                print(f"Job '{job_input}' not found.")
                continue
            print_job_hierarchy(job_input, dependency_map)
        else:
            print("Invalid action. Please enter 'view', 'remove', or 'exit'.")
