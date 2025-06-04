import pandas as pd
from collections import defaultdict

def read_excel_dependencies(file_path):
    df = pd.read_excel(file_path, usecols=[0, 1], engine='openpyxl')
    df.columns = ['Job', 'PostDependent']
    
    dependency_map = defaultdict(list)
    for _, row in df.iterrows():
        job = row['Job']
        dependent = row['PostDependent']
        if pd.notna(job) and pd.notna(dependent):
            dependency_map[job].append(dependent)
    
    return dependency_map

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
    print(f"\nHierarchy for job: {job_name}")
    chains = get_all_chains(job_name, dependency_map)
    if not chains:
        print("No post-dependents found.")
    else:
        for idx, chain in enumerate(chains, 1):
            print(f"{idx}. {' → '.join(chain)}")

# -------- Main Program --------

if __name__ == "__main__":
    input_excel_path = 'job_dependencies.xlsx'  # Your Excel file

    dependency_map = read_excel_dependencies(input_excel_path)

    while True:
        job_input = input("\nEnter job name (or 'exit' to quit): ").strip()
        if job_input.lower() == 'exit':
            break
        if job_input == "":
            print("Please enter a valid job name.")
            continue
        print_job_hierarchy(job_input, dependency_map)
