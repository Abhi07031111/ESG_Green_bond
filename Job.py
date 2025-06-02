import pandas as pd

def load_job_dependencies(file_path: str):
    # Load only first two rows (header=None for raw rows)
    df = pd.read_excel(file_path, header=None)

    # Row 0 = job names, Row 1 = their post-dependent jobs
    job_to_post = dict(zip(df.iloc[0], df.iloc[1]))
    return job_to_post

def build_dependency_chain(start_job, job_map):
    chain = [start_job]
    current = start_job
    visited = set()

    while current in job_map and pd.notna(job_map[current]) and current not in visited:
        visited.add(current)
        next_job = job_map[current]
        chain.append(next_job)
        current = next_job

    return chain

def main():
    file_path = 'your_excel_file.xlsx'  # Change this to your actual file path
    job_dependencies = load_job_dependencies(file_path)

    # Example: get chain for job1
    start_job = 'job1'
    chain = build_dependency_chain(start_job, job_dependencies)
    print(" → ".join(chain))

if __name__ == "__main__":
    main()
