import pandas as pd

def load_job_map(file_path: str):
    # Load the Excel file assuming two columns: [Job, PostDependent]
    df = pd.read_excel(file_path)

    job_map = {}
    for _, row in df.iterrows():
        job = str(row[0]).strip()
        post_job = str(row[1]).strip()
        if job and post_job and post_job.lower() != 'nan':
            job_map[job] = post_job
    return job_map

def build_dependency_chain(start_job, job_map):
    chain = []
    visited = set()
    current = start_job.strip()

    while current and current not in visited:
        chain.append(current)
        visited.add(current)
        current = job_map.get(current)

    return chain

def main():
    file_path = "your_excel_file.xlsx"  # <-- Update with your actual file path
    start_job = input("Enter the job name to get post-dependent chain: ").strip()

    job_map = load_job_map(file_path)

    # Allow search even if start_job is not in column 1 initially
    found = any(start_job in pair for pair in job_map.items())
    if not found and start_job not in job_map:
        print(f"No post-dependency found for job '{start_job}'.")
        return

    chain = build_dependency_chain(start_job, job_map)
    print(" → ".join(chain))

if __name__ == "__main__":
    main()
