import pandas as pd

def load_job_map(file_path: str):
    df = pd.read_excel(file_path, header=None)
    
    job_map = {}
    for i in range(df.shape[1]):
        job = df.iloc[0, i]
        post_job = df.iloc[1, i]
        if pd.notna(job) and pd.notna(post_job):
            job_map[job] = post_job
    return job_map

def build_full_chain(start_job, job_map):
    chain = [start_job]
    visited = set()
    
    current = start_job
    while current in job_map:
        if current in visited:
            print(f"Cycle detected at: {current}")
            break
        visited.add(current)
        next_job = job_map[current]
        chain.append(next_job)
        current = next_job
    
    return chain

def main():
    file_path = "your_excel_file.xlsx"  # <- Update this with your actual file path
    job_map = load_job_map(file_path)

    # Example: Build the chain starting from job1
    start_job = "job1"
    chain = build_full_chain(start_job, job_map)
    print(" → ".join(chain))

if __name__ == "__main__":
    main()
