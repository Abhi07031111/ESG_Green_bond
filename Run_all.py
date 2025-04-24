import subprocess

def run_script(script_name):
    try:
        print(f"Running {script_name}...")
        subprocess.run(['python', script_name], check=True)
        print(f"{script_name} finished successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        exit(1)

# Run both scripts
run_script('Green_wash.py')
run_script('lalita2.py')

print("All programs executed successfully.")
