# Run-Except Loop

## High-Level Objective
execution_command --> repo diff
Does not change the script. Uses the script to generate a diff on the repo so that no exception are thrown by the repo.
This script is for minor errors eg pytorch tensor dims are off, not to do fundamental refactors of a repo.

## Mid-Level Objectives
Run the script in a loop.
Catch exceptions.
Get files involved in the exception.
Ask LLM to fix the code.
Loop until no exceptions are thrown.

## Implementation Notes
For LLM calls, use aider.Coder with claude-3-7-sonnet. Use the `fnames` and `read_only_fnames` kwargs to pass the relevant files to the LLM.

## Context
### Beginning Context
adw/run_except_loop.py

### Ending Context
adw/run_except_loop.py

## Low-Level Tasks
1. Create `main` as stub.
```aider
CREATE def main():
    stub.
```

2. Build `create_branch` to create a new git branch.
```aider 
CREATE def create_branch(script_name: Union[str, Path]) -> str:
    Create a new git branch named "automated-exceptions-fix__<script_name_camel_case>" and switch to it, catching any errors in the process.
```

3. Build `run_script` to run command, and capture stdout/stderr.
```aider
CREATE def run_script(command) -> str:
    Use subprocess to run command;
    If exception, return error message + stacktrace with err_flg=True;
    Else return stdout with err_flg=False.
```

4. Build `get_editable_files` to get all files in repo that were part of stacktrace.
```aider
CREATE def get_editable_files(stacktrace: str, repo_path: Path) -> List[Path]:
    Parse the stacktrace to get any paths that reference files in the repo.
    Return the list of paths.
```

5. Build `get_readonly_files` that gets file that contains the lowest-level function in stacktrace above the script (ie contained in the repo), and all files that depend on that function.
```aider
CREATE def get_readonly_files(stacktrace: str, repo_path: Path) -> List[Path]:
    CREATE problem_root: Path = the last repo-contained path in the stacktrace.
    Find all files in the repo that depend on `problem_root`.
    Return the list of paths.
```

6. Update main with create_branch, run_script, get_editable_files, get_readonly_files, and a prompt containing the exception and stacktrace, indicating to the LLM it should fix the exception.
```aider
UPDATE main():
    Take two cli args: run-command (the command to run the script), target-repo-path (the path to the repo we are fixing).
    create_branch(script_name)
    CREATE loop_flg: bool = True
    while loop_flg:
        result, loop_flg = run_script(command)
        if loop_flg:
            CREATE editable_files, readonly_files = get_editable_files(stacktrace), get_readonly_files(stacktrace) # Make sure these lists are disjoint, presence in editable_files takes precedence over readonly_files.
            CREATE coder: aider.Coder with model="claude-3-7-sonnet-latest", fnames=editable_files, read_only_fnames=readonly_files, auto_commit=False, suggest_shell_commands=False, io=InputOutput(yes=True)
            CREATE prompt: str = Something like: f"The following exception occurred: {exception}. The stacktrace is: {stacktrace}. Fix the code so it runs without exceptions. Do not change the fundamental logic of the repo, only make minor changes." (rewrite this as you see fit to get the best results out of the LLM)
            coder.run(prompt)

    CREATE diff: str = the diff from the start commit of the branch to the current working commit.
    CREATE summary: str = An LLM summarizes the changes made in the branch (by passing the diff to an LLM).
    print(summary)
```