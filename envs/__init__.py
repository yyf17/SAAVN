
import os
import git
import sys
repo = git.Repo(".", search_parent_directories=True)
if f"{repo.working_tree_dir}" not in sys.path:
    sys.path.append(f"{repo.working_tree_dir}")
    print("add")



from envs.environments import NavRLEnv                  
