import glob  # <-- added for upload_3d_results
import os
import time

from github import Github


class GitHubUploader:
    """
    Fixed to:
    - Upload script to scripts/freesurface/SCRIPT_FILENAME.py (or root if scripts_path not provided).
    - Upload images/log to results/freesurface/SCRIPT_FILENAME/.
    - Overwrite existing files.
    Repo: faircm2/Lb-Python
    """
    
    def __init__(self, debug_log, timeout, script_filename, script_full_path, scripts_path=None, plots_path=None, 
                 repo_name='faircm2/Lb-Python', images_subdir='FreesurfaceImages', 
                 log_file='lbm_debug.log', token_file='github-repo-token.txt'):
        self.debug_log = debug_log
        self.timeout = timeout
        self.script_filename = script_filename  # Without .py
        self.script_full_path = script_full_path
        self.repo_name = repo_name
        self.images_subdir = images_subdir
        self.log_file = log_file
        self.token_file = token_file
        self.scripts_path = scripts_path or ''  # Default to empty (repo root)
        
        # Script dir (local)
        self.script_dir = os.path.dirname(script_full_path)
        # Local images dir
        base_dir = plots_path if plots_path else self.script_dir
        # Token file path
        self.token_path = os.path.join(self.script_dir, self.token_file)
        # Log file path (local)
        self.log_path = os.path.join(self.script_dir, self.log_file)
        # Results folder in repo
        if not plots_path:
            raise ValueError("plots_path is required for GitHub results folder")
        self.results_folder = os.path.join(plots_path, script_filename).replace(os.sep, '/')
        # Script repo path
        self.script_repo_path = os.path.join(self.scripts_path, f"{script_filename}.py").replace(os.sep, '/')
        # Log repo path
        self.log_repo_path = os.path.join(self.results_folder, self.log_file).replace(os.sep, '/')
        # NEW – expose plots_path for upload_file
        self.plots_path = plots_path

    def _load_token(self):
        token = os.getenv('GITHUB_TOKEN')
        if token:
            self.debug_log('INIT', 'Using GITHUB_TOKEN from environment')
            return token
        
        self.debug_log('INIT', f'GITHUB_TOKEN not in env; loading from {self.token_path}')
        if not os.path.exists(self.token_path):
            raise ValueError(f"Token file not found: {self.token_path}")
        
        with open(self.token_path, 'r') as f:
            token = f.read().strip()
        
        if not token:
            raise ValueError(f"Empty token in {self.token_path}")
        
        return token

    # ----------------------------------------------------------------------
    # NEW helper – delete a path if it already exists on GitHub
    # ----------------------------------------------------------------------
    def _delete_if_exists(self, repo, repo_path):
        try:
            contents = repo.get_contents(repo_path)
            repo.delete_file(repo_path,
                             f"Delete old {repo_path} before upload – {time.strftime('%Y-%m-%d %H:%M:%S')}",
                             contents.sha)
            self.debug_log('INIT', f'Deleted existing {repo_path} on GitHub')
            return True
        except Exception as e:
            if "404" in str(e):
                return False          # does not exist – nothing to delete
            self.debug_log('WARN', f'Could not check/delete {repo_path}: {e}')
            return False

    # ----------------------------------------------------------------------
    # Updated private upload routine – delete + create = clean overwrite
    # ----------------------------------------------------------------------
    def _upload_file_to_github(self, repo, file_path, repo_path):
        """Upload a single file – delete existing file first, then create."""
        if not os.path.exists(file_path):
            self.debug_log('WARN', f'File not found, skipping: {file_path}')
            return
        
        with open(file_path, 'rb') as f:
            content = f.read()
        
        commit_message = f"Upload from {self.script_filename}.py at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # 1. Delete if it already exists
        self._delete_if_exists(repo, repo_path)
        
        # 2. Always create (even if it existed – it is now gone)
        try:
            repo.create_file(repo_path, commit_message, content)
            self.debug_log('INIT', f'Created/Overwrote {repo_path} on GitHub')
        except Exception as e:
            self.debug_log('ERROR', f'Failed to create {repo_path}: {e}')
            raise

    # ----------------------------------------------------------------------
    # Public helper used by upload_3d_results (and any future single-file upload)
    # ----------------------------------------------------------------------
    def upload_file(self, local_file_path, repo_subdir):
        """
        Upload a single local file to <repo_subdir>/<basename>.
        Existing file with the same name is deleted first.
        """
        if not os.path.exists(local_file_path):
            self.debug_log('WARN', f'Local file not found, skipping: {local_file_path}')
            return

        github_token = self._load_token()
        g = Github(github_token, timeout=self.timeout)
        repo = g.get_repo(self.repo_name)

        file_name = os.path.basename(local_file_path)
        repo_path = os.path.join(repo_subdir, file_name).replace(os.sep, '/')

        self._upload_file_to_github(repo, local_file_path, repo_path)

    # ----------------------------------------------------------------------
    # The rest of the original methods – unchanged except they now benefit
    # from the delete-and-create logic inside _upload_file_to_github
    # ----------------------------------------------------------------------
    def _upload_script_to_github(self):
        """Upload and overwrite script file."""
        self.debug_log('INIT', f'Uploading script to {self.script_repo_path}...')
        
        github_token = self._load_token()
        g = Github(github_token, timeout=self.timeout)
        try:
            repo = g.get_repo(self.repo_name)
        except Exception as e:
            self.debug_log('ERROR', f'Failed to access repo {self.repo_name}: {e}')
            raise
        
        self._upload_file_to_github(repo, self.script_full_path, self.script_repo_path)
    

    def _upload_images_to_github(self):
        """Upload images to results folder."""
        self.debug_log('INIT', f'Uploading images to {self.results_folder}/...')
        
        github_token = self._load_token()
        g = Github(github_token, timeout=self.timeout)
        try:
            repo = g.get_repo(self.repo_name)
        except Exception as e:
            self.debug_log('ERROR', f'Failed to access repo {self.repo_name}: {e}')
            raise
        
        if not os.path.exists(self.images_subdir):
            raise ValueError(f"Images directory not found: {self.images_subdir}")
        
        uploaded_count = 0
        for root, dirs, files in os.walk(self.images_subdir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.images_subdir)
                repo_path = os.path.join(self.results_folder, rel_path).replace(os.sep, '/')
                
                self._upload_file_to_github(repo, file_path, repo_path)
                uploaded_count += 1
        
        self.debug_log('INIT', f'Uploaded {uploaded_count} images to {self.results_folder}/')
    

    def _upload_log_to_github(self):
        """Upload log to results folder."""
        self.debug_log('INIT', f'Uploading log to {self.log_repo_path}...')
        
        github_token = self._load_token()
        g = Github(github_token, timeout=self.timeout)
        try:
            repo = g.get_repo(self.repo_name)
        except Exception as e:
            self.debug_log('ERROR', f'Failed to access repo {self.repo_name}: {e}')
            raise
        
        self._upload_file_to_github(repo, self.log_path, self.log_repo_path)
    

    def upload_results(self, upload_log=True):
        """
        Main method:
        1. Upload/overwrites script to scripts/freesurface/SCRIPT_FILENAME.py.
        2. Upload images/log to results/freesurface/SCRIPT_FILENAME/.
        """
        self.debug_log('INIT', 'Simulation complete. Preparing upload to GitHub...')
        
        try:
            self._upload_script_to_github()
            self._upload_images_to_github()
            if upload_log:
                self._upload_log_to_github()
        except Exception as e:
            self.debug_log('ERROR', f'Upload failed: {e}')
            raise


    # ----------------------------------------------------------------------
    # Fixed / expanded upload_3d_results – now really works
    # ----------------------------------------------------------------------
    def upload_3d_results(self, local_dir, patterns):
        """Upload 3D visualization files – overwrites existing ones."""
        for pattern in patterns:
            files = glob.glob(os.path.join(local_dir, pattern))
            for file_path in files:
                # repo_subdir = self.plots_path + "3D_Visualizations/"
                repo_subdir = os.path.join(self.plots_path, "3D_Visualizations").replace(os.sep, '/')
                self.upload_file(file_path, repo_subdir)