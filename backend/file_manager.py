import os
import glob

class FileManager:
    def __init__(self, strategies_dir: str = "strategies"):
        # Ensure the path is absolute
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.strategies_dir = os.path.join(base_path, strategies_dir)
        if not os.path.exists(self.strategies_dir):
            os.makedirs(self.strategies_dir)

    def list_strategies(self):
        """List all .py files in the strategies directory."""
        files = glob.glob(os.path.join(self.strategies_dir, "*.py"))
        return [os.path.basename(f) for f in files]

    def read_strategy(self, filename: str):
        """Read the content of a specific strategy file."""
        self._validate_filename(filename)
        filepath = os.path.join(self.strategies_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Strategy file '{filename}' not found.")
            
        with open(filepath, "r") as f:
            return f.read()

    def save_strategy(self, filename: str, content: str):
        """Save content to a strategy file."""
        self._validate_filename(filename)
        filepath = os.path.join(self.strategies_dir, filename)
        
        with open(filepath, "w") as f:
            f.write(content)
        return {"status": "success", "filename": filename}

    def _validate_filename(self, filename: str):
        """Simple validation to prevent directory traversal."""
        if ".." in filename or "/" in filename or "\\" in filename:
            raise ValueError("Invalid filename: Directory traversal not allowed.")
        if not filename.endswith(".py"):
            raise ValueError("Invalid filename: Must be a .py file.")
