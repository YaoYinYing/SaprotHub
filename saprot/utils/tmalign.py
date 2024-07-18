import os
import subprocess
import requests
import platform
import subprocess
from typing import Optional

class TMalignSetup:
    """
    A class to handle the fetching, modifying, and compiling of the TMalign binary.

    Attributes:
        binary_path (str): The path where the TMalign binary should be located.
        source_url (str): The URL from which to fetch the TMalign source code.
    """

    def __init__(self, binary_path: str = "./bin/TMalign", source_url: str = "https://zhanggroup.org/TM-align/TMalign.cpp"):
        """
        Initializes the TMalignCompiler with the specified binary path and source URL.

        Args:
            binary_path (str): The path where the TMalign binary should be located.
            source_url (str): The URL from which to fetch the TMalign source code.
        """
        self.binary_path: str = binary_path
        self.source_url: str = source_url


    def check_binary(self) -> bool:
        """
        Checks if the TMalign binary exists at the specified path.

        Returns:
            bool: True if the binary exists, False otherwise.
        """
        return os.path.isfile(self.binary_path)

    def check_compile_toolchain(self) -> None:
        """
        Checks if the g++ compiler is accessible on the system.

        Raises:
            RuntimeError: If g++ is not accessible.
        """
        print("Checking if g++ is accessible...")
        if subprocess.call(['which', 'g++'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
            raise RuntimeError("g++ is not accessible on this system.")
        print("g++ is accessible.")

    def fetch_source_code(self) -> None:
        """
        Fetches the TMalign source code from the specified URL.

        Raises:
            RuntimeError: If the source code could not be fetched.
        """
        print(f"Fetching source code from {self.source_url}...")
        response = requests.get(self.source_url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch source code from {self.source_url}")
        
        with open("TMalign.cpp", "w") as source_file:
            source_file.write(response.text)
        print("Source code fetched successfully.")

    def modify_source_code(self) -> None:
        """
        Modifies the TMalign source code for compatibility with macOS if necessary.
        """
        print("Modifying source code for macOS compatibility if needed...")
        with open("TMalign.cpp", "r") as source_file:
            source_code = source_file.read()
        
        if platform.system() == "Darwin":  # macOS
            modified_source_code = source_code.replace("#include <malloc.h>", "#include <stdlib.h>")
            with open("TMalign.cpp", "w") as source_file:
                source_file.write(modified_source_code)
            print("Source code modified for macOS.")

    def compile_source_code(self) -> None:
        """
        Compiles the TMalign source code into a binary.

        Raises:
            RuntimeError: If the compilation fails.
        """
        print("Compiling the source code...")
        compile_command = ["g++", "-O3", "-ffast-math", "-lm", "-o", self.binary_path, "TMalign.cpp"]
        if platform.system() != "Darwin":  # Only add -static flag for non-macOS systems
            compile_command.insert(1, "-static")

        result = subprocess.run(compile_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed: {result.stderr.decode('utf-8')}")
        print("Compilation successful.")

    def ensure_binary(self) -> str:
        """
        Ensures that the TMalign binary is available. If not, it fetches the source code,
        modifies it if necessary, and compiles it.

        Returns:
            str: The path to the TMalign binary.

        Raises:
            RuntimeError: If the binary could not be compiled.
        """
        if self.check_binary():
            print(f"Binary already exists at {self.binary_path}.")
            os.chmod(self.binary_path, 0o755)  # Make the binary executable
            return self.binary_path
        
        self.check_compile_toolchain()
        os.makedirs(os.path.dirname(self.binary_path), exist_ok=True)
        self.fetch_source_code()
        self.modify_source_code()  # Modify the source code before compilation
        self.compile_source_code()
        
        if not self.check_binary():
            raise RuntimeError("Failed to compile TMalign binary.")
        
        os.chmod(self.binary_path, 0o755)  # Make the binary executable
        print(f"Binary compiled and available at {self.binary_path}.")
        return self.binary_path
    



class TMalign:
    """
    A wrapper class for calling the TMalign program from Python.

    Attributes:
        binary_path (str): The path to the TMalign binary.
    """

    def __init__(self, binary_path: str = "./bin/TMalign"):
        """
        Initializes the TMalignWrapper with the specified binary path.

        Args:
            binary_path (str): The path to the TMalign binary.
        """
        self.binary_path: str = binary_path
        if not os.path.isfile(self.binary_path):
            raise RuntimeError(f"TMalign binary not found at {self.binary_path}")

    def align(self, pdb1: str, pdb2: str, align_file: Optional[str] = None, stick_align: bool = False,
              output_superposition: Optional[str] = None, normalize_length: Optional[int] = None,
              scale_d0: Optional[float] = None, output_matrix: Optional[str] = None, 
              based_on_sequence: bool = False) -> str:
        """
        Runs the TMalign program with the specified parameters.

        Args:
            pdb1 (str): The first PDB file.
            pdb2 (str): The second PDB file.
            align_file (Optional[str]): The alignment file (for -i or -I options).
            stick_align (bool): Whether to stick the alignment to the alignment file.
            output_superposition (Optional[str]): The output superposition file.
            normalize_length (Optional[int]): The length to normalize TM-score.
            scale_d0 (Optional[float]): The d0 scale value.
            output_matrix (Optional[str]): The output rotation matrix file.
            based_on_sequence (bool): Whether to calculate TM-score based on sequence alignment.

        Returns:
            str: The standard output from the TMalign execution.
        """
        cmd = [self.binary_path, pdb1, pdb2]

        if align_file:
            cmd += ['-I' if stick_align else '-i', align_file]
        
        if output_superposition:
            cmd += ['-o', output_superposition]
        
        if normalize_length:
            cmd += ['-L', str(normalize_length)]
        
        if scale_d0:
            cmd += ['-d', str(scale_d0)]
        
        if output_matrix:
            cmd += ['-m', output_matrix]
        
        if based_on_sequence:
            cmd += ['-seq']

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            raise RuntimeError(f"TMalign failed with error: {result.stderr.decode('utf-8')}")
        
        return result.stdout.decode('utf-8')