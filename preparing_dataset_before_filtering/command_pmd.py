from pathlib import Path
import subprocess
import sys


# ============================================================
# PROJECT PATHS
# ============================================================

# Hybrid-Code-Review project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# PMD installation
PMD_PATH = PROJECT_ROOT / "tools" / "pmd" / "bin" / "pmd.bat"

# Java source files
JAVA_FOLDER = PROJECT_ROOT / "final_formatted_java_code"

# PMD ruleset
RULESET_PATH = PROJECT_ROOT / "tools" / "pmd" / "ruleset.xml"

# PMD output
OUTPUT_FOLDER = PROJECT_ROOT / "final_pmd_output"


# ============================================================
# CHECK PATHS
# ============================================================

if not PMD_PATH.exists():
    print(f"ERROR: PMD was not found at:")
    print(PMD_PATH)
    sys.exit(1)

if not JAVA_FOLDER.exists():
    print(f"ERROR: Java folder was not found at:")
    print(JAVA_FOLDER)
    print()
    print("Create the folder first or change JAVA_FOLDER.")
    sys.exit(1)

if not RULESET_PATH.exists():
    print(f"ERROR: PMD ruleset was not found at:")
    print(RULESET_PATH)
    sys.exit(1)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND JAVA FILES
# ============================================================

java_files = sorted(JAVA_FOLDER.glob("*.java"))

if not java_files:
    print(f"No Java files found in:")
    print(JAVA_FOLDER)
    sys.exit(0)


print(f"Project root : {PROJECT_ROOT}")
print(f"Java folder  : {JAVA_FOLDER}")
print(f"PMD          : {PMD_PATH}")
print(f"Ruleset      : {RULESET_PATH}")
print(f"Output       : {OUTPUT_FOLDER}")
print()
print(f"Found {len(java_files)} Java file(s).")
print()


# ============================================================
# RUN PMD
# ============================================================

for java_file in java_files:

    output_file = OUTPUT_FOLDER / f"{java_file.stem}.xml"

    print(f"Analyzing: {java_file.name}")

    command = [
        str(PMD_PATH),
        "check",
        "-d",
        str(java_file),
        "-f",
        "xml",
        "-R",
        str(RULESET_PATH)
    ]

    try:

        with open(output_file, "w", encoding="utf-8") as output:

            result = subprocess.run(
                command,
                stdout=output,
                stderr=subprocess.PIPE,
                text=True
            )

        if result.returncode not in (0, 4):

            print(f"PMD returned an unexpected exit code: {result.returncode}")

            if result.stderr:
                print(result.stderr)

        print(f"Output: {output_file}")
        print()

    except Exception as error:

        print(f"ERROR while analyzing {java_file.name}")
        print(error)


print("PMD analysis completed.")