@echo off
echo ====================================
echo FalconOne v1.9.0 - Git Update Script
echo ====================================
echo.

cd /d "%~dp0"

echo Checking git status...
git status
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Git is not installed or not in PATH
    echo Please install Git from: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo.
echo Staging all changes...
git add .

echo.
echo Creating commit...
git commit -m "docs: Fix critical inconsistencies and expand documentation (v1.9.0)

- Fix CVE count: 97→96 across all documentation (matches actual database)
- Standardize Python requirement: 3.11+ (not 3.10+) for pattern matching
- Fix test metrics: 485 lines (not 700+) in test_ransacked_exploits.py
- Clarify OpenAPI spec status: Planned, not currently available
- Document Faraday cage: Manual verification required (auto-detection TODO)
- Expand CHANGELOG v1.7.0: 13→200+ lines with detailed feature descriptions
- Add config.yaml.example: 800+ lines comprehensive annotated configuration
- Update INSTALLATION.md with configuration guide and examples

Files modified:
- README.md (CVE counts, test lines, Faraday cage guidance)
- API_DOCUMENTATION.md (CVE counts, test lines, OpenAPI status)
- DEVELOPER_GUIDE.md (test lines)
- DOCUMENTATION_INDEX.md (CVE count)
- INSTALLATION.md (Python version, config guide)
- CHANGELOG.md (expanded v1.7.0 entry)
- config/config.yaml.example (new file, 800+ lines)

All documentation now accurately reflects v1.9.0 codebase implementation."

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Commit created successfully!
    echo.
    echo Pushing to remote repository...
    git push
    
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo ====================================
        echo SUCCESS: Repository updated!
        echo ====================================
    ) else (
        echo.
        echo ERROR: Push failed. Please check your credentials and network connection.
    )
) else (
    echo.
    echo No changes to commit or commit failed.
)

echo.
pause
