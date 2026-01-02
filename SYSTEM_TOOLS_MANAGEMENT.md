# System Tools Management Feature

## Overview
The **System Tools Management** feature has been added to the FalconOne Dashboard to provide a unified interface for managing external system tools required for cellular monitoring operations. This includes tools like gr-gsm, LTESniffer, srsRAN, GNU Radio, and more.

## Feature Location
- **Navigation Tab**: 🛠️ System Tools (10th tab in the sidebar)
- **API Endpoints**: `/api/system_tools/*`
- **Backend Methods**: `_get_system_tools_status()`, `_install_system_tool()`, `_uninstall_system_tool()`, `_test_system_tool()`

## Managed System Tools

### 1. GSM Tools
- **gr-gsm** - GSM signal monitoring with GNU Radio
- **kalibrate-rtl** - GSM base station scanner  
- **OsmocomBB** - Open Source mobile communications baseband

### 2. LTE Tools
- **LTESniffer** - LTE downlink/uplink sniffer

### 3. 5G Tools
- **srsRAN** - Open source SDR 4G/5G software radio
- **srsRAN Project** - 5G RAN implementation
- **Open5GS** - 5G Core Network
- **OpenAirInterface (OAI)** - Open source 5G software

### 4. SDR Drivers
- **UHD** - USRP Hardware Driver (Ettus Research)
- **BladeRF** - Nuand bladeRF libraries

### 5. Frameworks
- **GNU Radio** - Software radio toolkit
- **SoapySDR** - Vendor neutral SDR API
- **gr-osmosdr** - GNU Radio OsmoSDR source block

## Features

### Status Monitoring
- Real-time detection of installed vs. missing tools
- Version detection for installed tools
- Category grouping (GSM, LTE, 5G, SDR Drivers, Frameworks)
- Completion percentage tracking
- Visual status indicators (✅ Installed / ❌ Not Installed)

### Installation Management
- One-click access to installation commands
- Copy-to-clipboard functionality
- Detailed installation instructions
- sudo privilege warnings
- System-appropriate package manager commands

### Testing & Verification
- Test individual tools for functionality
- Diagnostic output display
- Working/Not Working status indicators
- Error message reporting

### Uninstallation
- Confirmation prompts before uninstall
- Clean removal commands
- Automatic package manager detection

## API Endpoints

### GET /api/system_tools/status
Returns the status of all system tools.

**Response:**
```json
{
  "tools": {
    "gr-gsm": {
      "id": "gr-gsm",
      "name": "gr-gsm",
      "description": "GSM signal monitoring with GNU Radio",
      "category": "GSM",
      "icon": "📱",
      "installed": true,
      "status": "ready",
      "version": "1.0.0",
      "install_cmd": "sudo apt-get install gr-gsm"
    },
    ...
  },
  "total": 13,
  "installed": 5,
  "missing": 8,
  "completion_percent": 38.5,
  "timestamp": 1234567890.123
}
```

### POST /api/system_tools/install
Initiates installation of a system tool.

**Request:**
```json
{
  "tool": "gr-gsm"
}
```

**Response:**
```json
{
  "success": true,
  "tool": "gr-gsm",
  "command": "sudo apt-get install gr-gsm",
  "message": "Installation command prepared for gr-gsm",
  "instructions": "Please run the following command in your terminal with sudo privileges:\nsudo apt-get install gr-gsm",
  "requires_sudo": true
}
```

### POST /api/system_tools/uninstall
Initiates uninstallation of a system tool.

**Request:**
```json
{
  "tool": "gr-gsm"
}
```

**Response:**
```json
{
  "success": true,
  "tool": "gr-gsm",
  "command": "sudo apt-get remove --purge gr-gsm",
  "message": "Uninstallation command prepared for gr-gsm",
  "instructions": "Please run the following command in your terminal with sudo privileges:\nsudo apt-get remove --purge gr-gsm",
  "requires_sudo": true
}
```

### POST /api/system_tools/test
Tests a system tool for functionality.

**Request:**
```json
{
  "tool": "gr-gsm"
}
```

**Response:**
```json
{
  "success": true,
  "tool": "gr-gsm",
  "working": true,
  "message": "gr-gsm is working correctly",
  "output": "grgsm_livemon version 1.0.0\n..."
}
```

## Backend Implementation

### _get_system_tools_status()
- Scans system for all 13 configured tools
- Executes test commands (with 3-second timeout)
- Extracts version information from output
- Calculates completion statistics
- Returns comprehensive status object

### _install_system_tool(tool_name)
- Validates tool name
- Returns installation command for the tool
- Provides user instructions
- Notes sudo requirements

### _uninstall_system_tool(tool_name)
- Maps tool to uninstallation command
- Confirms user intent
- Returns removal command with instructions

### _test_system_tool(tool_name)
- Validates tool is installed
- Executes test command (5-second timeout)
- Captures output for diagnostics
- Returns working status

### _extract_version(output)
- Parses version information from command output
- Tries multiple regex patterns:
  - `version \d+\.\d+\.?\d*`
  - `v\d+\.\d+\.?\d*`
  - `\d+\.\d+\.?\d+`
- Returns "unknown" if no version found

## Frontend UI Components

### Summary Dashboard
Located at the top of the System Tools tab:
- **Installed Tools** - Count of ready tools (green)
- **Missing Tools** - Count of uninstalled tools (red)
- **Completion** - Percentage of tools installed (blue)
- **Refresh Button** - Manually update status

### Tools Grid
Dynamic grid showing all 13 tools with:
- Tool icon and name
- Category badge
- Description
- Installation status badge
- Version information (if installed)
- Action buttons:
  - **Install** - For missing tools
  - **Test** - For installed tools
  - **Uninstall** - For installed tools
  - **Details** - View more information

### Tool Details Modal
Popup modal displaying:
- Installation instructions with command
- Copy-to-clipboard button
- Test output (when testing)
- Status messages
- Close button (✕)

### Notification System
Toast notifications for:
- ✅ Success (green background)
- ❌ Error (red background)
- ⚠️ Warning (orange background)
- ℹ️ Info (blue background)

## JavaScript Functions

### loadSystemToolsStatus()
- Fetches tool status from API
- Updates summary statistics
- Renders tools grid
- Groups tools by category
- Handles loading states and errors

### installSystemTool(toolName)
- Posts install request to API
- Shows tool details modal with instructions
- Displays success/error notifications

### uninstallSystemTool(toolName)
- Confirms user intent
- Posts uninstall request to API
- Shows removal instructions
- Displays success/error notifications

### testSystemTool(toolName)
- Posts test request to API
- Shows testing notification
- Displays test results
- Shows diagnostic output in modal

### showToolDetails(toolName, data)
- Opens modal overlay
- Renders installation instructions
- Shows test output
- Provides copy-to-clipboard functionality

### closeToolDetails()
- Closes the tool details modal

### showNotification(message, type)
- Creates toast notification
- Auto-dismisses after 4 seconds
- Color-coded by type

### Auto-load on Tab Switch
When the "System Tools" tab is opened:
- Automatically calls `loadSystemToolsStatus()`
- 300ms delay for smooth transition

## Usage Instructions

### For End Users

1. **Access the System Tools Tab**
   - Open FalconOne Dashboard (http://127.0.0.1:5000)
   - Click "🛠️ System Tools" in the left sidebar

2. **View Tool Status**
   - Green badges (✅) indicate installed tools
   - Red badges (❌) indicate missing tools
   - Check completion percentage at the top

3. **Install a Tool**
   - Click "📥 Install" button on a missing tool
   - Copy the installation command from the modal
   - Run the command in your system terminal with sudo
   - Refresh the status to verify installation

4. **Test a Tool**
   - Click "🧪 Test" button on an installed tool
   - View test results in the notification
   - Check details modal for diagnostic output

5. **Uninstall a Tool**
   - Click "🗑️ Uninstall" button
   - Confirm the uninstallation
   - Copy the removal command
   - Run in terminal with sudo

### For Developers

#### Adding New Tools
To add a new system tool, edit `_get_system_tools_status()` in [dashboard.py](falconone/ui/dashboard.py):

```python
system_tools = {
    'my_new_tool': {
        'name': 'My New Tool',
        'description': 'Description of the tool',
        'test_cmd': 'my_tool --version',
        'install_cmd': 'sudo apt-get install my-tool',
        'category': 'Category Name',
        'icon': '🔧'
    },
    ...
}
```

#### Customizing Test Commands
Each tool's `test_cmd` should:
- Exit with code 0 if installed
- Exit with non-zero if not found
- Ideally output version information

#### Adding Uninstall Commands
Edit `_uninstall_system_tool()` to add uninstall mapping:

```python
uninstall_map = {
    'my_new_tool': 'sudo apt-get remove --purge my-tool',
    ...
}
```

## Technical Details

### File Locations
- **Backend**: `falconone/ui/dashboard.py` (lines ~3050-3340)
- **API Routes**: `falconone/ui/dashboard.py` (lines ~950-1000)
- **Frontend UI**: `falconone/ui/dashboard.py` (lines ~5700-5800)
- **JavaScript**: `falconone/ui/dashboard.py` (lines ~7520-7740)

### Dependencies
- Python 3.7+
- Flask 3.1.2
- subprocess module (built-in)
- re module (built-in)

### Security Considerations
- All installation commands require sudo privileges
- Commands are NOT executed automatically by the dashboard
- Users must manually copy and run commands in their terminal
- This design prevents unauthorized system modifications
- Test commands run with 3-5 second timeouts to prevent hangs

### Error Handling
- Connection errors display retry button
- Unknown tools return error messages
- Missing tools show installation instructions
- Test failures show diagnostic output
- API errors are caught and logged

## Testing

### Manual Testing
1. Start dashboard: `python start_dashboard.py`
2. Navigate to System Tools tab
3. Verify tool detection works
4. Test install command generation
5. Test tool testing functionality
6. Verify modal display and copy-to-clipboard

### Automated Testing
Create test cases for:
- `_get_system_tools_status()` - Verify all tools detected
- `_install_system_tool()` - Verify command generation
- `_uninstall_system_tool()` - Verify removal commands
- `_test_system_tool()` - Verify test execution
- API endpoints - Verify JSON responses

## Future Enhancements

### Planned Features
- [ ] Automated installation (with user confirmation)
- [ ] Dependency checking (install prerequisites automatically)
- [ ] Version update notifications
- [ ] Tool compatibility matrix
- [ ] Installation progress tracking
- [ ] Package manager auto-detection (apt, yum, pacman, brew)
- [ ] Docker container deployment option
- [ ] Pre-built binary downloads

### Possible Improvements
- WebSocket real-time installation progress
- Multi-tool installation queue
- Rollback functionality
- Configuration management
- Tool-specific settings panels
- Integration with CI/CD pipelines

## Changelog

### v1.7.0 (Current)
- ✅ Added System Tools Management tab
- ✅ Implemented 13 system tools monitoring
- ✅ Created 4 new API endpoints
- ✅ Built responsive UI with status grid
- ✅ Added copy-to-clipboard functionality
- ✅ Implemented testing & diagnostics
- ✅ Fixed syntax errors in dashboard.py (escape sequences)

## Support

For issues or questions:
1. Check the FalconOne documentation
2. Review API endpoint responses for error messages
3. Check browser console for JavaScript errors
4. Verify tool installation manually in terminal
5. Contact FalconOne support team

## License
Part of the FalconOne SIGINT Platform v1.7.0
For research and authorized use only.

---
**Last Updated**: January 2025
**Author**: FalconOne Development Team
**Version**: 1.0.0
