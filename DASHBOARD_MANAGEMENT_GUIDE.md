# FalconOne Dashboard Management Guide

## Overview
The FalconOne Dashboard is now a **complete operational control center** that allows researchers to manage all aspects of the SIGINT platform from a web interface, eliminating the need for CLI usage in most scenarios.

## What's New: Complete Dashboard Manageability

### 🎯 Key Enhancements

#### 1. **Comprehensive Exploit Management** (NEW)
- **Interactive Forms**: Each exploit type now has dynamic parameter input forms
- **Help Modals**: Built-in documentation for every attack type with detailed explanations
- **Real-time Monitoring**: Live progress tracking for all exploit operations
- **Operation History**: Complete history of all exploit executions with statistics
- **Export Functionality**: Export results in JSON/CSV formats

#### 2. **Complete API Coverage** (NEW)
Added 20+ new API endpoints:
- `/api/exploits/help/<exploit_type>` - Get detailed documentation
- `/api/exploits/run` - Execute exploits with parameters
- `/api/exploits/stop/<operation_id>` - Stop running operations
- `/api/exploits/status/<operation_id>` - Monitor progress
- `/api/exploits/history` - View execution history
- `/api/exploits/export` - Export results
- `/api/targets/add` - Add new targets
- `/api/targets/delete/<target_id>` - Delete targets
- `/api/targets/update/<target_id>` - Update target info
- `/api/targets/<target_id>/monitor` - Start monitoring
- `/api/targets/<target_id>/stop_monitor` - Stop monitoring
- `/api/captures/filter` - Advanced capture filtering
- `/api/captures/export` - Export captured data
- `/api/captures/delete` - Delete captures
- `/api/captures/analyze` - AI/ML analysis
- `/api/analytics/start` - Start analytics operations
- `/api/analytics/stop/<analytics_id>` - Stop analytics
- `/api/analytics/results/<analytics_id>` - Get results

---

## Exploit Control Center

### Available Exploit Types

#### 1. 🔐 **Cryptographic Attacks**

**A5/1 Stream Cipher Crack**
- **Purpose**: Break GSM A5/1 encryption in real-time
- **Parameters**:
  - Target IMSI (Required)
  - Frame Count (Default: 1000)
  - Rainbow Table (standard/extended)
- **Requirements**: gr-gsm, RTL-SDR
- **Time**: 30-120 seconds

**KASUMI/SNOW 3G Attack**
- **Purpose**: Exploit weaknesses in 3G/4G encryption
- **Parameters**:
  - Target IMSI (Required)
  - Algorithm (KASUMI/SNOW3G)
  - Attack Mode (known_plaintext/chosen_plaintext)
- **Requirements**: LTESniffer, known plaintext samples
- **Time**: 5-15 minutes

**SUCI Deconcealment**
- **Purpose**: Decrypt 5G Subscription Concealed Identifiers
- **Parameters**:
  - SUCI Value (Required)
  - Protection Scheme (Profile_A/Profile_B)
  - Use ML (checkbox)
- **Requirements**: 5G-capable SDR, ML models
- **Time**: 1-5 minutes with ML

---

#### 2. 🛰️ **Non-Terrestrial Network (NTN) Attacks**

**Satellite Communication Hijacking**
- **Purpose**: Intercept/inject messages into satellite links
- **Parameters**:
  - Satellite ID (Required)
  - Frequency MHz (Required)
  - Beam ID (Optional)
  - Mode (monitor/active)
- **Requirements**: High-gain antenna, 5G SDR, satellite tracking
- **Time**: 10-30 minutes setup

**IoT-NTN Vulnerability Exploitation**
- **Purpose**: Exploit IoT devices using NTN connections
- **Parameters**:
  - Device Type (sensor/tracker/meter)
  - Target Identifier (Required)
  - Exploit Type (dos/intercept/inject)
- **Requirements**: NTN-capable equipment
- **Time**: 15-45 minutes

---

#### 3. 🚗 **Vehicle-to-Everything (V2X) Attacks**

**C-V2X Jamming Attack**
- **Purpose**: Disrupt vehicle communications
- **Parameters**:
  - Frequency MHz (Default: 5900.0)
  - Power dBm (Default: 20)
  - Range meters (Default: 300)
- **Requirements**: SDR with 5.9 GHz, directional antenna
- **Time**: Immediate effect

**V2X Message Injection**
- **Purpose**: Inject false messages (false warnings)
- **Parameters**:
  - Message Type (BSM/DENM/CAM)
  - Payload (Required)
  - Repetition ms (Default: 100)
- **Requirements**: V2X SDR, message crafting tools
- **Time**: 5-10 minutes setup

---

#### 4. 💉 **Message Injection Attacks**

**Silent SMS (Type 0)**
- **Purpose**: Send invisible SMS for location tracking
- **Parameters**:
  - Target Phone Number (Required)
  - Message Count (Default: 1)
  - Interval seconds (Default: 300)
- **Requirements**: SMS gateway or cellular modem
- **Time**: Immediate delivery

**Fake Emergency Alert**
- **Purpose**: Broadcast false emergency alerts
- **Parameters**:
  - Alert Type (earthquake/tsunami/storm)
  - Area Code (Required)
  - Message Text (Required)
- **Requirements**: Cell broadcast capability, fake BTS
- **Time**: 1-2 minutes

---

#### 5. 🔽 **Downgrade Attacks**

**LTE to 3G Downgrade**
- **Purpose**: Force UE to downgrade from LTE
- **Parameters**:
  - Target IMSI (Required)
  - Target Generation (3G/2G)
  - Fake PLMN (Optional)
- **Requirements**: Fake BTS (srsRAN, OpenBTS)
- **Time**: 2-5 minutes

**5G Security Context Downgrade**
- **Purpose**: Force 5G device to weaker security
- **Parameters**:
  - Target SUCI (Required)
  - Target Algorithm (NEA0/NEA1/NEA2)
  - Disable Integrity Protection (checkbox)
- **Requirements**: 5G fake gNB
- **Time**: 5-10 minutes

---

#### 6. 📡 **Paging Spoofing**

**Passive Paging Monitoring**
- **Purpose**: Monitor paging to detect active devices
- **Parameters**:
  - Frequency MHz (Required)
  - Bandwidth MHz (Default: 10.0)
  - Duration minutes (Default: 60)
- **Requirements**: SDR, paging decoder
- **Time**: Continuous monitoring

**Paging Storm Attack (DoS)**
- **Purpose**: Flood network with fake paging
- **Parameters**:
  - Target Cell ID (Required)
  - Paging Rate per second (Default: 1000)
  - Duration seconds (Default: 60)
- **Requirements**: Fake BTS, high-speed SDR
- **Time**: Immediate effect

---

#### 7. 🤖 **AIoT (AI + IoT) Exploits**

**AIoT Model Poisoning**
- **Purpose**: Inject adversarial data to poison AI models
- **Parameters**:
  - Device ID (Required)
  - Model Type (image/audio/sensor)
  - Adversarial Method (FGSM/PGD/C&W)
- **Requirements**: Device comm access, adversarial ML tools
- **Time**: 30-60 minutes

**Federated Learning Manipulation**
- **Purpose**: Manipulate federated learning updates
- **Parameters**:
  - Target Network (Required)
  - Manipulation Type (gradient/model/data)
  - Scale Factor (Default: 10.0)
- **Requirements**: Federated learning access, AI expertise
- **Time**: 1-2 hours

---

#### 8. 🌐 **Semantic 6G Attacks**

**Semantic Information Injection**
- **Purpose**: Inject false semantic information into 6G
- **Parameters**:
  - Target Session (Required)
  - Semantic Vector (Required)
  - Confidence Score (Default: 0.95)
- **Requirements**: 6G testbed, semantic comm understanding
- **Time**: 45-90 minutes

**Knowledge Graph Poisoning**
- **Purpose**: Poison distributed knowledge graph
- **Parameters**:
  - Target Knowledge Graph (Required)
  - False Relations JSON (Required)
  - Propagation Mode (direct/indirect)
- **Requirements**: 6G network access, graph theory knowledge
- **Time**: 2-4 hours

---

#### 9. 🛡️ **Security Audit**

**Complete Security Audit**
- **Purpose**: Run all security checks on target
- **Parameters**:
  - Target ID (Required)
  - Audit Depth (quick/standard/deep)
  - Include Crypto (checkbox)
  - Include Network (checkbox)
- **Requirements**: Multi-band SDR, analysis tools
- **Time**: 10-60 minutes depending on depth

**Vulnerability Scanner**
- **Purpose**: Scan for known vulnerabilities
- **Parameters**:
  - Target Type (device/cell/network)
  - CVE Database (local/nist)
- **Requirements**: Vulnerability database
- **Time**: 15-30 minutes

---

## How to Use the Dashboard

### 1. **Running an Exploit**

**Step 1**: Navigate to the Exploits tab
```
Click on "⚔️ Exploits" in the left sidebar
```

**Step 2**: Select exploit type and attack
```
Choose from dropdowns (e.g., Crypto Attacks → A5/1 Stream Cipher Crack)
```

**Step 3**: Fill in parameters
```
- Required fields are marked with a red asterisk (*)
- Optional fields have default values
- Hover over labels for tooltips
```

**Step 4**: Click Help button (optional)
```
Click the "ℹ️ Help" button on each panel for detailed documentation
```

**Step 5**: Execute
```
Click "▶️ Execute" button
You'll receive an Operation ID to track progress
```

**Step 6**: Monitor Progress
```
Results appear in the panel below the form
Operation status updates in real-time via WebSocket
```

---

### 2. **Viewing Exploit History**

```
Click "🕐 View History" button at the top of Exploits tab

Statistics shown:
- Total Operations
- Successful (green)
- Failed (red)
- Stopped (orange)

Table columns:
- Time (when started)
- Type (exploit category)
- Attack (specific attack name)
- Status (completed/failed/stopped/running)
- Duration (in minutes)
```

---

### 3. **Exporting Results**

```
Click "⬇️ Export Results" button

Choose format:
- JSON (full data with metadata)
- CSV (tabular format for Excel)

Files saved to: ./exports/exploit_results_TIMESTAMP.format
```

---

### 4. **Target Management**

**Add New Target**:
```javascript
// Navigate to Devices tab → Targets section
// Click "Add Target" button
// Fill in:
{
  "name": "Target Name",
  "type": "device|cell|network",
  "imsi": "123456789012345",  // Optional
  "msisdn": "+1234567890",    // Optional
  "suci": "suci-value",       // Optional
  "generation": "5G",
  "tags": ["tag1", "tag2"],
  "notes": "Description"
}
```

**Start Monitoring**:
```
Select target from list
Click "Start Monitoring"
Choose generation (2G/3G/4G/5G/6G)
Set monitoring options
```

**Stop Monitoring**:
```
Click "Stop Monitoring" button on active target
View monitoring duration and summary
```

---

### 5. **Capture Management**

**Filter Captures**:
```
Navigate to Captures tab
Use filter panel:
- Generation: 2G/3G/4G/5G/6G
- Protocol: Specific protocol
- Time Range: Start/End timestamps
- Search: Text search across all fields
```

**Analyze Captures**:
```
Select captures (checkboxes)
Click "Analyze" button
Choose analysis type:
- Signal Classification
- Anomaly Detection
- Protocol Analysis
- Pattern Recognition
```

**Export Captures**:
```
Select captures or export all
Click "Export" button
Choose format: JSON/CSV/PCAP
Download from exports directory
```

---

### 6. **Analytics Operations**

**Start Analytics**:
```
Navigate to Analytics tab
Click "Start" on desired analytics type:
- Spectrum Analysis
- Cyber-RF Fusion
- Signal Classification
- Federated Agents
- RIC Optimization
- Carbon Emissions
- Precision Geolocation

Set parameters and click "Start"
```

**View Results**:
```
Results update in real-time
Charts and visualizations render automatically
Export data using "Export" button
```

---

## Dashboard vs CLI Comparison

| Feature | CLI | Dashboard | Status |
|---------|-----|-----------|--------|
| **Exploits** | ✅ Full | ✅ Full with UI | ⭐ **Enhanced** |
| **Targets** | ✅ Full | ✅ Full with UI | ⭐ **Enhanced** |
| **Monitoring** | ✅ Full | ✅ Full with UI | ⭐ **Enhanced** |
| **Captures** | ✅ Full | ✅ Full with filters | ⭐ **Enhanced** |
| **Analytics** | ✅ Full | ✅ Full with charts | ⭐ **Enhanced** |
| **System Tools** | ✅ Full | ✅ Full with testing | ⭐ **Enhanced** |
| **Export** | ⚠️ Limited | ✅ Multiple formats | ⭐ **Enhanced** |
| **Help Docs** | ❌ None | ✅ Built-in modals | 🆕 **New** |
| **History** | ⚠️ Logs only | ✅ Structured view | 🆕 **New** |
| **Real-time** | ❌ No | ✅ WebSocket updates | 🆕 **New** |

**Conclusion**: Dashboard now provides **100% CLI functionality** with additional benefits like real-time updates, visual feedback, and built-in documentation.

---

## Security Considerations

### Authentication
```
Dashboard requires authentication by default
Default login: admin / Change password on first login!
```

### Authorization Levels
```
- Admin: Full access to all features
- Operator: Execute operations, view data
- Viewer: Read-only access
```

### Rate Limiting
```
- API calls limited to prevent abuse
- Exploit operations have cooldown periods
- Export operations are throttled
```

### Audit Logging
```
All operations are logged:
- Timestamp
- User
- Action
- Parameters
- Result
```

---

## Troubleshooting

### Issue: Exploit Won't Start
**Solutions**:
1. Check required parameters are filled
2. Verify target is valid and accessible
3. Ensure SDR device is connected
4. Check system tools are installed
5. View logs in Terminal tab

### Issue: Real-time Updates Not Working
**Solutions**:
1. Check WebSocket connection (console)
2. Refresh page to reconnect
3. Verify network/firewall settings
4. Check server status in System tab

### Issue: Export Failed
**Solutions**:
1. Check exports directory exists
2. Verify write permissions
3. Ensure disk space available
4. Try different format (JSON vs CSV)

### Issue: Help Modal Not Loading
**Solutions**:
1. Check API endpoint status
2. Clear browser cache
3. Verify authentication token
4. Check console for errors

---

## API Reference

### Exploit Management

#### Get Exploit Help
```http
GET /api/exploits/help/<exploit_type>

Response:
{
  "success": true,
  "exploit_type": "crypto",
  "documentation": {
    "name": "Cryptographic Attacks",
    "description": "...",
    "attacks": { ... },
    "general_info": "..."
  }
}
```

#### Execute Exploit
```http
POST /api/exploits/run
Content-Type: application/json

{
  "exploit_type": "crypto",
  "attack_name": "a5_1_crack",
  "target_id": "target-123",
  "parameters": {
    "target_imsi": "123456789012345",
    "frame_count": 1000,
    "rainbow_table": "standard"
  }
}

Response:
{
  "success": true,
  "operation_id": "op-uuid-123",
  "message": "Exploit started successfully",
  "estimated_duration": "5-30 minutes"
}
```

#### Get Operation Status
```http
GET /api/exploits/status/<operation_id>

Response:
{
  "success": true,
  "status": {
    "id": "op-uuid-123",
    "exploit_type": "crypto",
    "attack_name": "a5_1_crack",
    "status": "running",
    "progress": 45,
    "start_time": 1234567890,
    "logs": [...]
  }
}
```

#### Stop Operation
```http
POST /api/exploits/stop/<operation_id>

Response:
{
  "success": true,
  "message": "Operation stopped successfully"
}
```

### Target Management

#### Add Target
```http
POST /api/targets/add
Content-Type: application/json

{
  "name": "Test Device",
  "type": "device",
  "imsi": "123456789012345",
  "generation": "5G",
  "tags": ["test", "5g"]
}

Response:
{
  "success": true,
  "target_id": "target-uuid-123"
}
```

#### Start Monitoring
```http
POST /api/targets/<target_id>/monitor
Content-Type: application/json

{
  "generation": "5G",
  "options": {
    "capture_suci": true,
    "analyze_traffic": true
  }
}

Response:
{
  "success": true,
  "message": "Started monitoring target on 5G"
}
```

### Capture Management

#### Filter Captures
```http
GET /api/captures/filter?generation=5G&protocol=NAS&limit=100

Response:
{
  "captures": [...],
  "count": 100,
  "filters_applied": {...}
}
```

#### Analyze Captures
```http
POST /api/captures/analyze
Content-Type: application/json

{
  "capture_ids": ["cap-1", "cap-2"],
  "analysis_type": "signal_classification"
}

Response:
{
  "success": true,
  "analysis_type": "signal_classification",
  "results": {
    "summary": "...",
    "findings": [...],
    "confidence": 0.85
  }
}
```

---

## Best Practices

### 1. **Always Read Help Documentation**
- Click Help button before using new exploit
- Understand risks and requirements
- Know expected execution time

### 2. **Test in Controlled Environment**
- Use test targets first
- Verify equipment setup
- Start with passive monitoring

### 3. **Monitor Operations**
- Check progress regularly
- Review logs for errors
- Stop if unexpected behavior

### 4. **Export Regularly**
- Export results after each session
- Keep backups of important captures
- Document findings externally

### 5. **Legal Compliance**
- Only test on authorized targets
- Follow local regulations
- Maintain proper documentation

---

## Future Enhancements

### Planned Features (v1.8.0)
- [ ] Batch exploit execution
- [ ] Custom exploit scripting
- [ ] Advanced scheduling
- [ ] Collaborative mode (multi-user)
- [ ] Machine learning-based recommendations
- [ ] Automated vulnerability assessment
- [ ] Integration with external tools (Wireshark, etc.)
- [ ] Mobile app companion
- [ ] Voice command support
- [ ] AR/VR visualization

---

## Support

### Getting Help
- 📖 Documentation: See USER_MANUAL.md
- 🐛 Issues: Check logs in Terminal tab
- 💬 Community: Join FalconOne community
- 📧 Email: support@falconone.io

### Contributing
We welcome contributions! To add new exploits:
1. Add exploit logic to appropriate module
2. Update `_get_exploit_help()` with documentation
3. Add form generation in JavaScript
4. Test thoroughly
5. Submit pull request

---

## Changelog

### v1.7.1 (Current)
- ✅ Complete exploit management UI
- ✅ Interactive parameter forms for all exploit types
- ✅ Built-in help documentation modals
- ✅ Real-time operation monitoring
- ✅ Exploit history with statistics
- ✅ Export functionality (JSON/CSV)
- ✅ Enhanced target management
- ✅ Advanced capture filtering
- ✅ AI/ML capture analysis
- ✅ Analytics operation control
- ✅ 20+ new API endpoints
- ✅ Comprehensive CSS styling
- ✅ Responsive design improvements

### Previous Versions
- v1.7.0: System Tools Management, Phase 1 features
- v1.6.0: Multi-user authentication, RBAC
- v1.5.0: Real-time WebSocket updates
- v1.0.0: Initial dashboard release

---

## License
FalconOne Dashboard is proprietary software. See LICENSE file for details.

---

## Disclaimer
⚠️ **IMPORTANT**: This software is for authorized security research and testing only. Unauthorized interception of communications is illegal in most jurisdictions. Users are solely responsible for compliance with all applicable laws and regulations.

---

**Last Updated**: 2024
**Version**: 1.7.1
**Status**: Production Ready ✅
