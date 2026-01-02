# FalconOne System Dependencies

**Version:** 1.8.0  
**Last Updated:** January 1, 2026  
**Platform:** Linux (Ubuntu 20.04+, Debian 11+, Kali Linux 2024+)

---

## Core Cellular & Exploit Stack

FalconOne requires the following components for full RANSacked exploit capabilities:

### 1. SDR & Radio Processing

#### gr-gsm (GNU Radio GSM)
**Purpose:** GSM protocol analysis, IMSI catching, SMS interception  
**Version:** 0.42.2+  
**Installation:**
```bash
sudo apt-get install gr-gsm
# Or build from source:
git clone https://github.com/ptrkrysik/gr-gsm.git
cd gr-gsm && mkdir build && cd build
cmake .. && make -j$(nproc) && sudo make install
```

**Used For:**
- GSM BCCH/CCCH channel decoding
- IMSI catching from paging messages
- A5/1 encryption cracking
- FALCON-001 (Rogue Base Station) exploit

#### kalibrate-rtl
**Purpose:** GSM base station frequency calibration  
**Installation:**
```bash
sudo apt-get install kalibrate-rtl
# Or: git clone https://github.com/steve-m/kalibrate-rtl.git
```

**Used For:**
- Scanning for GSM cells (ARFCN discovery)
- Frequency offset calibration
- Rogue cell deployment preparation

#### GNU Radio
**Purpose:** SDR signal processing framework  
**Version:** 3.10.5+  
**Installation:**
```bash
sudo apt-get install gnuradio gr-osmosdr
```

**Used For:**
- Custom PHY layer implementations
- Signal generation/analysis
- All FALCON-series exploits

---

### 2. Cellular Core Networks (Exploit Targets)

#### Open5GS (5G/LTE Core)
**Purpose:** Primary exploit target - 14 RANSacked CVEs  
**Version:** 2.7.0+  
**Installation:**
```bash
sudo add-apt-repository ppa:open5gs/latest
sudo apt-get update && sudo apt-get install open5gs
```

**Exploits Supported:**
- CVE-2024-24428 (Zero-length NAS DoS) - 95% success
- CVE-2024-24427 (Malformed SUCI assertion)
- CVE-2024-24425 (OOB read in AMF)
- CVE-2024-24426 (Missing IE assertions)
- +10 additional CVEs

#### srsRAN / srsRAN Project
**Purpose:** LTE/5G simulation - 24 RANSacked CVEs  
**Version:** 23.11+  
**Installation:**
```bash
sudo apt-get install srsran
# Or: git clone https://github.com/srsran/srsRAN_Project.git
```

**Components:**
- srsenb (LTE eNodeB)
- srsue (LTE/5G UE)
- srsgnb (5G gNB)

**Exploits Supported:**
- CVE-2019-19770 (RRC MIB parsing overflow)
- CVE-2022-39330 (5G RRC setup heap overflow)
- +22 additional CVEs

#### OpenAirInterface (OAI)
**Purpose:** 5G gNB/UE simulation - 18 RANSacked CVEs  
**Installation:**
```bash
git clone https://gitlab.eurecom.fr/oai/openairinterface5g.git
cd openairinterface5g
source oaienv && cd cmake_targets
./build_oai --gNB --UE -w USRP
```

**Exploits Supported:**
- CVE-2024-24445 (Null dereference NGAP)
- CVE-2024-24450 (Buffer overflow PDU Session)
- +16 additional CVEs

---

### 3. LTE/5G Sniffing & Analysis

#### LTESniffer
**Purpose:** LTE downlink sniffing and IMSI extraction  
**Version:** 2.0+  
**Installation:**
```bash
git clone https://github.com/SysSec-KAIST/LTESniffer.git
cd LTESniffer && mkdir build && cd build
cmake .. && make -j$(nproc) && sudo make install
```

**Used For:**
- LTE PDSCH/PDCCH decoding
- Capturing attach requests
- IMSI/TMSI extraction
- Real-time packet auditing for CVE detection

#### OsmocomBB
**Purpose:** GSM baseband firmware reverse engineering  
**Installation:**
```bash
git clone https://git.osmocom.org/osmocom-bb
cd osmocom-bb && git submodule update --init
cd src/host/layer23 && make
```

**Used For:**
- GSM layer 2/3 protocol implementation
- Custom GSM packet injection
- RACH flooding attacks

---

### 4. SDR Hardware Drivers

#### UHD (USRP Hardware Driver)
**Purpose:** Ettus USRP support  
**Version:** 4.5.0+  
**Installation:**
```bash
sudo apt-get install libuhd-dev uhd-host
sudo uhd_images_downloader
```

**Supported Devices:**
- USRP B200/B210 (70 MHz - 6 GHz) - Recommended
- USRP B200mini
- USRP X310

#### BladeRF Libraries
**Purpose:** Nuand bladeRF support  
**Installation:**
```bash
sudo apt-get install bladerf libbladerf-dev
```

**Supported Devices:**
- bladeRF x40/x115
- bladeRF 2.0 micro

---

### 5. Python Exploit Stack

#### Core Libraries
```bash
pip install scapy>=2.5.0          # Packet crafting (NAS, S1AP, NGAP, GTP)
pip install pyshark>=0.6.0         # Packet parsing
pip install pycrate>=0.6.0         # 3GPP protocol encoding/decoding
pip install pycryptodome>=3.19.0   # NAS encryption/integrity
```

#### Protocol Libraries
```bash
pip install sctp>=0.23.0           # SCTP transport (S1AP, NGAP)
pip install pyasn1>=0.5.0          # ASN.1 encoding/decoding
```

**Used For:**
- Payload generation for all 25 exploits
- NAS/S1AP/NGAP/GTP packet crafting
- Exploit chain automation

---

## Hardware Requirements

### Minimum Configuration
- **CPU:** Intel i5/AMD Ryzen 5 (4 cores)
- **RAM:** 8 GB
- **Storage:** 50 GB SSD
- **SDR:** HackRF One or RTL-SDR
- **OS:** Ubuntu 20.04 LTS

### Recommended Configuration (For Full Exploit Testing)
- **CPU:** Intel i7/AMD Ryzen 7 (8 cores)
- **RAM:** 16 GB
- **Storage:** 100 GB NVMe SSD
- **SDR:** USRP B210 or bladeRF 2.0 micro
- **OS:** Ubuntu 22.04 LTS or Kali Linux 2024

---

## Installation Order

```bash
# 1. System packages
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-pip \
    libusb-1.0-0-dev libboost-all-dev libfftw3-dev

# 2. SDR drivers
sudo apt-get install libuhd-dev uhd-host libbladerf-dev

# 3. GNU Radio
sudo apt-get install gnuradio gr-osmosdr

# 4. GSM tools
sudo apt-get install gr-gsm kalibrate-rtl

# 5. Cellular cores
sudo apt-get install open5gs  # Primary target (14 CVEs)

# 6. LTESniffer (from source)
git clone https://github.com/SysSec-KAIST/LTESniffer.git
cd LTESniffer && mkdir build && cd build
cmake .. && make -j$(nproc) && sudo make install

# 7. Python exploit dependencies
pip3 install scapy pyshark pycrate pycryptodome sctp pyasn1

# 8. FalconOne
cd FalconOne
pip3 install -r requirements.txt
python3 install_dependencies.py
```

---

## Verification Commands

```bash
# Check SDR hardware
uhd_find_devices          # USRP devices
bladeRF-cli -p            # bladeRF devices

# Check GNU Radio
gnuradio-config-info --version

# Check gr-gsm
grgsm_livemon --help

# Check kalibrate
kal -s GSM900 -g 40

# Check LTESniffer
LTESniffer --help

# Check Open5GS
open5gs-amfd --version

# Check Python dependencies
python3 -c "from scapy.all import *; print('Scapy OK')"
python3 -c "from pycrate_mobile.TS24301_NAS import *; print('Pycrate OK')"
```

---

## Exploit-Specific Dependencies

### For CVE-2024-24428 (Zero-Length NAS DoS)
- Open5GS 2.7.0 or earlier
- Scapy for NAS packet crafting
- Network access to AMF (port 38412)

### For FALCON-001 (Rogue Base Station)
- USRP B210 or bladeRF
- GNU Radio with gr-gsm
- kalibrate-rtl for frequency calibration

### For FALCON-003 (SMS Interception)
- gr-gsm for GSM monitoring
- OsmocomBB for protocol manipulation
- RTL-SDR minimum (USRP recommended)

---

## Troubleshooting

### Issue: SDR not detected
```bash
sudo usermod -a -G plugdev $USER
sudo udevadm control --reload-rules
# Logout and login again
```

### Issue: GNU Radio import errors
```bash
sudo apt-get install --reinstall gnuradio python3-gnuradio
```

### Issue: Open5GS fails to start
```bash
sudo systemctl status open5gs-amfd
sudo journalctl -u open5gs-amfd -n 50
```

---

## Legal Notice

⚠️ **WARNING:** These tools enable penetration testing of cellular networks.

**Legal Requirements:**
- ✅ Use only on authorized networks
- ✅ Obtain written permission before testing
- ✅ Follow responsible disclosure for discovered vulnerabilities
- ❌ Never test production networks without authorization

Unauthorized interception of cellular communications is **illegal** in most jurisdictions.

---

## Support

- **Documentation:** https://docs.falconone.io
- **GitHub Issues:** https://github.com/falconone/falconone/issues
- **Security Contact:** security@falconone.io

---

*Last Updated: January 1, 2026*  
*Document Version: 1.8.0*
