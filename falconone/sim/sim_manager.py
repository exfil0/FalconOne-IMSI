"""
FalconOne SIM Card Management Module
pySim integration for SIM card programming
"""

from typing import Dict, List, Any, Optional
import logging
import subprocess
import json

from ..utils.logger import ModuleLogger


class SIMManager:
    """SIM card programming and management using pySim"""
    
    def __init__(self, config, logger: logging.Logger):
        """Initialize SIM manager"""
        self.config = config
        self.logger = ModuleLogger('SIMManager', logger)
        
        self.reader_device = config.get('sim.reader_device', '/dev/ttyUSB0')
        self.adm_key = config.get('sim.adm_key')
        
        self.logger.info("SIM manager initialized", device=self.reader_device)
    
    def program_sim(self, sim_params: Dict[str, Any]) -> bool:
        """
        Program SIM card with specified parameters
        
        Args:
            sim_params: Dictionary with IMSI, Ki, OPc, etc.
            
        Returns:
            True if successful
        """
        try:
            imsi = sim_params.get('imsi')
            ki = sim_params.get('ki')
            opc = sim_params.get('opc')
            iccid = sim_params.get('iccid')
            
            if not all([imsi, ki, opc]):
                self.logger.error("Missing required SIM parameters")
                return False
            
            self.logger.info(f"Programming SIM with IMSI: {imsi}")
            
            # Use pySim-prog tool
            cmd = [
                'pySim-prog.py',
                '-p', '0',  # PC/SC reader
                '-t', 'sysmoUSIM-SJS1',  # Card type
                '--imsi', imsi,
                '--ki', ki,
                '--opc', opc
            ]
            
            if iccid:
                cmd.extend(['--iccid', iccid])
            
            if self.adm_key:
                cmd.extend(['--adm', self.adm_key])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.logger.info("SIM programming successful")
                return True
            else:
                self.logger.error(f"SIM programming failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            self.logger.error("pySim not found. Install from: https://github.com/osmocom/pysim")
            return False
        except Exception as e:
            self.logger.error(f"SIM programming error: {e}")
            return False
    
    def read_sim(self) -> Optional[Dict[str, Any]]:
        """
        Read SIM card information
        
        Returns:
            Dictionary with SIM data
        """
        try:
            cmd = [
                'pySim-read.py',
                '-p', '0'  # PC/SC reader
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse output
                sim_data = self._parse_pysim_output(result.stdout)
                self.logger.info("SIM read successful")
                return sim_data
            else:
                self.logger.error(f"SIM read failed: {result.stderr}")
                return None
                
        except FileNotFoundError:
            self.logger.error("pySim not found")
            return None
        except Exception as e:
            self.logger.error(f"SIM read error: {e}")
            return None
    
    def _parse_pysim_output(self, output: str) -> Dict[str, Any]:
        """Parse pySim tool output"""
        sim_data = {}
        
        for line in output.split('\\n'):
            if 'IMSI:' in line:
                sim_data['imsi'] = line.split(':')[1].strip()
            elif 'ICCID:' in line:
                sim_data['iccid'] = line.split(':')[1].strip()
            elif 'MCC/MNC:' in line:
                mcc_mnc = line.split(':')[1].strip()
                sim_data['mcc_mnc'] = mcc_mnc
        
        return sim_data
    
    def clone_sim(self, source_imsi: str, target_params: Dict[str, Any]) -> bool:
        """
        Clone SIM card (for authorized testing only)
        
        Args:
            source_imsi: IMSI to clone
            target_params: Target SIM parameters
            
        Returns:
            True if successful
        """
        try:
            # Read source SIM
            source_data = self.read_sim()
            
            if not source_data:
                self.logger.error("Failed to read source SIM")
                return False
            
            # Merge with target parameters
            clone_params = {**source_data, **target_params}
            
            # Program target SIM
            return self.program_sim(clone_params)
            
        except Exception as e:
            self.logger.error(f"SIM cloning error: {e}")
            return False
    
    def generate_test_credentials(self, mcc: str, mnc: str) -> Dict[str, Any]:
        """
        Generate test SIM credentials
        
        Args:
            mcc: Mobile Country Code
            mnc: Mobile Network Code
            
        Returns:
            Dictionary with test credentials
        """
        import random
        
        # Generate random IMSI
        msin = ''.join([str(random.randint(0, 9)) for _ in range(10)])
        imsi = f"{mcc}{mnc}{msin}"
        
        # Generate random Ki (128-bit key)
        ki = ''.join([f'{random.randint(0, 255):02x}' for _ in range(16)])
        
        # Generate random OPc
        opc = ''.join([f'{random.randint(0, 255):02x}' for _ in range(16)])
        
        # Generate ICCID
        iccid = f"89{mcc}{mnc}" + ''.join([str(random.randint(0, 9)) for _ in range(12)])
        
        credentials = {
            'imsi': imsi,
            'ki': ki,
            'opc': opc,
            'iccid': iccid,
            'mcc': mcc,
            'mnc': mnc
        }
        
        self.logger.info(f"Generated test credentials for {mcc}-{mnc}")
        
        return credentials
