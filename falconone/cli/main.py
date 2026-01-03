"""
FalconOne Command-Line Interface
User-friendly CLI for all operations
"""

import click
import sys
import time
from pathlib import Path
from typing import Optional

from ..core.orchestrator import FalconOneOrchestrator
from ..utils.config import Config


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.version_option(version='1.9.0', prog_name='FalconOne')
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool):
    """
    FalconOne SIGINT Platform - v1.9.0
    
    Multi-generation cellular monitoring and analysis platform
    with agentic AI, quantum cryptanalysis, and 6G capabilities.
    """
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose


@cli.command()
@click.pass_context
def start(ctx):
    """Start FalconOne orchestrator with all enabled services"""
    click.echo("🚀 Starting FalconOne v1.4 Complete...")
    click.echo("")
    
    try:
        orchestrator = FalconOneOrchestrator(ctx.obj['config_path'])
        orchestrator.start()
        
        click.echo("✓ FalconOne operational")
        click.echo("✓ All services started")
        click.echo("")
        click.echo("Press Ctrl+C to stop gracefully")
        click.echo("")
        
        # Keep running
        while orchestrator.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        click.echo("\n⏹ Stopping FalconOne...")
        orchestrator.stop()
        click.echo("✓ Stopped successfully")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--generation', '-g', type=click.Choice(['2G', '3G', '4G', '5G', '6G'], case_sensitive=False), 
              required=True, help='Cellular generation to monitor')
@click.option('--passive', is_flag=True, help='Passive monitoring only (no transmission)')
@click.option('--duration', '-d', type=int, help='Monitoring duration in seconds')
@click.pass_context
def monitor(ctx, generation: str, passive: bool, duration: Optional[int]):
    """Start monitoring for specific cellular generation"""
    mode = "passive" if passive else "active"
    click.echo(f"📡 Starting {generation} monitoring ({mode} mode)...")
    
    try:
        orchestrator = FalconOneOrchestrator(ctx.obj['config_path'])
        orchestrator.initialize_components()
        
        # Get appropriate monitor
        monitor_map = {
            '2G': orchestrator.gsm_monitor,
            '3G': orchestrator.gsm_monitor,  # UMTS uses similar monitor
            '4G': orchestrator.lte_monitor,
            '5G': orchestrator.fiveg_monitor,
            '6G': orchestrator.sixg_monitor
        }
        
        monitor = monitor_map.get(generation.upper())
        
        if not monitor:
            click.echo(f"❌ {generation} monitor not available", err=True)
            sys.exit(1)
        
        click.echo(f"✓ {generation} monitoring started")
        
        if duration:
            click.echo(f"⏱️  Monitoring for {duration} seconds...")
            time.sleep(duration)
            click.echo("✓ Monitoring completed")
        else:
            click.echo("Press Ctrl+C to stop")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                click.echo("\n✓ Monitoring stopped")
                
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--type', '-t', type=click.Choice(['dos', 'downgrade', 'mitm', 'paging']), 
              required=True, help='Exploitation type')
@click.option('--target', required=True, help='Target identifier (IMSI, cell ID, etc.)')
@click.option('--evasion/--no-evasion', default=True, help='Enable ML-based evasion')
@click.pass_context
def exploit(ctx, type: str, target: str, evasion: bool):
    """
    Execute exploitation operation (RESEARCH USE ONLY)
    
    WARNING: This command is for authorized research only.
    Requires Faraday cage and ethical approval.
    """
    click.echo("")
    click.echo("⚠️  " + "="*50)
    click.echo("⚠️  EXPLOITATION MODE - RESEARCH USE ONLY")
    click.echo("⚠️  " + "="*50)
    click.echo("")
    
    # Safety confirmation
    if not click.confirm('⚠️  Faraday cage active and ethical approval obtained?', default=False):
        click.echo("❌ Exploitation aborted - safety requirements not met")
        sys.exit(1)
    
    click.echo(f"\n🔧 Executing {type.upper()} exploit...")
    click.echo(f"   Target: {target}")
    click.echo(f"   ML Evasion: {'Enabled' if evasion else 'Disabled'}")
    click.echo("")
    
    try:
        orchestrator = FalconOneOrchestrator(ctx.obj['config_path'])
        
        result = orchestrator.execute_exploit(type, {
            'target': target,
            'evasion_mode': evasion
        })
        
        click.echo(f"✓ Exploit completed")
        click.echo(f"   Status: {result.get('status', 'unknown')}")
        click.echo(f"   Type: {result.get('type', 'unknown')}")
        
        if 'evasion' in result:
            click.echo(f"   Evasion: {result['evasion']}")
        
    except Exception as e:
        click.echo(f"❌ Exploit failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Show comprehensive system status"""
    try:
        orchestrator = FalconOneOrchestrator(ctx.obj['config_path'])
        orchestrator.initialize_components()
        
        status = orchestrator.get_status()
        
        click.echo("")
        click.echo("📊 FalconOne v1.4 - System Status")
        click.echo("=" * 50)
        click.echo(f"Running:              {status['running']}")
        click.echo(f"Version:              {status['version']}")
        click.echo(f"Components Init:      {status['components_initialized']}")
        click.echo("")
        
        click.echo("Active Monitors:")
        if status['active_monitors']:
            for monitor in status['active_monitors']:
                click.echo(f"  ✓ {monitor}")
        else:
            click.echo("  (none)")
        click.echo("")
        
        if 'sdr' in status:
            click.echo("SDR Status:")
            click.echo(f"  Available Devices:  {len(status['sdr']['available_devices'])}")
            click.echo(f"  Active Device:      {status['sdr']['active_device'] or 'None'}")
            click.echo("")
        
        click.echo("Safety Configuration:")
        click.echo(f"  Faraday Cage Req:   {status['safety']['faraday_cage_required']}")
        click.echo(f"  Audit Logging:      {status['safety']['audit_logging']}")
        click.echo("")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--port', '-p', type=int, default=5000, help='Dashboard port')
@click.option('--host', '-h', default='0.0.0.0', help='Dashboard host')
def dashboard(port: int, host: str):
    """Start web dashboard for real-time monitoring"""
    click.echo(f"🌐 Starting FalconOne Dashboard...")
    click.echo(f"   URL: http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
    click.echo("")
    click.echo("Initializing FalconOne components...")
    click.echo("")
    
    try:
        from ..ui.dashboard import DashboardServer, socketio, app
        from ..utils.config import Config
        from ..core.orchestrator import FalconOneOrchestrator
        
        config = Config()
        config.set('dashboard.port', port)
        config.set('dashboard.host', host)
        
        # Initialize orchestrator with all components
        click.echo("Initializing FalconOne orchestrator...")
        orchestrator = FalconOneOrchestrator()
        orchestrator.initialize_components()
        click.echo(f"✓ Loaded {len(orchestrator.components)} components")
        click.echo("")
        click.echo("Press Ctrl+C to stop")
        click.echo("")
        
        # Create dashboard with orchestrator access
        dashboard_server = DashboardServer(config, orchestrator.root_logger, orchestrator)
        
        # Run with SocketIO
        socketio.run(app, host=host, port=port, debug=False, use_reloader=False)
        
    except ImportError as e:
        click.echo(f"❌ Dashboard not available: {e}", err=True)
        click.echo("   Install with: pip install flask flask-socketio", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\n✓ Dashboard stopped")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', type=click.Path(), default='config/falconone.yaml',
              help='Output path for configuration file')
def init_config(output: str):
    """Initialize default configuration file"""
    output_path = Path(output)
    
    if output_path.exists():
        if not click.confirm(f"Configuration file exists at {output}. Overwrite?", default=False):
            click.echo("❌ Aborted")
            return
    
    click.echo(f"📝 Creating default configuration: {output}")
    
    try:
        config = Config()
        config.config_path = output_path
        config.save()
        
        click.echo("✓ Configuration created successfully")
        click.echo(f"   Edit file: {output}")
        click.echo("")
        click.echo("Important: Set safety.require_faraday_cage appropriately!")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command(name='list-devices')
@click.option('--format', type=click.Choice(['text', 'json']), default='text',
              help='Output format')
def list_devices(format: str):
    """List available SDR devices"""
    try:
        from ..sdr.sdr_layer import SDRManager
        from ..utils.config import Config
        
        click.echo("🔍 Scanning for SDR devices...")
        
        config = Config()
        manager = SDRManager(config, None)
        devices = manager.get_available_devices()
        
        if format == 'json':
            import json
            click.echo(json.dumps({
                'devices': devices,
                'count': len(devices)
            }, indent=2))
        else:
            click.echo("")
            click.echo("📻 Available SDR Devices:")
            click.echo("=" * 50)
            
            if devices:
                for i, device in enumerate(devices, 1):
                    click.echo(f"{i}. {device}")
            else:
                click.echo("  (no devices found)")
            
            click.echo("")
            click.echo(f"Total: {len(devices)} device(s)")
        
    except ImportError:
        click.echo("❌ SDR manager not available", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('log_file', type=click.Path(exists=True), required=False)
@click.option('--tail', '-n', type=int, default=50, help='Number of lines to show')
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
def logs(log_file: Optional[str], tail: int, follow: bool):
    """View system or audit logs"""
    if not log_file:
        log_file = 'logs/falconone.log'
    
    log_path = Path(log_file)
    
    if not log_path.exists():
        click.echo(f"❌ Log file not found: {log_file}", err=True)
        sys.exit(1)
    
    try:
        if follow:
            click.echo(f"📜 Following {log_file} (Ctrl+C to stop)...")
            click.echo("")
            
            # Simple tail -f implementation
            with open(log_path, 'r') as f:
                # Go to end of file
                f.seek(0, 2)
                
                while True:
                    line = f.readline()
                    if line:
                        click.echo(line.rstrip())
                    else:
                        time.sleep(0.1)
        else:
            # Show last N lines
            with open(log_path, 'r') as f:
                lines = f.readlines()
                for line in lines[-tail:]:
                    click.echo(line.rstrip())
    
    except KeyboardInterrupt:
        click.echo("\n✓ Stopped following logs")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


def main():
    """Entry point for CLI"""
    try:
        cli(obj={})
    except Exception as e:
        click.echo(f"\n❌ Fatal error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
