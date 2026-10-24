"""Install one new task-owned Supervisor job for a reviewed experiment command."""
import argparse
from pathlib import Path
import shlex
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--program',required=True)
parser.add_argument('--log',type=Path,required=True)
parser.add_argument('--config-dir',type=Path,default=Path('/etc/supervisor/conf.d'))
parser.add_argument('command',nargs=argparse.REMAINDER)
args=parser.parse_args()
assert args.program and all(c.isalnum() or c in '-_' for c in args.program)
command=args.command[1:] if args.command and args.command[0]=='--' else args.command
assert command and Path(command[0]).is_absolute()
assert not args.log.exists(),'Preserve existing task logs'
config=args.config_dir/(args.program+'.conf')
assert not config.exists(),'Never overwrite an existing service'
args.log.parent.mkdir(parents=True,exist_ok=True)
config.write_text(f'[program:{args.program}]\ncommand={shlex.join(command).replace("%","%%")}\n'
                  'autostart=true\nautorestart=false\nstartsecs=2\nstopasgroup=true\nkillasgroup=true\n'
                  f'stdout_logfile={args.log}\nredirect_stderr=true\nstdout_logfile_maxbytes=20MB\n'
                  'stdout_logfile_backups=2\n')
subprocess.run(['supervisorctl','reread'],check=True)
subprocess.run(['supervisorctl','update',args.program],check=True)
subprocess.run(['supervisorctl','status',args.program],check=True)
