"""
InfernoCLI - A modern command-line interface for Inferno

A powerful, feature-rich CLI framework for the Inferno inference server.
Built with a focus on usability and developer experience.

Basic Usage:
    >>> from inferno.cli import app
    >>> app.run()

Advanced Usage:
    >>> from inferno.cli import app
    >>> @app.command()
    ... def custom_command():
    ...     '''My custom command'''
    ...     print("Running custom command")
"""

import sys
import os
import inspect
from typing import Any, Dict, List, Optional, Callable, Union, Type
from functools import wraps
from pathlib import Path

# Import version from package
from inferno import __version__
# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from inferno.utils.logger import get_logger, InfernoLogger
from inferno.config.server_config import ServerConfig

# Set up console for rich output
if RICH_AVAILABLE:
    console = Console()
else:
    # Fallback console implementation
    class SimpleConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = SimpleConsole()

logger = get_logger(__name__)


# Exception classes
class InfernoCliError(Exception):
    """Base exception for InfernoCLI errors"""
    pass

class CommandError(InfernoCliError):
    """Raised when a command encounters an error"""
    pass

class UsageError(InfernoCliError):
    """Raised when the CLI is used incorrectly"""
    pass

class ParameterError(UsageError):
    """Raised when a parameter is invalid"""
    pass


class Command:
    """
    Represents a CLI command with its arguments and options.

    Attributes:
        name (str): The name of the command
        help (str): Help text for the command
        function (Callable): The function to execute
        options (List): Command options
        arguments (List): Command arguments
    """
    def __init__(self, name: str, function: Callable, help: str = None):
        self.name = name
        self.function = function
        self.help = help or function.__doc__
        self.options = getattr(function, '_options', [])
        self.arguments = getattr(function, '_arguments', [])
        self.parent = None

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class Group:
    """
    Command group that can contain subcommands.

    Attributes:
        name (str): The name of the group
        help (str): Help text for the group
        commands (Dict): Subcommands in this group
        parent (Group): Parent group if any
    """
    def __init__(self, name: str, help: str = None):
        self.name = name
        self.help = help
        self.commands = {}
        self.parent = None

    def command(self, name: str = None, help: str = None):
        """Register a new command in this group"""
        def decorator(f):
            cmd_name = name or f.__name__
            cmd = Command(cmd_name, f, help)
            cmd.parent = self
            self.commands[cmd_name] = cmd
            return f
        return decorator

    def group(self, name: str = None, help: str = None):
        """Create a subgroup in this group"""
        def decorator(f):
            grp_name = name or f.__name__
            grp = Group(grp_name, help or f.__doc__)
            grp.parent = self
            self.commands[grp_name] = grp
            return grp
        return decorator

    def _print_help(self):
        """Print help for this group"""
        console.print(f"\n[bold]{self.name}[/bold] - {self.help or ''}")

        if self.commands:
            console.print("\n[bold]Commands:[/bold]")
            for name, cmd in self.commands.items():
                if isinstance(cmd, Command):
                    console.print(f"  {self.name} {name:20} {cmd.help or ''}")

        if any(isinstance(cmd, Group) for cmd in self.commands.values()):
            console.print("\n[bold]Subgroups:[/bold]")
            for name, cmd in self.commands.items():
                if isinstance(cmd, Group):
                    console.print(f"  {self.name} {name:20} {cmd.help or ''}")

        console.print("\nUse -h or --help with any command for more information")

class InfernoCLI:
    """
    The main CLI application class for Inferno.

    Attributes:
        name (str): The name of the CLI application
        version (str): The version of the application
        help (str): Help text for the application
        commands (Dict): Top-level commands
        groups (Dict): Command groups
    """
    def __init__(self, name: str = "inferno", version: str = None, help: str = None):
        self.name = name
        self.version = version or "0.1.0"
        self.help = help or "Inferno: A professional inference server for HelpingAI models"
        self.commands = {}
        self.groups = {}

    def command(self, name: str = None, help: str = None):
        """Register a new top-level command"""
        def decorator(f):
            cmd_name = name or f.__name__
            cmd = Command(cmd_name, f, help)
            self.commands[cmd_name] = cmd
            return f
        return decorator

    def group(self, name: str = None, help: str = None):
        """Create a command group"""
        def decorator(f):
            grp_name = name or f.__name__
            grp = Group(grp_name, help or f.__doc__)
            self.groups[grp_name] = grp
            return grp
        return decorator

    def _parse_args(self, args: List[str]) -> Dict[str, Any]:
        """Parse command line arguments"""
        if not args:
            self._print_help()
            return {}

        command_name = args[0]
        command_args = args[1:]

        # Check for help flag
        if command_name in ["-h", "--help"]:
            self._print_help()
            return {}

        # Check for version flag
        if command_name in ["-v", "--version"]:
            console.print(f"{self.name} version {self.version}")
            return {}

        # Check if it's a group command
        if command_name in self.groups:
            group = self.groups[command_name]
            if not command_args:
                group._print_help()
                return {}

            subcommand = command_args[0]
            if subcommand in ["-h", "--help"]:
                group._print_help()
                return {}

            # Check if it's a help request for a specific subcommand
            if len(command_args) > 1 and command_args[1] in ["-h", "--help"]:
                if subcommand in group.commands:
                    command = group.commands[subcommand]
                    console.print(f"\n[bold]{group.name} {command.name}[/bold] - {command.help or ''}")
                    # Print command usage if available
                    if hasattr(command.function, '_arguments') or hasattr(command.function, '_options'):
                        console.print("\n[bold]Usage:[/bold]")
                        usage = f"  {group.name} {command.name}"

                        # Add arguments to usage
                        if hasattr(command.function, '_arguments'):
                            for arg in command.function._arguments:
                                arg_str = f"<{arg['name']}>"
                                if not arg.get('required', True):
                                    arg_str = f"[{arg_str}]"
                                usage += f" {arg_str}"

                        console.print(usage)

                        # Print arguments
                        if hasattr(command.function, '_arguments'):
                            console.print("\n[bold]Arguments:[/bold]")
                            for arg in command.function._arguments:
                                required = "(required)" if arg.get('required', True) else "(optional)"
                                console.print(f"  {arg['name']:20} {arg.get('help', '')} {required}")

                        # Print options
                        if hasattr(command.function, '_options'):
                            console.print("\n[bold]Options:[/bold]")
                            for opt in command.function._options:
                                param_decls = ", ".join(opt['param_decls'])
                                default = f"(default: {opt['default']})" if 'default' in opt and opt['default'] is not None else ""
                                required = "(required)" if opt.get('required', False) else ""
                                console.print(f"  {param_decls:20} {opt.get('help', '')} {default} {required}")
                    return {}

            if subcommand not in group.commands:
                console.print(f"[red]Unknown command: {command_name} {subcommand}[/red]")
                group._print_help()
                return {}

            command = group.commands[subcommand]
            return self._parse_command_args(command, command_args[1:])

        # Check if it's a direct command
        if command_name not in self.commands:
            console.print(f"[red]Unknown command: {command_name}[/red]")
            self._print_help()
            return {}

        command = self.commands[command_name]
        return self._parse_command_args(command, command_args)

    def _parse_command_args(self, command: Command, args: List[str]) -> Dict[str, Any]:
        """Parse arguments for a specific command"""
        params = {}

        # Get function signature
        sig = inspect.signature(command.function)

        # Process options
        if hasattr(command.function, '_options'):
            for opt in command.function._options:
                param_decls = sorted(opt['param_decls'], key=len, reverse=True)
                param_name = param_decls[0].lstrip('-').replace('-', '_')

                # Find matching parameter in signature
                for param in sig.parameters.values():
                    if param.name in [d.lstrip('-').replace('-', '_') for d in param_decls]:
                        param_name = param.name
                        break

                # Look for the option in args
                found = False
                i = 0
                while i < len(args):
                    if args[i] in param_decls:
                        if opt.get('is_flag', False):
                            params[param_name] = True
                            args.pop(i)
                            found = True
                            break
                        else:
                            if i + 1 < len(args):
                                value = args[i + 1]
                                # Convert value to the correct type
                                if 'type' in opt:
                                    try:
                                        value = opt['type'](value)
                                    except ValueError:
                                        raise ParameterError(f"Invalid value for {args[i]}: {value}")

                                params[param_name] = value
                                args.pop(i)
                                args.pop(i)  # Remove the value too
                                found = True
                                break
                            else:
                                raise UsageError(f"Option {args[i]} requires a value")
                    else:
                        i += 1

                # Handle required options
                if not found and opt.get('required', False):
                    raise UsageError(f"Option {param_decls[0]} is required")

                # Set default value if not found
                if not found and 'default' in opt:
                    params[param_name] = opt['default']

        # Process positional arguments
        if hasattr(command.function, '_arguments'):
            for i, arg in enumerate(command.function._arguments):
                if i < len(args):
                    value = args[i]
                    # Convert value to the correct type
                    if 'type' in arg:
                        try:
                            value = arg['type'](value)
                        except ValueError:
                            raise ParameterError(f"Invalid value for {arg['name']}: {value}")
                    params[arg['name']] = value
                elif arg.get('required', True):
                    raise UsageError(f"Argument {arg['name']} is required")
                elif 'default' in arg:
                    params[arg['name']] = arg['default']

        return params

    def _print_help(self):
        """Print main help message"""
        console.print(f"\n[bold]{self.name}[/bold] - {self.help}")
        console.print(f"Version: {self.version}\n")

        if self.commands:
            console.print("[bold]Commands:[/bold]")
            for name, cmd in self.commands.items():
                console.print(f"  {name:20} {cmd.help or ''}")

        if self.groups:
            console.print("\n[bold]Command Groups:[/bold]")
            for name, group in self.groups.items():
                console.print(f"  {name:20} {group.help or ''}")

        console.print("\nUse -h or --help with any command for more information")

    def _print_command_help(self, command, group_name=None):
        """Print help for a specific command"""
        prefix = f"{group_name} " if group_name else ""
        console.print(f"\n[bold]{prefix}{command.name}[/bold] - {command.help or ''}")

        # Print command usage if available
        if hasattr(command.function, '_arguments') or hasattr(command.function, '_options'):
            console.print("\n[bold]Usage:[/bold]")
            usage = f"  {prefix}{command.name}"

            # Add arguments to usage
            if hasattr(command.function, '_arguments'):
                for arg in command.function._arguments:
                    arg_str = f"<{arg['name']}>"
                    if not arg.get('required', True):
                        arg_str = f"[{arg_str}]"
                    usage += f" {arg_str}"

            console.print(usage)

            # Print arguments
            if hasattr(command.function, '_arguments'):
                console.print("\n[bold]Arguments:[/bold]")
                for arg in command.function._arguments:
                    required = "(required)" if arg.get('required', True) else "(optional)"
                    console.print(f"  {arg['name']:20} {arg.get('help', '')} {required}")

            # Print options
            if hasattr(command.function, '_options'):
                console.print("\n[bold]Options:[/bold]")
                for opt in command.function._options:
                    param_decls = ", ".join(opt['param_decls'])
                    default = f"(default: {opt['default']})" if 'default' in opt and opt['default'] is not None else ""
                    required = "(required)" if opt.get('required', False) else ""
                    console.print(f"  {param_decls:20} {opt.get('help', '')} {default} {required}")

    def run(self, args: List[str] = None):
        """Run the CLI application"""
        if args is None:
            args = sys.argv[1:]

        try:
            # Parse arguments
            if not args:
                self._print_help()
                return 0

            command_name = args[0]

            # Handle help and version flags
            if command_name in ["-h", "--help"]:
                self._print_help()
                return 0

            if command_name in ["-v", "--version"]:
                console.print(f"{self.name} version {self.version}")
                return 0

            # Handle group commands
            if command_name in self.groups:
                group = self.groups[command_name]
                if len(args) == 1:
                    group._print_help()
                    return 0

                subcommand = args[1]
                if subcommand in ["-h", "--help"]:
                    group._print_help()
                    return 0

                # Check if it's a help request for a specific subcommand
                if len(args) > 2 and args[2] in ["-h", "--help"]:
                    if subcommand in group.commands:
                        command = group.commands[subcommand]
                        self._print_command_help(command, group_name=group.name)
                        return 0

                if subcommand not in group.commands:
                    console.print(f"[red]Unknown command: {command_name} {subcommand}[/red]")
                    group._print_help()
                    return 1

                command = group.commands[subcommand]
                params = self._parse_command_args(command, args[2:])
                command.function(**params)
                return 0

            # Handle direct commands
            if command_name not in self.commands:
                console.print(f"[red]Unknown command: {command_name}[/red]")
                self._print_help()
                return 1

            command = self.commands[command_name]

            # Check if it's a help request for a direct command
            if len(args) > 1 and args[1] in ["-h", "--help"]:
                self._print_command_help(command)
                return 0

            params = self._parse_command_args(command, args[1:])
            command.function(**params)
            return 0

        except UsageError as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            return 1
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            import traceback
            console.print(traceback.format_exc())
            return 1


# Decorator functions
def option(*param_decls, **attrs):
    """Decorator to add an option to a command"""
    def decorator(f):
        if not hasattr(f, '_options'):
            f._options = []

        # Set default values
        attrs.setdefault('type', str)
        attrs.setdefault('required', False)
        attrs.setdefault('default', None)
        attrs.setdefault('help', None)
        attrs.setdefault('is_flag', False)

        f._options.append({
            'param_decls': param_decls,
            **attrs
        })
        return f
    return decorator


def argument(name, **attrs):
    """Decorator to add a positional argument to a command"""
    def decorator(f):
        if not hasattr(f, '_arguments'):
            f._arguments = []
        f._arguments.append({
            'name': name,
            **attrs
        })
        return f
    return decorator


# Create the CLI application
app = InfernoCLI(
    name="inferno",
    version=__version__,
    help="A professional inference server for HelpingAI models"
)


# Define model group
@app.group()
def model():
    """Model management commands"""
    pass


@model.command()
@argument("model_path", help="Path to the model to download")
@option("--revision", "-r", help="Specific model revision to download")
@option("--force", "-f", help="Force re-download even if model exists", is_flag=True)
def download(model_path, revision=None, force=False):
    """Download a model from Hugging Face"""
    # Skip execution if model_path is a help flag (this is a workaround for the help system)
    if model_path in ["-h", "--help"]:
        return None

    try:
        from huggingface_hub import snapshot_download
        import os

        console.print(f"Downloading model: {model_path}")
        if revision:
            console.print(f"Revision: {revision}")

        # Create a progress context
        if RICH_AVAILABLE:
            from rich.progress import Progress
            with Progress() as progress:
                task = progress.add_task(f"Downloading {model_path}...", total=None)

                # Download the model
                path = snapshot_download(
                    repo_id=model_path,
                    revision=revision,
                    force_download=force
                )

                progress.update(task, completed=True)
        else:
            # Fallback without rich progress bar
            path = snapshot_download(
                repo_id=model_path,
                revision=revision,
                force_download=force
            )

        console.print(f"[green]Model downloaded successfully to: {path}[/green]")
        return path
    except Exception as e:
        console.print(f"[red]Error downloading model: {str(e)}[/red]")
        return None


@model.command()
@argument("model_path", help="Path to the model to check")
def info(model_path):
    """Show information about a model"""
    try:
        from transformers import AutoConfig
        import os

        console.print(f"Loading model info: {model_path}")

        # Check if it's a local path or Hugging Face model
        is_local = os.path.exists(model_path)
        if is_local:
            console.print(f"Model type: Local model at {model_path}")
        else:
            console.print(f"Model type: Hugging Face model {model_path}")

        # Get model configuration
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # Display model information in a table
        if RICH_AVAILABLE:
            table = Table(title=f"Model Information: {model_path}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            # Add model properties
            table.add_row("Model Type", config.__class__.__name__)
            table.add_row("Architecture", getattr(config, 'architectures', ['Unknown'])[0])
            table.add_row("Hidden Size", str(getattr(config, 'hidden_size', 'Unknown')))
            table.add_row("Num Layers", str(getattr(config, 'num_hidden_layers', 'Unknown')))
            table.add_row("Num Attention Heads", str(getattr(config, 'num_attention_heads', 'Unknown')))
            table.add_row("Vocab Size", str(getattr(config, 'vocab_size', 'Unknown')))

            # Add any other interesting properties
            for key, value in config.__dict__.items():
                if key not in ['_name_or_path', 'architectures', 'hidden_size', 'num_hidden_layers',
                              'num_attention_heads', 'vocab_size'] and not key.startswith('_'):
                    if isinstance(value, (str, int, float, bool)):
                        table.add_row(key, str(value))

            console.print(table)
        else:
            # Fallback without rich tables
            print(f"Model Information: {model_path}")
            print(f"Model Type: {config.__class__.__name__}")
            print(f"Architecture: {getattr(config, 'architectures', ['Unknown'])[0]}")
            print(f"Hidden Size: {getattr(config, 'hidden_size', 'Unknown')}")
            print(f"Num Layers: {getattr(config, 'num_hidden_layers', 'Unknown')}")
            print(f"Num Attention Heads: {getattr(config, 'num_attention_heads', 'Unknown')}")
            print(f"Vocab Size: {getattr(config, 'vocab_size', 'Unknown')}")

            # Add any other interesting properties
            for key, value in config.__dict__.items():
                if key not in ['_name_or_path', 'architectures', 'hidden_size', 'num_hidden_layers',
                              'num_attention_heads', 'vocab_size'] and not key.startswith('_'):
                    if isinstance(value, (str, int, float, bool)):
                        print(f"{key}: {value}")
    except Exception as e:
        console.print(f"[red]Error getting model info: {str(e)}[/red]")
        return None


# Define server command
@app.command()
@option("--model", "-m", help="Path to HuggingFace model or local model directory", default="HelpingAI/HelpingAI-15B")
@option("--model-revision", help="Specific model revision to load")
@option("--tokenizer", help="Path to tokenizer (defaults to model path)")
@option("--tokenizer-revision", help="Specific tokenizer revision to load")
@option("--device", "-d", help="Device to load the model on", default="auto")
@option("--device-map", help="Device map for model distribution", default="auto")
@option("--cuda-device-idx", help="CUDA device index to use", type=int, default=0)
@option("--dtype", help="Data type for model weights", default="float16")
@option("--load-8bit", help="Load model in 8-bit precision", is_flag=True)
@option("--load-4bit", help="Load model in 4-bit precision", is_flag=True)
@option("--host", help="Host to bind the server to", default="0.0.0.0")
@option("--port", "-p", help="Port to bind the server to", type=int, default=8000)
@option("--api-keys", help="Comma-separated list of valid API keys")
@option("--max-concurrent", help="Maximum number of concurrent requests", type=int, default=10)
@option("--max-queue", help="Maximum queue size for pending requests", type=int, default=100)
@option("--timeout", help="Timeout for requests in seconds", type=int, default=60)
@option("--log-level", help="Log level", default="info")
@option("--log-file", help="Log file path (logs to console if not specified)")
@option("--additional-models", help="Additional models to load (comma-separated list)")
@option("--use-tpu", help="Enable TPU support (requires torch_xla)", is_flag=True)
@option("--force-tpu", help="Force TPU usage even if not detected automatically", is_flag=True)
@option("--tpu-cores", help="Number of TPU cores to use", type=int, default=8)
@option("--tpu-memory-limit", help="Memory limit for TPU", default="90GB")
@option("--enable-gguf", help="Enable GGUF model support", is_flag=True)
@option("--gguf-path", help="Path to GGUF model file")
@option("--download-gguf", help="Download GGUF model from Hugging Face", is_flag=True)
@option("--gguf-filename", help="Specific GGUF filename to download")
@option("--num-gpu-layers", help="Number of GPU layers for GGUF models", type=int, default=-1)
@option("--context-size", help="Context size for GGUF models (in tokens)", type=int, default=4096)
@option("--chat-format", help="Chat format for GGUF models (llama-2, mistral, gemma, phi, chatml)")
def server(**kwargs):
    """Start the Inferno server"""
    # Set up logging
    log_level = kwargs.get('log_level', 'info')
    log_file = kwargs.get('log_file')

    # Configure the logger
    global logger
    logger = InfernoLogger("inferno", level=log_level, log_file=log_file)

    # Log the configuration
    logger.info("Starting Inferno server with the following configuration:")
    for key, value in kwargs.items():
        logger.info(f"  {key}: {value}")

    # Process additional models
    additional_models = []
    if kwargs.get('additional_models'):
        additional_models = [m.strip() for m in kwargs['additional_models'].split(',')]
        kwargs['additional_models'] = additional_models

    # Process API keys
    api_keys = []
    if kwargs.get('api_keys'):
        api_keys = [k.strip() for k in kwargs['api_keys'].split(',')]
        kwargs['api_keys'] = api_keys

    # Create server configuration
    config = ServerConfig.from_dict({
        "model": {
            "name_or_path": kwargs.get('model', "HelpingAI/HelpingAI-15B"),
            "revision": kwargs.get('model_revision'),
            "tokenizer": kwargs.get('tokenizer'),
            "tokenizer_revision": kwargs.get('tokenizer_revision')
        },
        "hardware": {
            "device": kwargs.get('device', "auto"),
            "device_map": kwargs.get('device_map', "auto"),
            "cuda_device_idx": kwargs.get('cuda_device_idx', 0),
            "dtype": kwargs.get('dtype', "float16"),
            "load_8bit": kwargs.get('load_8bit', False),
            "load_4bit": kwargs.get('load_4bit', False)
        },
        "tpu": {
            "use_tpu": kwargs.get('use_tpu', False) or kwargs.get('force_tpu', False),
            "tpu_cores": kwargs.get('tpu_cores', 8),
            "tpu_memory_limit": kwargs.get('tpu_memory_limit', "90GB")
        },
        "gguf": {
            "enable_gguf": kwargs.get('enable_gguf', False),
            "gguf_path": kwargs.get('gguf_path'),
            "download_gguf": kwargs.get('download_gguf', False),
            "gguf_filename": kwargs.get('gguf_filename'),
            "num_gpu_layers": kwargs.get('num_gpu_layers', -1),
            "context_size": kwargs.get('context_size', 4096),
            "chat_format": kwargs.get('chat_format')
        },
        "server": {
            "host": kwargs.get('host', "0.0.0.0"),
            "port": kwargs.get('port', 8000),
            "api_keys": api_keys,
            "max_concurrent_requests": kwargs.get('max_concurrent', 10),
            "max_queue_size": kwargs.get('max_queue', 100),
            "request_timeout": kwargs.get('timeout', 60),
            "log_level": log_level,
            "log_file": log_file
        },
        "additional_models": additional_models
    })

    # Import the server module here to avoid circular imports
    from inferno.main import run_server

    # Run the server
    run_server(config)


# Define version command
@app.command()
def version():
    """Show the Inferno version"""
    console.print(f"Inferno version {__version__}")


# Define info command
@app.command()
def info():
    """Show system information"""
    import platform
    import torch
    import psutil
    from inferno.utils.device import XLA_AVAILABLE, get_device_info, XLA

    # Create a table for system info
    if RICH_AVAILABLE:
        table = Table(title="System Information")
        table.add_column("Component", style="cyan")
        table.add_column("Details", style="green")

        # System info
        table.add_row("OS", f"{platform.system()} {platform.release()}")
        table.add_row("Python", platform.python_version())
        table.add_row("CPU", platform.processor() or "Unknown")
        table.add_row("CPU Cores", str(psutil.cpu_count(logical=False)))
        table.add_row("Logical CPUs", str(psutil.cpu_count(logical=True)))

        # Memory info
        mem = psutil.virtual_memory()
        table.add_row("Total Memory", f"{mem.total / (1024**3):.2f} GB")
        table.add_row("Available Memory", f"{mem.available / (1024**3):.2f} GB")

        # PyTorch info
        table.add_row("PyTorch Version", torch.__version__)
        table.add_row("CUDA Available", str(torch.cuda.is_available()))

        if torch.cuda.is_available():
            table.add_row("CUDA Version", torch.version.cuda)
            table.add_row("GPU Count", str(torch.cuda.device_count()))
            for i in range(torch.cuda.device_count()):
                table.add_row(f"GPU {i}", torch.cuda.get_device_name(i))

        # Add TPU information if available
        if XLA_AVAILABLE:
            table.add_row("TPU Available", "True")
            # Get detailed TPU information
            tpu_info = get_device_info(XLA)
            if "count" in tpu_info:
                table.add_row("TPU Device Count", str(tpu_info["count"]))
            if "version" in tpu_info:
                table.add_row("TPU Version", tpu_info["version"])
            if "topology" in tpu_info:
                table.add_row("TPU Topology", tpu_info["topology"])

        console.print(table)
    else:
        # Fallback for when rich is not available
        print("System Information:")
        print(f"OS: {platform.system()} {platform.release()}")
        print(f"Python: {platform.python_version()}")
        print(f"CPU: {platform.processor() or 'Unknown'}")
        print(f"CPU Cores: {psutil.cpu_count(logical=False)}")
        print(f"Logical CPUs: {psutil.cpu_count(logical=True)}")

        mem = psutil.virtual_memory()
        print(f"Total Memory: {mem.total / (1024**3):.2f} GB")
        print(f"Available Memory: {mem.available / (1024**3):.2f} GB")

        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        # Add TPU information if available
        if XLA_AVAILABLE:
            print("TPU Available: True")
            # Get detailed TPU information
            tpu_info = get_device_info(XLA)
            if "count" in tpu_info:
                print(f"TPU Device Count: {tpu_info['count']}")
            if "version" in tpu_info:
                print(f"TPU Version: {tpu_info['version']}")
            if "topology" in tpu_info:
                print(f"TPU Topology: {tpu_info['topology']}")


# Define a utility group
@app.group()
def util():
    """Utility commands"""
    pass


@util.command()
def check_dependencies():
    """Check if all dependencies are installed"""
    import importlib.util

    # Define dependencies to check
    dependencies = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
        ("huggingface_hub", "Hugging Face Hub"),
        ("rich", "Rich"),
        ("psutil", "PSUtil"),
        ("bitsandbytes", "BitsAndBytes"),
        ("llama_cpp", "LLAMA CPP Python"),
    ]

    # Check each dependency
    if RICH_AVAILABLE:
        table = Table(title="Dependency Check")
        table.add_column("Package", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Version", style="yellow")

        for package, name in dependencies:
            spec = importlib.util.find_spec(package)
            if spec is not None:
                try:
                    module = importlib.import_module(package)
                    version = getattr(module, "__version__", "Unknown")
                    table.add_row(name, "[green]Installed[/green]", version)
                except ImportError:
                    table.add_row(name, "[yellow]Found but not importable[/yellow]", "N/A")
            else:
                table.add_row(name, "[red]Not installed[/red]", "N/A")

        console.print(table)
    else:
        print("Dependency Check:")
        for package, name in dependencies:
            spec = importlib.util.find_spec(package)
            if spec is not None:
                try:
                    module = importlib.import_module(package)
                    version = getattr(module, "__version__", "Unknown")
                    print(f"{name}: Installed (version: {version})")
                except ImportError:
                    print(f"{name}: Found but not importable")
            else:
                print(f"{name}: Not installed")


@util.command()
@option("--host", help="Host to bind the benchmark server to", default="localhost")
@option("--port", "-p", help="Port to bind the benchmark server to", type=int, default=8080)
@option("--model", "-m", help="Model to benchmark", default="HelpingAI/HelpingAI-15B")
@option("--num-requests", "-n", help="Number of requests to send", type=int, default=10)
@option("--concurrent", "-c", help="Number of concurrent requests", type=int, default=1)
@option("--prompt", help="Prompt to use for benchmarking", default="Hello, how are you?")
@option("--max-tokens", help="Maximum number of tokens to generate", type=int, default=100)
def benchmark(host, port, model, num_requests, concurrent, prompt, max_tokens):
    """Benchmark the Inferno server"""
    import time
    import requests
    import json
    import statistics
    from concurrent.futures import ThreadPoolExecutor

    console.print(f"Benchmarking Inferno server at {host}:{port}")
    console.print(f"Model: {model}")
    console.print(f"Number of requests: {num_requests}")
    console.print(f"Concurrent requests: {concurrent}")
    console.print(f"Prompt: {prompt}")
    console.print(f"Max tokens: {max_tokens}")

    # Define the request function
    def make_request():
        start_time = time.time()
        try:
            response = requests.post(
                f"http://{host}:{port}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            tokens = len(result.get("choices", [{}])[0].get("text", "").split())
            elapsed = time.time() - start_time
            return {
                "success": True,
                "elapsed": elapsed,
                "tokens": tokens,
                "tokens_per_second": tokens / elapsed if elapsed > 0 else 0
            }
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "success": False,
                "elapsed": elapsed,
                "error": str(e)
            }

    # Run the benchmark
    results = []
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]

        if RICH_AVAILABLE:
            from rich.progress import Progress
            with Progress() as progress:
                task = progress.add_task("Running benchmark...", total=num_requests)
                for future in futures:
                    results.append(future.result())
                    progress.update(task, advance=1)
        else:
            for i, future in enumerate(futures, 1):
                results.append(future.result())
                print(f"Progress: {i}/{num_requests}")

    # Calculate statistics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if successful:
        latencies = [r["elapsed"] for r in successful]
        tokens_per_second = [r["tokens_per_second"] for r in successful]

        if RICH_AVAILABLE:
            table = Table(title="Benchmark Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Total Requests", str(num_requests))
            table.add_row("Successful", str(len(successful)))
            table.add_row("Failed", str(len(failed)))
            table.add_row("Success Rate", f"{len(successful)/num_requests*100:.2f}%")

            if latencies:
                table.add_row("Min Latency", f"{min(latencies):.4f}s")
                table.add_row("Max Latency", f"{max(latencies):.4f}s")
                table.add_row("Mean Latency", f"{statistics.mean(latencies):.4f}s")
                table.add_row("Median Latency", f"{statistics.median(latencies):.4f}s")

            if tokens_per_second:
                table.add_row("Min Tokens/sec", f"{min(tokens_per_second):.2f}")
                table.add_row("Max Tokens/sec", f"{max(tokens_per_second):.2f}")
                table.add_row("Mean Tokens/sec", f"{statistics.mean(tokens_per_second):.2f}")
                table.add_row("Median Tokens/sec", f"{statistics.median(tokens_per_second):.2f}")

            console.print(table)
        else:
            print("\nBenchmark Results:")
            print(f"Total Requests: {num_requests}")
            print(f"Successful: {len(successful)}")
            print(f"Failed: {len(failed)}")
            print(f"Success Rate: {len(successful)/num_requests*100:.2f}%")

            if latencies:
                print(f"Min Latency: {min(latencies):.4f}s")
                print(f"Max Latency: {max(latencies):.4f}s")
                print(f"Mean Latency: {statistics.mean(latencies):.4f}s")
                print(f"Median Latency: {statistics.median(latencies):.4f}s")

            if tokens_per_second:
                print(f"Min Tokens/sec: {min(tokens_per_second):.2f}")
                print(f"Max Tokens/sec: {max(tokens_per_second):.2f}")
                print(f"Mean Tokens/sec: {statistics.mean(tokens_per_second):.2f}")
                print(f"Median Tokens/sec: {statistics.median(tokens_per_second):.2f}")
    else:
        console.print("[red]No successful requests![/red]")

    if failed:
        console.print("\n[yellow]Failed requests:[/yellow]")
        for i, failure in enumerate(failed, 1):
            console.print(f"  {i}. Error: {failure.get('error', 'Unknown error')}")


# Define the main entry point
def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI"""
    return app.run(args)


if __name__ == "__main__":
    sys.exit(main())