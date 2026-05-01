"""Main CLI app definition for HypoEvolve."""

from __future__ import annotations

import click

from . import commands, runs
from .commands import CLI_EXAMPLES, CLI_TIPS
from .display import _detect_version, _render_banner, _render_kv_section


class StyledGroup(click.Group):
    """Customize help output with the project's banner and grouped sections."""

    def get_help(self, ctx: click.Context) -> str:
        command_rows = []
        for name in self.list_commands(ctx):
            command = self.get_command(ctx, name)
            if command is None:
                continue
            command_rows.append((name, command.get_short_help_str()))

        help_sections = [
            _render_banner(),
            "Usage:\n  hypoevolve [OPTIONS] COMMAND [ARGS]...",
            _render_kv_section("Commands", command_rows),
            "Options:\n  -h, --help     Show this message and exit.\n  --version      Show the installed CLI version and exit.",
            CLI_EXAMPLES,
            CLI_TIPS,
        ]
        return "\n\n".join(help_sections)


@click.group(
    cls=StyledGroup,
    context_settings={
        "help_option_names": ["-h", "--help"],
        "max_content_width": 100,
    },
    invoke_without_command=True,
)
@click.version_option(version=_detect_version(), prog_name="hypoevolve")
@click.pass_context
def app(ctx: click.Context) -> None:
    """Modern CLI for HypoEvolve."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(1)


# Register top-level commands
app.add_command(commands.run)
app.add_command(commands.seed)
app.add_command(commands.render)
app.add_command(commands.inspect)
app.add_command(commands.doctor)

# Register runs subgroup
app.add_command(runs.runs)


def main(argv: list[str] | None = None) -> int:
    """Entry point that returns a shell-friendly exit code."""
    try:
        result = app.main(args=argv, prog_name="hypoevolve", standalone_mode=False)
        return 0 if result is None else int(result)
    except click.exceptions.Exit as exc:
        return exc.exit_code
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.Abort:
        click.echo("Aborted.", err=True)
        return 1
