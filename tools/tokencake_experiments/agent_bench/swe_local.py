# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce official SWE task recipes locally and reuse official grading."""

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

import regex as re
import yaml

from .inputs import digest
from .transport import write_json

CONDA_PYPI_NAMES = {
    "matplotlib-base": "matplotlib",
    "python-tzdata": "tzdata",
    "pyqt": "PyQt5",
    "pyqtwebengine": "PyQtWebEngine",
    "brotli-python": "Brotli",
    "numpy-base": "numpy",
}

EVALUATION_LOCALE_SETUP = "sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen"


def evaluation_locale_setup(original: str) -> list[str]:
    """Recognize frozen shell setup without interpreting test-patch heredocs."""
    commands = []
    heredoc = None
    for line in original.splitlines():
        if heredoc is not None:
            if line == heredoc:
                heredoc = None
            continue
        match = re.search(r"<<'?(\w+)'?", line)
        if match:
            heredoc = match[1]
        if "/etc/locale.gen" in line or re.search(r"\blocale-gen\b", line):
            if line != EVALUATION_LOCALE_SETUP:
                raise ValueError("Evaluation locale setup needs an explicit adaptation")
            commands.append(line)
    if heredoc is not None:
        raise ValueError("Unterminated evaluation-script heredoc")
    return commands


def task_environment(platform: Path, directory: Path) -> dict[str, str]:
    directory = directory.resolve()
    activation = directory / "conda-activation.json"
    activated = json.loads(activation.read_text()) if activation.exists() else {}
    locale_path = directory / ".venv/lib/locale"
    if locale_path.is_dir():
        activated["LOCPATH"] = str(locale_path)
    return activated | {
        "PATH": activated.get(
            "PATH",
            f"{directory}/.venv/bin:/root/.local/bin:/usr/local/bin:/usr/bin:/bin",
        ),
        "VIRTUAL_ENV": str(directory / ".venv"),
        "PYTHONPATH": str(directory / "repo"),
        "UV_CACHE_DIR": str(platform / "cache/uv"),
        "UV_PYTHON_INSTALL_DIR": str(platform / "cache/python"),
        "UV_LINK_MODE": "copy",
        "CONDA_PKGS_DIRS": str(platform / "cache/conda-pkgs"),
        "CONDA_ENVS_PATH": str(platform / "environments/conda-registry"),
        "XDG_CACHE_HOME": str(directory / "cache"),
        "TMPDIR": str(directory / "tmp"),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def recipe(task_source: Path) -> dict:
    dockerfile = task_source / "Dockerfile"
    source = dockerfile.read_text()
    match = re.search(
        r"cat <<'(?P<end>EOF_\w+)' > /root/environment.yml\n"
        r"(?P<yaml>.*?)\n(?P=end)",
        source,
        re.DOTALL,
    )
    if match is None:
        raise ValueError(f"No frozen environment export in {dockerfile}")
    exported = yaml.safe_load(match["yaml"])
    packages, native = {}, []
    python_version = None
    for item in exported["dependencies"]:
        if isinstance(item, dict):
            for requirement in item["pip"] or []:
                name = re.split(r"[=<>!~\[]", requirement, maxsplit=1)[0]
                packages[name.lower().replace("_", "-")] = requirement
            continue
        name, version, build = item.split("=", 2)
        if name == "python":
            python_version = version
        elif name == "python_abi":
            native.append(item)
        elif build.startswith("py") or name in ("pip", "setuptools", "wheel"):
            mapped = CONDA_PYPI_NAMES.get(name, name)
            packages[mapped.lower().replace("_", "-")] = f"{mapped}=={version}"
        else:
            native.append(item)
    if python_version is None:
        raise ValueError("Official recipe does not specify Python")
    marker = 'echo "Current environment: $CONDA_DEFAULT_ENV"\ncd /testbed\n'
    if marker not in source or "\n# Configure git" not in source:
        raise ValueError(f"Unrecognized installation block in {dockerfile}")
    installation = source.split(marker, 1)[1].split("\n# Configure git", 1)[0]
    locale_setup = [
        line
        for line in installation.splitlines()
        if "/etc/locale.gen" in line or re.match(r"locale-gen(?:\s|$)", line)
    ]
    locales = []
    for line in locale_setup:
        if line.startswith("locale-gen "):
            locales.extend(shlex.split(line)[1:])
    if locale_setup and (not locales or any(name != "en_US.UTF-8" for name in locales)):
        raise ValueError("Task locale setup needs an explicit local adaptation")
    evaluation = task_source / "eval.sh"
    evaluation_setup = evaluation_locale_setup(
        evaluation.read_text() if evaluation.exists() else ""
    )
    if evaluation_setup:
        locales.append("en_US.UTF-8")
    system_setup = [
        line
        for line in installation.splitlines()
        if re.search(r"\b(?:apt-get|apt|dnf|yum)\s", line) or line in locale_setup
    ]
    return {
        "dockerfile": str(dockerfile),
        "dockerfile_sha256": digest(dockerfile),
        "python": python_version,
        "conda_environment": exported,
        "requirements": sorted(packages.values()),
        "native_conda_packages": native,
        "python_conda_package": next(
            item
            for item in exported["dependencies"]
            if isinstance(item, str) and item.startswith("python=")
        ),
        "original_installation": installation,
        "system_setup_commands": system_setup,
        "evaluation_script": str(evaluation) if evaluation.exists() else None,
        "evaluation_script_sha256": digest(evaluation) if evaluation.exists() else None,
        "evaluation_locale_setup_commands": evaluation_setup,
        "locales": sorted(set(locales)),
        "local_installation": "\n".join(
            line for line in installation.splitlines() if line not in system_setup
        ),
        "environment_difference": (
            "uv-managed CPython and PyPI packages at the exported versions; "
            "native libraries come from the host and wheel distributions. "
            "Image-level system package installation/upgrade is not run on the host. "
            "Base/reference test validation is required per task."
        ),
    }


def legacy_interpreter(platform: Path, details: dict) -> Path:
    """Supply the exact historical binary; uv still manages each task venv."""
    version = details["python"]
    if tuple(map(int, version.split(".")[:2])) < (3, 6):
        raise ValueError(f"Python {version} requires --environment-manager miniconda")
    prefix = platform / "cache/legacy-python" / version
    binary = prefix / "bin/python"
    if (prefix / "identity.json").exists():
        return binary
    prefix.mkdir(parents=True, exist_ok=True)
    packages = [details["python_conda_package"]]
    for package in details["native_conda_packages"]:
        if package.split("=", 1)[0] in (
            "openssl",
            "libffi",
            "ncurses",
            "readline",
            "sqlite",
            "tk",
            "xz",
            "zlib",
        ):
            packages.append(package)
    archives = platform / "cache/conda-binaries"
    archives.mkdir(parents=True, exist_ok=True)
    identities = []
    for package in packages:
        filename = package.replace("=", "-") + ".tar.bz2"
        url = "https://repo.anaconda.com/pkgs/main/linux-64/" + filename
        archive = archives / filename
        if not archive.exists():
            temporary = archive.with_suffix(".download")
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                temporary.open("wb") as stream,
            ):
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            temporary.rename(archive)
        with tarfile.open(archive) as package_tar:
            members = [
                member for member in package_tar if not member.name.startswith("info/")
            ]
            package_tar.extractall(prefix, members=members, filter="data")
        identities.append({"url": url, "sha256": digest(archive)})
    write_json(prefix / "identity.json", {"python": version, "packages": identities})
    return binary


def prepare_tox_interpreters(task_source: Path, directory: Path):
    """Preserve the task venv when tox-current-env creates its fake Python env.

    Its symlinks lose pyvenv.cfg when the active interpreter is from a venv.
    Exec wrappers preserve that interpreter's prefix and .pth processing while
    retaining the official tox command, options, and tests.
    """
    original = (task_source / "eval.sh").read_text()
    for name in re.findall(r"tox\s+--current-env\s+-e\s*(\w+)", original):
        binary_dir = directory / "repo/.tox" / name / "bin"
        binary_dir.mkdir(parents=True, exist_ok=True)
        version = json.loads((directory / "recipe.json").read_text())["python"]
        minor = ".".join(version.split(".")[:2])
        for suffix in ("", "3", minor):
            path = binary_dir / f"python{suffix}"
            if path.exists() or path.is_symlink():
                path.unlink()
            path.write_text(
                "#!/bin/sh\nexec "
                + shlex.quote(str(directory / ".venv/bin/python"))
                + ' "$@"\n'
            )
            path.chmod(0o755)


def replace_install(
    command: str, python: Path, *, legacy: bool = False, use_uv: bool = True
) -> str:
    # Historical Miniconda interpreters below 3.6 cannot be launched through uv.
    # Their frozen installer still runs only via the task's explicit interpreter.
    replacement = f"uv pip install --python {shlex.quote(str(python))}"
    if not use_uv:
        replacement = (
            f"{shlex.quote(str(python))} -m pip install --disable-pip-version-check"
        )
    elif legacy:
        # CPython 3.6's final setuptools lacks build_editable. Use uv to run
        # the task venv's explicit interpreter and its pinned legacy installer.
        # This never invokes a system Python or a bare pip executable.
        quoted = shlex.quote(str(python))
        replacement = (
            f"uv run --no-project --python {quoted} {quoted} "
            "-m pip install --disable-pip-version-check"
        )
    command = re.sub(r"(?m)^python -m pip install\b", replacement, command)
    command = re.sub(r"(?m)^pip install\b", replacement, command)
    command = re.sub(r"(?m)^python(?:[23])?(?=\s)", shlex.quote(str(python)), command)
    if use_uv:
        command = command.replace(" --no-use-pep517", "")
    return command


def task_pip_command(directory: Path, version: str, action: str) -> list[str]:
    python = str(directory / ".venv/bin/python")
    if tuple(map(int, version.split(".")[:2])) < (3, 6):
        command = [python, "-m", "pip", action, "--disable-pip-version-check"]
        return command + (["--all"] if action == "freeze" else [])
    return ["uv", "pip", action, "--python", python]


def command_run(command, cwd: Path, env: dict, log: Path, timeout=1800):
    started = time.monotonic()
    with log.open("x") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=os.environ | env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    return {"exit_code": code, "duration_s": time.monotonic() - started}


def prepare_locales(platform: Path, directory: Path, details: dict):
    if not details.get("locales"):
        return
    if details["locales"] != ["en_US.UTF-8"]:
        raise ValueError("Unsupported task locale configuration")
    locale_path = directory / ".venv/lib/locale"
    locale_path.mkdir(parents=True, exist_ok=True)
    command = [
        "localedef",
        "--no-archive",
        "-i",
        "en_US",
        "-f",
        "UTF-8",
        str(locale_path / "en_US.utf8"),
    ]
    result = command_run(
        command,
        directory,
        task_environment(platform, directory),
        directory / "logs/locale.log",
        timeout=60,
    )
    if result["exit_code"]:
        raise RuntimeError("Task-local en_US.UTF-8 generation failed")
    observed = subprocess.check_output(
        [
            str(directory / ".venv/bin/python"),
            "-c",
            "import locale; print(locale.setlocale(locale.LC_ALL, 'en_US.UTF-8'))",
        ],
        cwd=directory,
        env=os.environ | task_environment(platform, directory),
        text=True,
        timeout=60,
    ).strip()
    write_json(
        directory / "locale-preparation.json",
        {"command": command, "observed_locale": observed, **result},
    )


def capture_prepared_state(directory: Path) -> dict:
    """Include ignored builds and installed packages in the clean grading state."""
    archive = directory / "prepared-state.tar"
    with tarfile.open(archive, "x") as stream:
        for name in ("repo", ".venv"):
            stream.add(directory / name, arcname=name)
    identity = {"path": str(archive), "sha256": digest(archive)}
    write_json(directory / "prepared-state.json", identity)
    return identity


def restore_prepared_state(directory: Path):
    identity = json.loads((directory / "prepared-state.json").read_text())
    archive = directory / "prepared-state.tar"
    if digest(archive) != identity["sha256"]:
        raise ValueError("Prepared environment snapshot changed")
    for name in ("repo", ".venv", "cache", "tmp"):
        path = directory / name
        if path.is_symlink():
            raise ValueError(f"Refusing to replace a symlink: {path}")
        if path.exists():
            shutil.rmtree(path)
    # These are locally created, hash-checked snapshots. The venv interpreter
    # intentionally links to the uv-managed interpreter outside this directory.
    with tarfile.open(archive) as stream:
        stream.extractall(directory, filter="fully_trusted")
    for name in ("cache", "tmp"):
        (directory / name).mkdir()


def prepare(
    platform: Path, task_source: Path, directory: Path, *, manager: str = "uv"
) -> dict:
    if directory.exists():
        raise FileExistsError(f"Use a fresh preparation directory: {directory}")
    started = time.time()
    try:
        return _prepare(platform, task_source, directory, manager=manager)
    except Exception as exc:
        if not (directory / "preparation.json").exists():
            write_json(
                directory / "preparation.json",
                {
                    "status": "environment_error",
                    "instance_id": task_source.name,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "started_at": started,
                    "ended_at": time.time(),
                },
            )
        raise


def repository_source(platform: Path, task: dict) -> str:
    cache = platform / "cache/swe-repositories" / task["instance_id"]
    if not cache.exists():
        return "origin"
    identity = json.loads((cache / "identity.json").read_text())
    if any(identity[key] != task[key] for key in ("repo", "base_commit")):
        raise ValueError(f"Repository cache does not match the frozen task: {cache}")
    repository = cache / "repo.git"
    commit = subprocess.check_output(
        ["git", "--git-dir", str(repository), "rev-parse", "refs/heads/frozen-base"],
        text=True,
    ).strip()
    if commit != task["base_commit"]:
        raise ValueError(f"Repository cache revision changed: {cache}")
    return str(repository)


def miniconda_command(
    platform: Path, task_source: Path, directory: Path, details: dict
):
    conda = Path("/root/miniconda3/bin/conda")
    if not conda.is_file():
        raise FileNotFoundError("The requested server Miniconda installation is absent")
    exported = details["conda_environment"]
    cache = platform / "cache/conda-locks" / task_source.name
    channels = []
    if cache.exists():
        identity = json.loads((cache / "identity.json").read_text())
        if identity["dockerfile_sha256"] != details["dockerfile_sha256"]:
            raise ValueError("Conda lock does not match the official task recipe")
        lock = cache / "conda-explicit.txt"
        if digest(lock) != identity["explicit_sha256"]:
            raise ValueError("Frozen Conda package lock changed")
        specs = directory / "conda-explicit.txt"
        shutil.copyfile(lock, specs)
        archives = platform / "cache/conda-archives" / task_source.name
        if archives.exists():
            local = ["@EXPLICIT"]
            for line in lock.read_text().splitlines():
                if not line.startswith("https://"):
                    continue
                url, expected = line.rsplit("#", 1)
                archive = archives / url.rsplit("/", 1)[1]
                if hashlib.md5(archive.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Local Conda archive checksum differs: {archive}")
                local.append(f"{archive.as_uri()}#{expected}")
            specs = directory / "conda-local-explicit.txt"
            specs.write_text("\n".join(local) + "\n")
    else:
        specs = directory / "conda-specs.txt"
        specs.write_text(
            "\n".join(
                item for item in exported["dependencies"] if isinstance(item, str)
            )
            + "\n"
        )
        channels = ["--override-channels", "--no-channel-priority"]
        for channel in exported["channels"]:
            channels += ["-c", channel]
    pip_requirements = [
        requirement
        for item in exported["dependencies"]
        if isinstance(item, dict)
        for requirement in item.get("pip", [])
    ]
    (directory / "requirements.txt").write_text("\n".join(pip_requirements) + "\n")
    return [
        str(conda),
        "create",
        "--yes",
        "--json",
        "--copy",
        "--no-default-packages",
        "--prefix",
        str(directory / ".venv"),
        "--file",
        str(specs),
        *channels,
    ]


def conda_process_environment(environment: dict) -> dict:
    # Conda uses its own Python and dependencies, not the task's source tree.
    return environment | {"PYTHONPATH": ""}


def record_conda_activation(platform: Path, directory: Path, details: dict):
    packages = {
        package["name"]: package
        for path in (directory / ".venv/conda-meta").glob("*.json")
        for package in [json.loads(path.read_text())]
    }
    for spec in details["conda_environment"]["dependencies"]:
        if isinstance(spec, str):
            name, version, build = spec.split("=", 2)
            actual = packages.get(name, {})
            if actual.get("version") != version or actual.get("build") != build:
                raise ValueError(f"Installed Conda package differs from recipe: {spec}")
    environment = conda_process_environment(
        {
            key: os.environ[key]
            for key in (
                "HOME",
                "PATH",
                "LANG",
                "LC_ALL",
                "SSL_CERT_FILE",
                "SSL_CERT_DIR",
            )
            if key in os.environ
        }
        | task_environment(platform, directory)
    )
    result = subprocess.check_output(
        [
            "/root/miniconda3/bin/conda",
            "run",
            "--prefix",
            str(directory / ".venv"),
            "--no-capture-output",
            str(directory / ".venv/bin/python"),
            "-c",
            "import json,os; print(json.dumps(dict(os.environ)))",
        ],
        text=True,
        env=environment,
        cwd=directory,
        timeout=60,
    )
    activated = json.loads(result)
    delta = {
        key: value
        for key, value in activated.items()
        if environment.get(key) != value and key not in ("PWD", "SHLVL", "_", "PS1")
    }
    pkgconfig = str(directory / ".venv/lib/pkgconfig")
    inherited_pkgconfig = activated.get("PKG_CONFIG_PATH")
    delta["PKG_CONFIG_PATH"] = (
        f"{pkgconfig}:{inherited_pkgconfig}" if inherited_pkgconfig else pkgconfig
    )
    write_json(directory / "conda-activation.json", delta)
    write_json(directory / "conda-packages.json", packages)


def _prepare(
    platform: Path, task_source: Path, directory: Path, *, manager: str = "uv"
) -> dict:
    """Create a new owned environment; never reset an existing user checkout."""
    if directory.exists():
        raise FileExistsError(f"Use a fresh preparation directory: {directory}")
    directory.mkdir(parents=True)
    for name in ("tmp", "cache", "repo", "logs"):
        (directory / name).mkdir()
    task = yaml.safe_load((task_source / "task.yaml").read_text())
    details = recipe(task_source)
    if manager not in ("uv", "miniconda"):
        raise ValueError(f"Unknown task environment manager: {manager}")
    details["environment_manager"] = manager
    if manager == "miniconda":
        details["environment_difference"] = (
            "Official frozen Conda versions/builds plus exported pip distributions; "
            "the shared host still supplies system tools. "
            "Reference validation required."
        )
    write_json(directory / "recipe.json", details)
    (directory / "requirements.txt").write_text(
        "\n".join(details["requirements"]) + "\n"
    )
    environment = task_environment(platform, directory)
    repo = directory / "repo"
    interpreter = details["python"]
    if manager == "uv" and tuple(map(int, interpreter.split(".")[:2])) < (3, 8):
        interpreter = str(legacy_interpreter(platform, details))
    commands = [
        ["git", "init", str(repo)],
        [
            "git",
            "-C",
            str(repo),
            "remote",
            "add",
            "origin",
            f"https://github.com/{task['repo']}.git",
        ],
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "http.version=HTTP/1.1",
            "fetch",
            "--depth",
            "1",
            repository_source(platform, task),
            task["base_commit"],
        ],
        ["git", "-C", str(repo), "checkout", "--detach", "FETCH_HEAD"],
        ["git", "-C", str(repo), "remote", "remove", "origin"],
        ["uv", "venv", str(directory / ".venv"), "--python", interpreter]
        if manager == "uv"
        else miniconda_command(platform, task_source, directory, details),
        task_pip_command(directory, details["python"], "install")
        + [
            "--no-deps",
            "-r",
            str(directory / "requirements.txt"),
        ],
    ]
    records = []
    try:
        for index, command in enumerate(commands):
            if manager == "miniconda" and index == 5:
                # Conda 24 can race on shared .partial downloads across processes.
                # Serialize package-cache writes, leaving agent execution parallel.
                with (platform / "cache/conda-install.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    record = command_run(
                        command,
                        directory,
                        conda_process_environment(environment),
                        directory / f"logs/setup-{index}.log",
                    )
            else:
                record = command_run(
                    command,
                    directory,
                    environment,
                    directory / f"logs/setup-{index}.log",
                )
            records.append({"command": command, **record})
            if record["exit_code"]:
                raise RuntimeError(f"Setup command {index} failed: {command}")
            if manager == "miniconda" and index == 5:
                record_conda_activation(platform, directory, details)
                environment = task_environment(platform, directory)
        prepare_locales(platform, directory, details)
        environment = task_environment(platform, directory)
        installation = replace_install(
            details["local_installation"],
            directory / ".venv/bin/python",
            legacy=details["python"].startswith("3.6."),
            use_uv=tuple(map(int, details["python"].split(".")[:2])) >= (3, 6),
        )
        (directory / "install.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\n" + installation + "\n"
        )
        record = command_run(
            ["bash", str(directory / "install.sh")],
            repo,
            environment,
            directory / "logs/install.log",
        )
        records.append({"command": ["bash", str(directory / "install.sh")], **record})
        if record["exit_code"]:
            raise RuntimeError("Project installation failed")
        # Match official image preparation: source setup edits form the initial
        # working state and must not be submitted as the agent's patch.
        subprocess.run(
            ["git", "add", "-u"], cwd=repo, check=True, env=os.environ | environment
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=SWE-bench",
                "-c",
                "user.email=setup@swebench.com",
                "commit",
                "--allow-empty",
                "-m",
                "SWE-bench local environment preparation",
            ],
            cwd=repo,
            check=True,
            env=os.environ | environment,
            capture_output=True,
        )
        freeze = subprocess.check_output(
            task_pip_command(directory, details["python"], "freeze"),
            cwd=directory,
            text=True,
            env=os.environ | environment,
        )
        (directory / "installed.txt").write_text(freeze)
        prepare_tox_interpreters(task_source, directory)
        snapshot = capture_prepared_state(directory)
        result = {
            "status": "prepared",
            "environment_manager": manager,
            "instance_id": task["instance_id"],
            "base_commit": task["base_commit"],
            "prepared_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "records": records,
            "environment": environment,
            "snapshot": snapshot,
        }
    except Exception as exc:
        result = {
            "status": "environment_error",
            "instance_id": task["instance_id"],
            "records": records,
            "error": str(exc),
        }
        write_json(directory / "preparation.json", result)
        raise
    write_json(directory / "preparation.json", result)
    return result


def evaluation_script(original: str, directory: Path) -> str:
    from swebench.harness.utils import parse_eval_script, record_test_exit_code

    locale_setup = evaluation_locale_setup(original)
    details = json.loads((directory / "recipe.json").read_text())
    if locale_setup and (
        "en_US.UTF-8" not in details.get("locales", [])
        or not (directory / ".venv/lib/locale/en_US.utf8/LC_CTYPE").is_file()
    ):
        raise ValueError("Evaluation locale was not prepared; use a fresh environment")
    lines = []
    heredoc = None
    for line in original.splitlines():
        if heredoc is not None:
            lines.append(line)
            if line == heredoc:
                heredoc = None
            continue
        if line.startswith(
            ("source /opt/miniconda3/", "conda activate ", "git config --global ")
        ):
            continue
        if line in locale_setup:
            # prepare_locales generated this locale inside the task snapshot.
            # Keep the official LANG/LC_ALL exports and use task_environment's
            # LOCPATH instead of modifying the host's locale configuration.
            continue
        if re.search(r"\b(?:apt-get|apt|dnf|yum)\s", line):
            raise ValueError(
                "System package setup in an evaluation script needs local adaptation"
            )
        match = re.search(r"<<'?(\w+)'?", line)
        if match:
            heredoc = match[1]
        translated = line.replace("/testbed", shlex.quote(str(directory / "repo")))
        version = details["python"]
        lines.append(
            replace_install(
                translated,
                directory / ".venv/bin/python",
                legacy=version.startswith("3.6."),
                use_uv=tuple(map(int, version.split(".")[:2])) >= (3, 6),
            )
        )
    if heredoc is not None:
        raise ValueError("Unterminated evaluation-script heredoc")
    script = "\n".join(lines)
    # Preserve the upstream capture of the actual test command's exit code.
    return (
        "\n".join(
            [
                "#!/bin/bash",
                "set -uxo pipefail",
                *record_test_exit_code(parse_eval_script(script)),
            ]
        )
        + "\n"
    )


def grade(
    platform: Path, task: dict, directory: Path, prediction: dict, output: Path
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        with (directory / "grading.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = _grade(platform, task, directory, prediction, output)
    except Exception as exc:
        result = {
            "status": "grading_timeout"
            if isinstance(exc, subprocess.TimeoutExpired)
            else "grading_error",
            "resolved": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
    result.update(
        {
            "instance_id": task["instance_id"],
            "environment": "custom_local_"
            + (
                json.loads((directory / "recipe.json").read_text()).get(
                    "environment_manager", "uv"
                )
                if (directory / "recipe.json").exists()
                else "unknown"
            ),
            "started_at": started,
            "ended_at": time.time(),
            "duration_s": time.time() - started,
            "runtime_environment": task_environment(platform, directory),
        }
    )
    write_json(output / "report.json", result)
    return result


def apply_model_patch(repo: Path, environment: dict, patch_path: Path, output: Path):
    from swebench.harness.run_evaluation import GIT_APPLY_CMDS

    attempts = []
    for index, command in enumerate(GIT_APPLY_CMDS):
        if index:
            # Match the frozen official harness: --reject may leave partial
            # changes. Preserve the official reset commands and retry behavior.
            for step, reset in enumerate(
                (["git", "checkout", "--", "."], ["git", "clean", "-fd"])
            ):
                command_run(
                    reset, repo, environment, output / f"apply-reset-{index}-{step}.log"
                )
        argv = [*shlex.split(command), str(patch_path)]
        result = command_run(argv, repo, environment, output / f"apply-{index}.log")
        attempts.append({"command": argv, **result})
        if not result["exit_code"]:
            return {"applied": True, "attempts": attempts}
    # The official chain can apply every hunk but still return nonzero.
    argv = ["git", "apply", "--check", "--reverse", str(patch_path)]
    result = command_run(argv, repo, environment, output / "apply-reverse-check.log")
    attempts.append({"command": argv, **result})
    return {"applied": result["exit_code"] == 0, "attempts": attempts}


def _grade(
    platform: Path, task: dict, directory: Path, prediction: dict, output: Path
) -> dict:
    from swebench.harness.grading import get_eval_report, get_logs_eval
    from swebench.harness.utils import make_test_spec

    preparation = json.loads((directory / "preparation.json").read_text())
    if preparation["status"] != "prepared":
        raise ValueError("Grading requires a prepared environment")
    repo = directory / "repo"
    environment = task_environment(platform, directory)
    # Reset the entire prepared state, including ignored compiled extensions,
    # installed dependencies and caches that git reset/clean would leave behind.
    restore_prepared_state(directory)
    patch_path = output / "prediction.patch"
    patch_path.write_text(prediction["model_patch"])
    if prediction["model_patch"]:
        applied = apply_model_patch(repo, environment, patch_path, output)
        write_json(output / "patch-application.json", applied)
        if not applied["applied"]:
            result = {
                "instance_id": task["instance_id"],
                "resolved": False,
                "status": "patch_apply_error",
                **applied,
            }
            return result
    script = output / "eval.sh"
    script.write_text(evaluation_script(task["eval_script"], directory))
    log = output / "test_output.txt"
    execution = command_run(["bash", str(script)], repo, environment, log, timeout=1800)
    report = get_eval_report(
        make_test_spec(task), prediction, str(log), include_tests_status=True
    )
    parsed_tests, found = get_logs_eval(make_test_spec(task), str(log))
    result = {
        "status": "graded" if found else "test_output_missing",
        "resolved": bool(found and report[task["instance_id"]]["resolved"]),
        "execution": execution,
        "official_report": report,
        "parsed_tests": parsed_tests,
        "test_output_found": found,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("recipe", "prepare", "grade"))
    parser.add_argument("--platform", required=True, type=Path)
    parser.add_argument("--task-source", required=True, type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--task-json", type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--environment-manager", choices=("uv", "miniconda"), default="uv"
    )
    args = parser.parse_args()
    if args.action == "recipe":
        write_json(args.directory, recipe(args.task_source))
    elif args.action == "prepare":
        print(
            json.dumps(
                prepare(
                    args.platform,
                    args.task_source,
                    args.directory,
                    manager=args.environment_manager,
                )
            )
        )
    else:
        task = json.loads(args.task_json.read_text())
        prediction = json.loads(args.prediction.read_text())
        print(
            json.dumps(
                grade(args.platform, task, args.directory, prediction, args.output)
            )
        )


if __name__ == "__main__":
    main()
