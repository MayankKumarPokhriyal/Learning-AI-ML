# Security Policy

## Supported versions

Only the `main` branch is maintained. Fixes land there first; learners pick them up by merging `main` into their `learning` branch.

## Reporting a vulnerability

Please **don't open a public issue** for security problems. Report them privately:

1. Use GitHub's **Report a vulnerability** button on the repository's **Security** tab, if it is available, or
2. Send a direct message to the maintainer on [LinkedIn](https://www.linkedin.com/in/mayank-kumar-pokhriyal/).

Include the file or notebook, what an attacker could do, and steps to reproduce. You'll get an acknowledgement as soon as possible, and credit in the fix if you'd like it.

Examples that count as security issues: a notebook or capstone project that leaks secrets, a sandbox or guardrail example that is described as safe but can be escaped in a way the notebook doesn't disclose, or a dependency pin with a known critical vulnerability.

## Safety notes for learners

- **You run code locally.** Notebooks execute Python, start local servers and (in some modules) run code written by an LLM. Read a cell before running it, especially in the agent notebooks.
- **The course never needs API keys.** If you choose to use a hosted provider, keep keys in environment variables — never in a notebook, and never commit them.
- **Sandbox demos are teaching tools.** The notebooks state what each sandbox does and does not protect against; a subprocess or restricted interpreter is not a security boundary for untrusted production workloads.
- **Local servers bind to `127.0.0.1`.** Keep it that way unless you know what you are exposing.
