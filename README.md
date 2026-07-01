# LLM Bench

A modern, highly secure web application and CLI suite designed for benchmarking remote LLM models across multiple quality categories using curated questions and rule-based evaluation.

Built with **Python 3.13**, **FastAPI**, **HTMX**, **Tailwind CSS**, and **SQLite**. Powered by **`uv`** for ultra-fast dependency management and containerized with secure **Docker** setups.

---

## Features

- **Rich Web Dashboard:** Visualize and compare test runs, view average scores, and inspect detailed category-by-category quality comparisons.
- **Dynamic Run Creation:** Run existing benchmark suites against configured LLM model configurations directly from the admin UI.
- **SSRF Protection (URL Guard):** Advanced DNS-resolution-based validation prevents Outbound Server-Side Request Forgery (SSRF) when connecting to custom model base URLs.
- **WAF & Reverse Proxy Ready:** Configured to automatically trust reverse-proxy headers (`X-Forwarded-*`) for secure cookie transmission and origin-based CSRF validation.
- **Hardened Security Defaults:** Implements response security headers, origin-based CSRF protection, and SameSite strict session cookies.

---

## Getting Started (Local Development)

### Prerequisites

- [Python 3.13+](https://www.python.org/)
- [uv](https://github.com/astral-sh/uv) (for package management)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd llm-bench
   ```

2. Copy the example environment file and configure local options:
   ```bash
   cp .env.example .env
   ```

3. Sync dependencies and build the local virtual environment:
   ```bash
   uv sync
   ```

### Running the Web Application

To launch the FastAPI development server with hot-reload enabled:

```bash
uv run uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

- **Admin Login:** Use the credentials configured in your `.env` (defaults to `admin` / `changeme`).

### Running the CLI Benchmark

You can also run benchmarks directly from the command line:

```bash
uv run python benchmark.py --category logical_reasoning
```

To see all CLI options:
```bash
uv run python benchmark.py --help
```

---

## Deployment with Docker Compose

For production deployments, LLM Bench is pre-configured to run behind a reverse proxy (such as Nginx, Traefik, or Caddy) with a Web Application Firewall (WAF).

### 1. Configure Production Environment Variables

Ensure your production environment contains strong, non-default values for core secrets. In production mode (`ENVIRONMENT=production`), the application will refuse to start if these are missing or insecure:

```ini
ENVIRONMENT=production
SECRET_KEY=generate-a-strong-random-key-here
ADMIN_PASSWORD=generate-a-strong-unique-password-here
```

### 2. Build and Start the Application

Start the multi-stage production Docker container:

```bash
docker compose up -d --build
```

### Security Architecture for Production

- **Reverse Proxy Header Trust:** The Docker image starts Uvicorn with `--proxy-headers` and `--forwarded-allow-ips=*` so it properly respects standard headers sent by the proxy.
- **User Separation:** The application inside the container drops privileges immediately, running as a non-root user `llmbench`.
- **Response Headers:** Adds `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and `Referrer-Policy: strict-origin-when-cross-origin` to protect clients.
- **CSRF Defense:** SameSite session cookies coupled with matching origin/referrer verification prevent cross-site request forgery.

---

## Code Quality & Formatting

To run the linter and formatters (using `ruff`):

```bash
uv run ruff check .
uv run ruff format .
```
