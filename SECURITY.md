# Security Policy

## API keys

Never commit API keys to this repository.

Use environment variables, OS keyrings, or a proper secret manager in production.

Recommended environment variable names:

```text
OPENAI_API_KEY
ANTHROPIC_API_KEY
GEMINI_API_KEY
MISTRAL_API_KEY
XAI_API_KEY
DEEPSEEK_API_KEY
GROQ_API_KEY
TOGETHER_API_KEY
```

## AI-generated data

Bricks AI does not assume AI-generated content is true.

All generated training items should pass validation before being accepted. High-stakes domains should use stricter thresholds and should avoid personalized medical, legal, or financial advice.

## Unsafe training content

Contributions must not add pipelines intended to generate harmful content, malicious code, fraud instructions, or evasion techniques.

## Reporting vulnerabilities

Open a private security advisory if the repository is hosted on GitHub, or contact the maintainers through the project's listed security contact.
