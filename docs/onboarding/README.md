# Onboarding deck

`onboarding.md` is the source for the new-member onboarding presentation
(Marp — Markdown slides). It's a sparse, presenter-driven deck (~18 slides):
the conceptual overview **plus** the clone → env → train → evaluate walkthrough,
handing off to the repo docs for detail.

## Build the PDF

Requires [marp-cli](https://github.com/marp-team/marp-cli) (Node) and a
Chromium/Chrome for PDF export:

```bash
make onboarding.pdf          # → onboarding.pdf   (needs node + a browser)
# or directly:
npx --yes @marp-team/marp-cli onboarding.md --pdf --allow-local-files -o onboarding.pdf
```

`make onboarding.html` produces an HTML variant if you don't want a PDF.

> A built `onboarding.pdf` is committed for convenience; regenerate it with
> `make onboarding.pdf` after editing `onboarding.md` (needs a Node + browser
> toolchain). Speaker detail is in the HTML-comment presenter notes on each
> slide.

## Before presenting

- Contact on the "Where to get help" slide: Alexander Wikner
  (awikner@uchicago.edu) — update if it changes.
- Update the supported-model status if the AMIP diffusion path graduates from
  experimental.
