# Build the onboarding deck to PDF with Marp.
# Requires marp-cli (via npx, no install) and a Chromium/Chrome for PDF export:
#   make onboarding.pdf
# marp-cli: https://github.com/marp-team/marp-cli

MARP ?= npx --yes @marp-team/marp-cli

onboarding.pdf: onboarding.md
	$(MARP) onboarding.md --pdf --allow-local-files -o onboarding.pdf

# HTML variant (no browser needed for viewing, but PDF is the deliverable)
onboarding.html: onboarding.md
	$(MARP) onboarding.md --html --allow-local-files -o onboarding.html

.PHONY: clean
clean:
	rm -f onboarding.pdf onboarding.html
